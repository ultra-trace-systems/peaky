"""The TOF mass-only flag (peaky/assignment/mass_only.py): an annotation on a
TOF-class run's Assigned readings that no isotope line of their own supports.

What is pinned here, on synthetic ledgers at exact masses:
  * the line test -- a 13C child at its expected height keeps a reading
    unflagged; the reagent's own 81Br twin, a line at 5x its expectation, or a
    line in a file where the reading is only a Candidate do not; a brominated
    neutral's own 81Br line does only when the ion holds more Br than a reagent
    channel supplies (else it is the same ion, and line, as a reagent adduct);
  * the merged flag -- True / False on Assigned rows only, empty elsewhere and
    on every row off a TOF; known species exempt (a `known:` commit or the
    batch's known-species note); a re-read row finds its ion; the threshold
    changes the reason and the split, never which rows are flagged; nothing
    else on the merged ledger moves;
  * the batch writes it (merged_ledger.csv + batch_summary['tof_flag']); a
    non-default threshold reaches the cfg through run_batch / run_pooled_batches
    and through `peaky batch` / `pool`, and `peaky assign` stamps the sample's
    ledger, workbook and manifest at it;
  * the workbook, the PDF and the scorecard census show it, and a ledger
    without it renders as before.

Run: pytest tests/test_mass_only.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from peaky import cli
from peaky import pipeline as PL
from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import mass_only as MO
from peaky.assignment.passes import PassConfig
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.reporting import pdf_report as R
from peaky.reporting import report as RP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard as SC  # noqa: E402

D13 = ISO.ISOTOPE_SPACING["13C"]
D81 = ISO.ISOTOPE_SPACING["81Br"]
TOF = 9_000.0       # R(200) = 9 000 -> TOF class
ORBI = 120_000.0    # R(200) = 120 000 -> Orbitrap class


def _ledger(readings, kids=()) -> pd.DataFrame:
    """One file's ledger: `readings` = (peak_id, neutral, adduct, height, tier, method),
    `kids` = (peak_id, parent, label, ratio): an iso child at its label's exact
    spacing whose height is `ratio` x the parent's."""
    rows, mz_of, h_of = [], {}, {}
    for pid, nf, ad, h, _t, _m in readings:
        mz_of[pid], h_of[pid] = C.ion_mz(nf, ad), h
        rows.append((pid, mz_of[pid], h))
    for pid, parent, label, ratio in kids:
        shift = sum(ISO.ISOTOPE_SPACING[p] for p in label.split("+"))
        rows.append((pid, mz_of[parent] + shift, ratio * h_of[parent]))
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    for pid, nf, ad, _h, _t, method in readings:
        L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_score=0.9, compound_score=0.9,
                            ppm_error=0.1, pass_no=0 if method.startswith("known:") else 1, method=method,
                            confidence="High", commentary="stub")
    for pid, parent, label, _r in kids:
        L.attach_isotopologue(led, pid, parent, iso_label=label, iso_match_score=0.9)
    for pid, _nf, _ad, _h, tier, _m in readings:
        led.loc[led["peak_id"] == pid, "tier"] = tier
    return led


GRID = "cheminfo+grid"
# expected 13C line of a C10 ion: 10 x 1.07 % of the parent; one of one Br: 0.97x
F1 = _ledger([("a", "C10H16O5", "[M+Br]-", 1000.0, "Assigned", GRID),       # own 13C line in band
              ("b", "C10H16O6", "[M+Br]-", 1000.0, "Assigned", GRID),       # the reagent's 81Br twin only
              ("c", "C20H30O10", "[M+Br]-", 400.0, "Assigned", GRID),       # no line, m/z >= 350
              ("d", "HNO3", "[M+Br]-", 5000.0, "Assigned", "known:atmospheric"),
              ("e", "C10H16O7", "[M+Br]-", 1000.0, "Candidate", GRID),      # in band, but a Candidate here
              ("f", "CHBr3", "[M-H]-", 1000.0, "Assigned", GRID),           # own 81Br line, ion Br3 > 2
              ("g", "C12H20O8", "[M+Br]-", 500.0, "Assigned", GRID),
              ("h", "C9H14O4", "[M+Br]-", 300.0, "Candidate", GRID)],
             [("a13", "a", "13C", 0.107), ("b81", "b", "81Br", 0.97), ("e13", "e", "13C", 0.107),
              ("f81", "f", "81Br", EV.expected_ratio("81Br", "CBr3-"))])
F2 = _ledger([("a", "C10H16O5", "[M+Br]-", 900.0, "Assigned", GRID),
              ("e", "C10H16O7", "[M+Br]-", 1000.0, "Assigned", GRID),       # 13C at 5x its expectation
              ("g", "C12H20O8", "[M+Br]-", 500.0, "Assigned", GRID)],
             [("e13", "e", "13C", 0.535)])


def _merged() -> pd.DataFrame:
    rows = [("C10H16O5", "[M+Br]-", "Assigned", "f1,f2", ""),
            ("C10H16O6", "[M+Br]-", "Assigned", "f1", ""),
            ("C20H30O10", "[M+Br]-", "Assigned", "f1", ""),
            ("HNO3", "[M+Br]-", "Assigned", "f1", ""),
            ("C10H16O7", "[M+Br]-", "Assigned", "f1,f2", ""),
            ("CHBr3", "[M-H]-", "Assigned", "f1", ""),
            ("C12H20O8", "[M+Br]-", "Assigned", "f1,f2", MO.KNOWN_DECIDED + ": a listed species (C12H20O8 [M+Br]-)"),
            ("C9H14O4", "[M+Br]-", "Candidate", "f1", "")]
    return pd.DataFrame([dict(mz=C.ion_mz(nf, ad), neutral_formula=nf, adduct=ad, tier=t, srcs=s, tier_reason=n,
                              evidence_level="NA", claim="not assessed") for nf, ad, t, s, n in rows])


def _flag(merged, klass="tof", threshold=350.0):
    return MO.flag_merged(merged, {"f1": F1, "f2": F2}, klass=klass, threshold=threshold, halogen="Br",
                          log=lambda *a: None)


def _by(merged):
    return {(r.neutral_formula, r.adduct): (r.mass_only, r.mass_only_reason) for r in merged.itertuples()}


# --------------------------------------------------------------------------- the line test
def test_a_positive_line_is_an_own_element_child_at_its_expected_height():
    f = MO.pair_facts(F1, halogen="Br").set_index(["neutral_formula", "adduct"])
    assert f.loc[("C10H16O5", "[M+Br]-"), "positive"] and f.loc[("C10H16O5", "[M+Br]-"), "own_lines"] == "C"
    # the 81Br line of a Br-free neutral's bromide adduct is the reagent's twin: the ion holds the reagent
    assert not f.loc[("C10H16O6", "[M+Br]-"), "positive"]
    # a brominated neutral's own 81Br line speaks for the neutral when no reagent channel can put 3 Br on an ion
    assert f.loc[("CHBr3", "[M-H]-"), "positive"] and f.loc[("CHBr3", "[M-H]-"), "own_lines"] == "Br"
    assert not f.loc[("C20H30O10", "[M+Br]-"), "positive"]
    assert f.loc[("HNO3", "[M+Br]-"), "known"] and not f.loc[("C10H16O5", "[M+Br]-"), "known"]
    assert not f.loc[("C10H16O7", "[M+Br]-"), "assigned"]
    # a line at 5x its expectation is not this ion's line
    g = MO.pair_facts(F2, halogen="Br").set_index(["neutral_formula", "adduct"])
    assert g.loc[("C10H16O7", "[M+Br]-"), "assigned"] and not g.loc[("C10H16O7", "[M+Br]-"), "positive"]


def test_a_reagent_halogen_line_counts_only_beyond_what_a_reagent_channel_supplies():
    assert MO.reagent_supply("Br") == MO.reagent_supply("Cl") == 2
    assert MO.reagent_supply(None) == 0 and MO.reagent_supply("detect") == 0
    # one ion, two labels: C8H6BrNO4 [M+NO3]- == C8H6N2O7 [M+Br]- == C8H6BrN2O7-
    assert EV.ion_composition("C8H6BrNO4", "[M+NO3]-", None) == EV.ion_composition("C8H6N2O7", "[M+Br]-", None)
    as_cluster = _ledger([("p", "C8H6BrNO4", "[M+NO3]-", 1000.0, "Assigned", GRID),
                          ("q", "C10H20Br2", "[M+NO3]-", 1000.0, "Assigned", GRID),
                          ("r", "C5H9BrO3", "[M-H]-", 1000.0, "Assigned", GRID),
                          ("s", "C10H17BrO4", "[M+NO3]-", 1000.0, "Assigned", GRID)],
                         [("p81", "p", "81Br", EV.expected_ratio("81Br", "C8H6BrN2O7-")),
                          ("q81", "q", "81Br", EV.expected_ratio("81Br", "C10H20Br2NO3-")),
                          ("r81", "r", "81Br", EV.expected_ratio("81Br", "C5H8BrO3-")),
                          ("s81", "s", "81Br", EV.expected_ratio("81Br", "C10H17BrNO7-")),
                          ("s13", "s", "13C", EV.expected_ratio("13C", "C10H17BrNO7-"))])
    as_adduct = _ledger([("p", "C8H6N2O7", "[M+Br]-", 1000.0, "Assigned", GRID)],
                        [("p81", "p", "81Br", EV.expected_ratio("81Br", "C8H6BrN2O7-"))])
    br = MO.pair_facts(as_cluster, halogen="Br").set_index(["neutral_formula", "adduct"])
    # the scale sees an own 81Br line on each; on a bromide run the flag counts none of them alone ...
    for key in (("C8H6BrNO4", "[M+NO3]-"), ("C10H20Br2", "[M+NO3]-"), ("C5H9BrO3", "[M-H]-")):
        assert "Br" in br.loc[key, "scale_lines"].split("|") and not br.loc[key, "positive"], key
    # ... a 13C line beside it still counts
    assert br.loc[("C10H17BrO4", "[M+NO3]-"), "positive"] and br.loc[("C10H17BrO4", "[M+NO3]-"), "own_lines"] == "C"
    # ... and the reagent-adduct label of the same ion reads the same: not positive
    assert not MO.pair_facts(as_adduct, halogen="Br")["positive"].any()
    # without a reagent halogen (a nitrate run) the neutral's own Br line counts
    no = MO.pair_facts(as_cluster, halogen=None).set_index(["neutral_formula", "adduct"])
    assert no["positive"].all() and no.loc[("C10H20Br2", "[M+NO3]-"), "own_lines"] == "Br"
    assert MO.counted_lines("Br|C", "C8H6BrNO4", "[M+NO3]-", halogen="Br") == {"C"}
    assert MO.counted_lines("Br", "CHBr3", "[M-H]-", halogen="Br") == {"Br"}
    assert MO.counted_lines("Cl", "C2H3ClO2", "[M+Br]-", halogen="Br") == {"Cl"}
    # the batch flag follows: the two bromide-run readings flag, the 13C one does not
    merged = pd.DataFrame([dict(mz=C.ion_mz(nf, ad), neutral_formula=nf, adduct=ad, tier="Assigned", srcs="s",
                                tier_reason="")
                           for nf, ad in (("C8H6BrNO4", "[M+NO3]-"), ("C10H20Br2", "[M+NO3]-"),
                                          ("C10H17BrO4", "[M+NO3]-"))])
    MO.flag_merged(merged, {"s": as_cluster}, klass="tof", halogen="Br", log=lambda *a: None)
    assert merged[MO.COLUMN].tolist() == [True, True, False]
    MO.flag_merged(merged, {"s": as_cluster}, klass="tof", halogen=None, log=lambda *a: None)
    assert merged[MO.COLUMN].tolist() == [False, False, False]


# --------------------------------------------------------------------------- the merged flag
def test_the_merged_flag_marks_assigned_mass_only_rows_and_moves_nothing_else():
    merged = _merged()
    before = merged.copy()
    block = _flag(merged)
    by = _by(merged)
    assert by[("C10H16O5", "[M+Br]-")] == (False, by[("C10H16O5", "[M+Br]-")][1]) \
        and pd.isna(by[("C10H16O5", "[M+Br]-")][1])
    assert by[("C10H16O6", "[M+Br]-")][0] is True
    assert by[("C10H16O6", "[M+Br]-")][1].startswith("mass only: the reading rests on mass alone")
    assert by[("C20H30O10", "[M+Br]-")][0] is True
    assert by[("C20H30O10", "[M+Br]-")][1].startswith("mass only at m/z >= 350: a TOF's formula space is saturated")
    assert by[("HNO3", "[M+Br]-")][0] is False                     # pass-0 known species: exempt
    assert by[("C12H20O8", "[M+Br]-")][0] is False                 # the batch's known-species decision: exempt
    # in band only in a file where it is a Candidate; out of band where Assigned -> flagged
    assert by[("C10H16O7", "[M+Br]-")][0] is True
    assert by[("CHBr3", "[M-H]-")][0] is False
    assert pd.isna(by[("C9H14O4", "[M+Br]-")][0]) and pd.isna(by[("C9H14O4", "[M+Br]-")][1])
    # nothing but the two new columns
    pd.testing.assert_frame_equal(merged.drop(columns=list(MO.COLUMNS)), before)
    assert block["applied"] and block["instrument"] == "tof" and block["threshold_mz"] == 350.0
    assert (block["n_assigned"], block["n_flagged"], block["n_known"], block["n_positive"]) == (7, 3, 2, 2)
    # C12H20O8 [M+Br]- sits at m/z 371: at or above 350, exempt
    assert block["below"] == {"assigned": 5, "flagged": 2} and block["at_or_above"] == {"assigned": 2, "flagged": 1}
    assert MO.counts(merged) == {k: block[k] for k in ("threshold_mz", "n_assigned", "n_flagged", "below",
                                                       "at_or_above")}


@pytest.mark.parametrize("klass", ["orbitrap", None])
def test_off_a_tof_the_columns_exist_and_stay_empty(klass):
    merged = _merged()
    block = _flag(merged, klass=klass)
    assert set(MO.COLUMNS) <= set(merged.columns)
    assert merged[MO.COLUMN].isna().all() and merged[MO.REASON_COLUMN].isna().all()
    assert not block["applied"] and block["n_flagged"] == 0 and MO.counts(merged) is None


def test_the_threshold_moves_the_reason_and_the_split_never_the_flag():
    lo, hi = _merged(), _merged()
    b_lo, b_hi = _flag(lo, threshold=300.0), _flag(hi, threshold=600.0)
    assert lo[MO.COLUMN].tolist() == hi[MO.COLUMN].tolist()
    assert b_lo["n_flagged"] == b_hi["n_flagged"] == 3
    assert b_lo["at_or_above"]["flagged"] == 3 and b_hi["at_or_above"]["flagged"] == 0
    r = dict(zip(lo["neutral_formula"], lo[MO.REASON_COLUMN]))
    assert r["C10H16O6"].startswith("mass only at m/z >= 300")


def test_a_re_read_row_finds_its_ion_under_another_label():
    led = _ledger([("p", "C10H14O5", "[M+NH4]+", 1000.0, "Assigned", GRID)], [("p13", "p", "13C", 0.107)])
    merged = pd.DataFrame([dict(mz=C.ion_mz("C10H17NO5", "[M+H]+"), neutral_formula="C10H17NO5", adduct="[M+H]+",
                                tier="Assigned", srcs="s", tier_reason="")])
    MO.flag_merged(merged, {"s": led}, klass="tof", log=lambda *a: None)
    assert merged[MO.COLUMN].tolist() == [False]


def test_the_flag_reads_back_from_csv(tmp_path):
    merged = _merged()
    _flag(merged)
    merged.to_csv(tmp_path / "m.csv", index=False)
    back = pd.read_csv(tmp_path / "m.csv")
    assert MO.counts(back) == MO.counts(merged)
    assert MO.flagged(back).tolist() == MO.flagged(merged).tolist()


def test_the_single_sample_flag_reads_its_own_lines():
    led = F2.copy()
    block = MO.flag_ledger(led, klass="tof", halogen="Br")
    m0 = led[led["role"] == "M0"].set_index("neutral_formula")
    assert m0.loc["C10H16O5", MO.COLUMN] is True        # no line in THIS file
    assert m0.loc["C10H16O7", MO.COLUMN] is True        # a line out of band
    assert led.loc[led["role"] != "M0", MO.COLUMN].isna().all()
    assert block["n_assigned"] == 3 and block["n_flagged"] == 3
    led1 = F1.copy()
    MO.flag_ledger(led1, klass="tof", halogen="Br")
    m1 = led1[led1["role"] == "M0"].set_index("neutral_formula")
    assert m1.loc["HNO3", MO.COLUMN] is False and m1.loc["C10H16O5", MO.COLUMN] is False
    assert pd.isna(m1.loc["C10H16O7", MO.COLUMN])     # a Candidate is not judged


# --------------------------------------------------------------------------- the knob
def test_instrument_class_and_threshold_checks():
    assert MO.instrument_class(resolution=TOF) == "tof"
    assert MO.instrument_class(resolution=ORBI) == "orbitrap"
    assert MO.instrument_class(resolution={"coef": None}, instrument_types=["tof", "tof"]) == "tof"
    assert MO.instrument_class(instrument_types=["orbi"]) == "orbitrap"
    assert MO.instrument_class(instrument_types=["tof", "orbi"]) is None
    assert MO.instrument_class() is None
    for bad in (0, -1, float("nan"), float("inf"), "x", None):
        with pytest.raises(ValueError):
            MO.check_threshold(bad)
    assert PassConfig().tof_flag_mz == MO.DEFAULT_TOF_FLAG_MZ == 350.0
    assert MO.flag_mz(PassConfig(tof_flag_mz=420)) == 420.0 and MO.flag_mz(None) == 350.0
    assert PL.tof_flag_config(PassConfig(), 280).tof_flag_mz == 280.0
    assert PL.tof_flag_config(PassConfig(tof_flag_mz=410), None).tof_flag_mz == 410.0
    with pytest.raises(ValueError):
        PL.tof_flag_config(PassConfig(), 0)


def test_the_cli_takes_the_threshold_on_batch_pool_and_assign():
    p = cli.build_parser()
    assert p.parse_args(["batch", "--batch", "x", "--tof-flag-mz", "400"]).tof_flag_mz == 400.0
    assert p.parse_args(["pool", "--batches", "x", "--tof-flag-mz", "300"]).tof_flag_mz == 300.0
    assert p.parse_args(["assign", "--sample-id", "s", "--tof-flag-mz", "360"]).tof_flag_mz == 360.0
    assert p.parse_args(["batch", "--batch", "x"]).tof_flag_mz is None
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--batch", "x", "--tof-flag-mz", "0"])


class _Stop(Exception):
    """Cut a pipeline off at its first server call."""


def _stop(*a, **k):
    raise _Stop


def test_run_batch_and_run_pooled_batches_put_the_threshold_on_the_runs_cfg(monkeypatch, tmp_path):
    # the first server call (IO.connect) comes after the cfg is settled: the cfg the
    # assignment will get is the one handed in, carrying the threshold
    monkeypatch.setattr(PL.IO, "connect", _stop)
    cfg = PassConfig()
    with pytest.raises(_Stop):
        PL.run_batch(batch="b", base_out=str(tmp_path), tof_flag_mz=300, cfg=cfg, log=lambda *a: None)
    assert cfg.tof_flag_mz == 300.0
    cfg = PassConfig()
    with pytest.raises(_Stop):
        PL.run_pooled_batches(batches="b.*", base_out=str(tmp_path), tof_flag_mz=310, cfg=cfg,
                              log=lambda *a: None)
    assert cfg.tof_flag_mz == 310.0
    cfg = PassConfig(tof_flag_mz=420)
    with pytest.raises(_Stop):
        PL.run_batch(batch="b", base_out=str(tmp_path), cfg=cfg, log=lambda *a: None)
    assert cfg.tof_flag_mz == 420.0                      # None keeps the config's own


def test_peaky_batch_and_pool_hand_the_threshold_to_the_pipeline(monkeypatch, tmp_path):
    seen = {}

    def rec(name):
        def f(**kw):
            seen[name] = kw
            return {"ctx": SimpleNamespace(out_dir=str(tmp_path), run_id="rid")}
        return f

    monkeypatch.setattr(cli, "_require_creds", lambda: None)
    monkeypatch.setattr(PL, "run_batch", rec("batch"))
    monkeypatch.setattr(PL, "run_pooled_batches", rec("pool"))
    p = cli.build_parser()
    out = ["--out-dir", str(tmp_path), "--no-progress"]
    cli.cmd_batch(p.parse_args(["batch", "--batch", "B", "--tof-flag-mz", "300", *out]))
    cli.cmd_pool(p.parse_args(["pool", "--batches", "B.*", "--tof-flag-mz", "310", *out]))
    assert seen["batch"]["tof_flag_mz"] == 300.0 and seen["pool"]["tof_flag_mz"] == 310.0
    cli.cmd_batch(p.parse_args(["batch", "--batch", "B", *out]))
    assert seen["batch"]["tof_flag_mz"] is None


def test_peaky_assign_stamps_the_flag_at_its_threshold(monkeypatch, tmp_path):
    """`peaky assign` driven for real with assign.run stubbed: a TOF-class sample's
    ledger CSV, workbook and manifest carry the flag at --tof-flag-mz."""
    got = {}

    def fake_run(sid, context="ambient-air", *, cfg=None, **kw):
        got["cfg"] = cfg
        led = F1.copy()
        st = L.stats(led)
        st.update(resolution=TOF, reagent_halogen="Br", instrument_type="tof")
        return {"ledger": led, "stats": st, "context": context, "sample_id": sid, "prescan": {},
                "problems": [], "summaries": {}}

    monkeypatch.setattr(cli, "_require_creds", lambda: None)
    monkeypatch.setattr(A, "run", fake_run)
    args = cli.build_parser().parse_args(["assign", "--sample-id", "s1", "--reagent", "Br", "--no-progress",
                                          "--output-dir", str(tmp_path), "--tof-flag-mz", "300"])
    cli.cmd_assign(args)
    assert got["cfg"].tof_flag_mz == 300.0
    base = str(next(tmp_path.glob("s1_*_ledger.csv")))[:-len("_ledger.csv")]
    led = pd.read_csv(f"{base}_ledger.csv")
    m0 = led[(led["role"] == "M0") & (led["tier"] == "Assigned")].set_index("neutral_formula")
    assert MO.flagged(m0).to_dict() == {"C10H16O5": False, "C10H16O6": True, "C20H30O10": True, "HNO3": False,
                                        "CHBr3": False, "C12H20O8": True}
    assert m0.loc["C10H16O6", MO.REASON_COLUMN].startswith("mass only at m/z >= 300")
    tf = json.loads(Path(f"{base}_manifest.json").read_text())["stats"]["tof_flag"]
    assert tf["applied"] and tf["threshold_mz"] == 300.0 and tf["n_flagged"] == 3
    assert tf["below"] == {"assigned": 3, "flagged": 0} and tf["at_or_above"] == {"assigned": 3, "flagged": 3}
    summ = pd.read_excel(f"{base}_assignments.xlsx", sheet_name="Summary")
    summ["section"] = summ["section"].ffill()           # the sheet names a section on its first row only
    sec = summ[summ["section"] == "TOF mass-only flag"].set_index("metric")["value"]
    assert sec["below m/z 300"].startswith("0 of 3") and sec["at or above m/z 300"].startswith("3 of 3")
    assigned = pd.read_excel(f"{base}_assignments.xlsx", sheet_name="Assigned")
    assert MO.COLUMN in assigned.columns


# --------------------------------------------------------------------------- the batch writes it
def _batch_table(spec, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    return pd.DataFrame([dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t0 + pd.Timedelta(minutes=10 * i),
                              mz=float(mz), height=height)
                         for i, (sid, mzs) in enumerate(spec.items()) for mz in mzs])


def _run_batch(tmp_path, monkeypatch, *, resolving_power, cfg=None):
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    bg = list(range(100, 120))
    spec = {f"{k}{i}": bg + list(range(200 + 20 * i, 220 + 20 * i)) for i in range(4) for k in "ab"}
    pk = _batch_table(spec)

    def fake_assign(sid, context="ambient-air", **kw):
        led = _ledger([("a", "C10H16O5", "[M+Br]-", 1.0e5, "Assigned", GRID),
                       ("b", "C10H16O6", "[M+Br]-", 1.0e5, "Assigned", GRID)],
                      [("a13", "a", "13C", 0.107)])
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.5}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    kw = {"cfg": cfg} if cfg is not None else {}
    return AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path),
                  k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=resolving_power,
                  log=lambda *a: None, **kw)


def test_a_tof_batch_writes_the_flag_and_its_summary(tmp_path, monkeypatch):
    _run_batch(tmp_path, monkeypatch, resolving_power=TOF, cfg=PassConfig(tof_flag_mz=300.0))
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    by = dict(zip(merged["neutral_formula"], merged[MO.COLUMN]))
    assert by == {"C10H16O5": False, "C10H16O6": True}
    assert (merged["tier"] == "Assigned").all()
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tf = summ["tof_flag"]
    assert tf["applied"] and tf["instrument"] == "tof" and tf["threshold_mz"] == 300.0
    # m/z 295 sits below the run's 300, m/z 311 at or above it
    assert tf["n_assigned"] == 2 and tf["n_flagged"] == 1
    assert tf["below"] == {"assigned": 1, "flagged": 0} and tf["at_or_above"] == {"assigned": 1, "flagged": 1}
    assert list(summ).index("tof_flag") == list(summ).index("ion_only") + 1


def test_an_orbitrap_batch_has_the_columns_and_no_value(tmp_path, monkeypatch):
    _run_batch(tmp_path, monkeypatch, resolving_power=ORBI)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    assert set(MO.COLUMNS) <= set(merged.columns) and merged[MO.COLUMN].isna().all()
    tf = json.load(open(tmp_path / "batch_summary.json"))["tof_flag"]
    assert not tf["applied"] and tf["instrument"] == "orbitrap" and tf["n_flagged"] == 0


# --------------------------------------------------------------------------- the outputs show it
def test_the_workbook_shows_the_flag_only_where_the_ledger_carries_it(tmp_path):
    led = F1.copy()
    MO.flag_ledger(led, klass="tof", halogen="Br")
    sheets = RP.build_sheets(led, tof_flag_mz=350.0)
    a = sheets["Assigned"]
    cols = list(a.columns)
    assert cols.index(MO.COLUMN) == cols.index("tier_reason") + 1 and MO.REASON_COLUMN in cols
    s = sheets["Summary"]
    sec = s[s["section"] == "TOF mass-only flag"].set_index("metric")["value"]
    # one sample alone: C12H20O8 (m/z 371) has no batch decision to exempt it here
    assert sec["Assigned, flagged"].startswith("3 of 6") and sec["below m/z 350"].startswith("1 of 4")
    assert sec["at or above m/z 350"].startswith("2 of 2")
    # the run's threshold, not the package default, splits the Summary
    s300 = RP.build_sheets(led, tof_flag_mz=300.0)["Summary"]
    sec300 = s300[s300["section"] == "TOF mass-only flag"].set_index("metric")["value"]
    assert sec300["below m/z 300"].startswith("0 of 3") and sec300["at or above m/z 300"].startswith("3 of 3")
    assert "below m/z 350" not in sec300.index
    assert (sheets["Read me"]["section"] == "TOF mass-only flag").sum() == 2
    plain = RP.build_sheets(F1.copy())
    assert MO.COLUMN not in plain["Assigned"].columns
    assert not (plain["Summary"]["section"] == "TOF mass-only flag").any()
    assert not (plain["Read me"]["section"] == "TOF mass-only flag").any()
    off = F1.copy()
    MO.flag_ledger(off, klass="orbitrap")
    assert MO.COLUMN not in RP.build_sheets(off)["Assigned"].columns
    RP.write_excel(led, tmp_path / "w.xlsx", sample_id="s", tof_flag_mz=350.0)
    assert (tmp_path / "w.xlsx").stat().st_size > 0


def _page(monkeypatch, tmp_path, section, ctx) -> list:
    from matplotlib.backends.backend_pdf import PdfPages
    raw = []
    orig_lines, orig_close = R._text_lines, R._close

    def lines(fig, ls, **kw):
        raw.extend(t for st, t in ls if st != "gap" and isinstance(t, str))
        return orig_lines(fig, ls, **kw)

    def close(pdf, fig):
        from matplotlib.text import Text
        raw.extend(t.get_text() for t in fig.findobj(Text) if t.get_text())
        orig_close(pdf, fig)

    with monkeypatch.context() as mp:
        mp.setattr(R, "_text_lines", lines)
        mp.setattr(R, "_close", close)
        with PdfPages(tmp_path / f"page_{section.__name__}.pdf") as pdf:
            section(ctx, pdf)
    return raw


def _write_run(d, merged, *, summary):
    (d / "per_file").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    merged.assign(n_files=2, ion_score=0.9, formula_agree=True).to_csv(d / "merged_ledger.csv", index=False)
    for s in ("f1", "f2"):
        pf = [dict(peak_id=i + 1, mz=r.mz, role="M0", neutral_formula=r.neutral_formula, adduct=r.adduct,
                   height=1000.0 - 10 * i, tier=r.tier, ppm_error=0.2, parent_peak_id=None)
              for i, r in enumerate(merged.itertuples())]
        pd.DataFrame(pf).to_csv(d / "per_file" / f"{s}_ledger.csv", index=False)
    (d / "batch_summary.json").write_text(json.dumps(summary))


def test_the_pdf_says_the_counts_once_and_marks_the_species_tables(tmp_path, monkeypatch):
    merged = _merged()
    block = _flag(merged)
    d = tmp_path / "run"
    _write_run(d, merged.drop(columns=["evidence_level", "claim"]), summary={"tof_flag": block})
    ctx = R.load_context(str(d), tag="T", label="Br- CIMS")
    assert ctx["mass_only"]["n_flagged"] == 3
    assert ctx["mass_only_neutrals"] == {"C10H16O6", "C20H30O10", "C10H16O7"}
    txt = " ".join(" ".join(_page(monkeypatch, tmp_path, R.findings, ctx)).split())
    assert ("Mass-only readings (TOF): 3 of 7 Assigned readings show no attached isotope line of the neutral's "
            "own elements in any file that assigned them") in txt
    assert "2 of 5 below m/z 350, where the reading rests on mass alone, and 1 of 2 at or above it" in txt
    assert "At or above m/z 350 an unmarked reading is not supported either" in txt
    assert f"C10H16O6{R.MASS_ONLY_MARK}" in txt and f"C10H16O5{R.MASS_ONLY_MARK}" not in txt
    assert f"C20H30O10{R.MASS_ONLY_MARK}" in txt                           # the oligomer line
    app = _page(monkeypatch, tmp_path, R.assignments_table, ctx)
    flagged_rows = [t for t in app if f"Assigned{R.MASS_ONLY_MARK}" in t]
    # (each line is read twice: as handed to the page and as drawn)
    assert len(set(flagged_rows)) == 3 and any("TOF mass-only flag" in t for t in app)
    # the run's own threshold (batch_summary['tof_flag']), not the package default, splits the sentence
    m300 = _merged()
    b300 = _flag(m300, threshold=300.0)
    d3 = tmp_path / "run300"
    _write_run(d3, m300.drop(columns=["evidence_level", "claim"]), summary={"tof_flag": b300})
    ctx3 = R.load_context(str(d3), tag="T", label="Br- CIMS")
    assert ctx3["mass_only"]["threshold_mz"] == 300.0
    txt3 = " ".join(" ".join(_page(monkeypatch, tmp_path, R.findings, ctx3)).split())
    assert "0 of 3 below m/z 300, where the reading rests on mass alone, and 3 of 4 at or above it" in txt3
    assert "At or above m/z 300 an unmarked reading is not supported either" in txt3
    # a neutral is marked only when EVERY Assigned reading of it is flagged
    mix = pd.DataFrame([dict(mz=300.0, neutral_formula="C10H16O5", adduct="[M+Br]-", tier="Assigned", mass_only=True),
                        dict(mz=250.0, neutral_formula="C10H16O5", adduct="[M+NO3]-", tier="Assigned", mass_only=False),
                        dict(mz=260.0, neutral_formula="C9H14O5", adduct="[M+NO3]-", tier="Assigned", mass_only=True)])
    mo = R._mass_only_context({"merged": mix, "batch": {}})
    assert mo["mass_only_neutrals"] == {"C9H14O5"} and ("C10H16O5", "[M+Br]-") in mo["mass_only_pairs"]
    # a run without flag values: no sentence, no marker
    d2 = tmp_path / "run2"
    _write_run(d2, _merged().drop(columns=["evidence_level", "claim"]), summary={})
    ctx2 = R.load_context(str(d2), tag="T", label="Br- CIMS")
    assert "mass_only" not in ctx2
    txt2 = " ".join(_page(monkeypatch, tmp_path, R.findings, ctx2))
    assert "Mass-only readings" not in txt2 and R.MASS_ONLY_MARK not in txt2


def test_the_scorecard_census_counts_the_flag_at_the_runs_threshold():
    merged = _merged()
    block = _flag(merged, threshold=300.0)
    run = SimpleNamespace(ledger=merged, summary={"tof_flag": block})
    mo = SC.census(run)["mass_only"]
    assert mo["threshold_mz"] == 300.0 and mo["n_flagged"] == 3 and mo["at_or_above"] == {"assigned": 4, "flagged": 3}
    line = SC.mass_only_line(mo)
    assert line.startswith("TOF mass-only flag: 3 of 7 Assigned rows") and "below m/z 300" in line
    plain = SimpleNamespace(ledger=_merged(), summary={})
    assert SC.census(plain)["mass_only"] is None and SC.mass_only_line(None) == ""
    assert not math.isnan(mo["threshold_mz"])
