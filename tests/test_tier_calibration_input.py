"""The tier calibration as a file (assignment/calfile.py; `--tier-calibration`).

`peaky calibration pool` pools the per-file ledgers of finished run dirs into a
json (the batch pool's core rules plus inorganic ions with their own isotope
line, never a reagent identity, an alias-held row or a 13C-shadow line); a batch
reads it ONLY for a file that neither its own core nor the batch's pool
calibrates, as 'campaign' (reagent, polarity and scan range match) or 'lent'
(another scan range; another reagent only with --tier-calibration-lent over the
same days), a lent width within 1.5x of the lending runs' per-file spread. This
file pins the pool, the choice, the batch wiring (per-file stats and rows,
batch_summary `levels_calibration`, levels assessed, byte-identical tables where
a file calibrates either way, a spawned worker's keep flag), the manifest record
and the offline re-level.

Synthetic ledgers only. Run: pytest tests/test_tier_calibration_input.py -q
"""

from __future__ import annotations

import filecmp
import json
import os
import pickle

import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import calfile as CF
from peaky.assignment import calpool as CP
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.assignment.fluorine import SERIES_ALIAS_MARK
from peaky.assignment.levels import context as CX
from peaky.assignment.passes import PassConfig
from peaky.chem import chemistry as C
from tests.test_calpool import AD, NEUTRALS, ORBI, _batch_table, _file, _InProcessPool, _ion, _staged_table

DAY = "2025-10-01"


# --------------------------------------------------------------------------- calibration jsons
def _record(*, reagent="Br", polarity="-", scan_range=(100.0, 139.0), mu=0.0, sigma=0.3, spread=(0.25, 0.35),
            days=(DAY, DAY), usable=True, klass="orbitrap"):
    pf = ({"n": 4, "min": spread[0], "median": round(sum(spread) / 2, 4), "max": spread[1]} if spread
          else {"n": 0, "min": None, "median": None, "max": None})
    return {"schema": CF.SCHEMA, "source": "campaign", "reagent": reagent, "polarity": polarity,
            "scan_range": list(scan_range), "instrument_class": klass, "days": list(days) if days else None,
            "runs": [{"dir": "r1", "commit": "abc", "n_files": 4}],
            "core": {"n_ions": 20, "n_files": 4, "n_files_read": 4, "ions": []},
            "width": ({"mu": mu, "sigma": sigma, "basis": "run", "model": "constant",
                       "pooled": {"mu": mu, "sigma": sigma}, "run": None, "per_file_sigma": pf}
                      if usable else None),
            "usable": usable, "reason": None if usable else "too few ions"}


def _json(tmp_path, name="cal.json", **kw):
    p = str(tmp_path / name)
    CF.write(_record(**kw), p)
    return CF.load(p)


# --------------------------------------------------------------------------- the batch
def _run(tmp_path, monkeypatch, *, record=None, lent=False, iso_rows=3, calibrated=False, n_jobs=1,
         staged=False, lines=None, seen_cfg=None, keeps=None):
    """A batch of synthetic files (assign.run stubbed as in test_calpool): each commits 16 CHO [M-H]-
    readings plus one 2.5 ppm off, `iso_rows` of them with their 13C line in every file (3: the batch's
    pool falls short; 16: it calibrates), `calibrated` True: every file calibrates alone. `record`: the
    calibration json on the cfg (`lent`: tier_calibration_lent). `seen_cfg` collects the cfg each run
    was handed (the worker path included); `keeps` its keep_pool_inputs."""
    import concurrent.futures as cf

    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO
    if staged:
        pk, k_min, k_max = _staged_table(), 2, 2
    else:
        pk, k_min, k_max = _batch_table([f"f{i}" for i in range(4)], list(range(100, 140))), 3, 4

    def fake_assign(sid, context="ambient-air", **kw):
        keep = bool(kw.get("keep_pool_inputs", False))
        if seen_cfg is not None:
            seen_cfg.append(kw.get("cfg"))
        if keeps is not None:
            keeps[sid] = kw.get("keep_pool_inputs")
        rows = [(nf, AD, 0.05 * ((j % 5) - 2), 1e5 * (1 + j), j < iso_rows, "Good")
                for j, nf in enumerate(NEUTRALS[:16])]
        rows.append(("C17H28O6", AD, 2.5, 1e5, False, "Good"))     # off-calibration, uncorroborated
        at = _file(sid, rows, tier=False)
        led = at.copy(deep=True)
        T.apply_tiers(led)
        ri = EV.file_run_inputs(sample_id=sid, reagent="Br", context="ambient-air", resolution=ORBI,
                                height_gate_cps=10.0, degeneracy_cal=None)
        EV.apply_levels(led, run_inputs=ri)
        if calibrated:
            return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                    "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                              "degeneracy_cal": {"mu": 0.0, "sigma": 0.3},
                              "tier_calibration": T.CAL_SOURCE_PER_FILE},
                    **({"pool_inputs": None} if keep else {})}
        inputs = CP.PoolInputs(at_tiers=at, adducts=[AD], level_inputs=ri) if keep else None
        return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0, "degeneracy_cal": None,
                          "tier_calibration": T.CAL_SOURCE_NONE},
                **({"pool_inputs": inputs} if keep else {})}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    if n_jobs > 1:
        monkeypatch.setattr(cf, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(AB, "_W", {})
        monkeypatch.setenv("PEAKY_MATCH_WORKERS", "2")
    cfg = PassConfig(tier_calibration=record, tier_calibration_lent=lent)
    AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path), cfg=cfg,
           k_min=k_min, k_max=k_max, min_gain=0.0, n_jobs=n_jobs, resolving_power=ORBI,
           log=(lines.append if lines is not None else (lambda *a: None)))
    return json.load(open(tmp_path / "batch_summary.json"))


def _m0(tmp_path, sid):
    led = pd.read_csv(tmp_path / "per_file" / f"{sid}_ledger.csv")
    return led[led["role"] == "M0"]


def test_without_a_file_a_batch_whose_pool_falls_short_levels_nothing(tmp_path, monkeypatch):
    """The control of every test below: 3 isotope-backed rows per file, no json -> 'none', no level."""
    summ = _run(tmp_path, monkeypatch)
    n = len(summ["per_file"])
    assert n >= 2 and summ["levels_calibration"] == {
        "source": "none", "files": {"per_file": 0, "pooled": 0, "campaign": 0, "lent": 0, "none": n}, "file": None}
    assert summ["levels_not_assessed_reason"].startswith("Evidence levels were not assessed")
    assert "campaign" not in summ["tier_calibration"]["files"]


def test_a_matching_json_is_lent_as_campaign_and_the_files_get_levels(tmp_path, monkeypatch):
    rec = _json(tmp_path / "..", "c1.json")
    lines = []
    summ = _run(tmp_path, monkeypatch, record=rec, lines=lines)
    lc = summ["levels_calibration"]
    n = len(summ["per_file"])
    assert lc["source"] == "campaign" and lc["files"]["campaign"] == n and lc["files"]["none"] == 0
    f = lc["file"]
    assert f["sha256"] == rec["_sha256"] and f["read_as"] == "campaign" and f["used"] is True and f["reason"] is None
    assert f["runs"] == [{"dir": "r1", "commit": "abc", "n_files": 4}]
    assert set(f["lent_files"]) == {p["sample_id"] for p in summ["per_file"]}
    assert f["batch_facts"]["scan_range"] == [100.0, 139.0] and f["batch_facts"]["days"] == [DAY, DAY]
    assert summ["tier_calibration"]["source"] == "campaign"
    assert summ["tier_calibration"]["files"] == {"per_file": 0, "pooled": 0, "none": 0, "campaign": n, "lent": 0}
    assert summ["tier_calibration"]["pool"]["used"] is False      # the batch's own pool fell short
    for p in summ["per_file"]:
        assert p["tier_calibration"] == "campaign" and p["degeneracy_cal"]["sigma"] == pytest.approx(0.3)
        assert p["tier_pool"]["width"] == "file" and p["tier_pool"]["centre"] == "file"
    assert summ["levels_not_assessed_reason"] is None
    assert sum(summ["evidence_levels"]["pooled"].values()) > 0
    m0 = _m0(tmp_path, summ["per_file"][0]["sample_id"])
    assert (m0[T.TIER_CALIBRATION] == "campaign").all()
    off = m0[m0["neutral_formula"] == "C17H28O6"]
    assert (off["tier"] == "Candidate").all() and off["tier_reason"].str.contains("campaign-pooled").all()
    assert any("is read as campaign" in ln for ln in lines)


def test_a_json_of_another_scan_range_is_lent(tmp_path, monkeypatch):
    rec = _json(tmp_path / "..", "c2.json", scan_range=(40.0, 600.0))
    summ = _run(tmp_path, monkeypatch, record=rec)
    lc = summ["levels_calibration"]
    assert lc["source"] == "lent" and lc["files"]["lent"] == len(summ["per_file"]) and lc["file"]["read_as"] == "lent"
    assert summ["levels_not_assessed_reason"] is None
    off = _m0(tmp_path, summ["per_file"][-1]["sample_id"]).query("neutral_formula == 'C17H28O6'")
    assert off["tier_reason"].str.contains("off the lent instrument accuracy").all()


def test_another_reagent_is_lent_only_with_the_flag_and_only_over_the_same_days(tmp_path, monkeypatch):
    rec = _json(tmp_path / "..", "c3.json", reagent="NO3")
    summ = _run(tmp_path / "a", monkeypatch, record=rec)
    lc = summ["levels_calibration"]
    assert lc["source"] == "none" and lc["files"]["none"] == len(summ["per_file"]) and lc["file"]["read_as"] is None
    assert "lending across reagents needs --tier-calibration-lent" in lc["file"]["reason"]
    assert "A calibration file was given and not used" in summ["levels_not_assessed_reason"]
    assert all(p["degeneracy_cal"] is None for p in summ["per_file"])
    summ = _run(tmp_path / "b", monkeypatch, record=rec, lent=True)
    assert summ["levels_calibration"]["source"] == "lent" and summ["levels_calibration"]["file"]["lent_flag"] is True
    far = _json(tmp_path / "..", "c4.json", reagent="NO3", days=("2024-01-01", "2024-01-05"))
    summ = _run(tmp_path / "c", monkeypatch, record=far, lent=True)
    assert summ["levels_calibration"]["source"] == "none"
    assert "needs the same days" in summ["levels_calibration"]["file"]["reason"]


def test_a_lent_width_outside_the_lending_spread_is_refused(tmp_path, monkeypatch):
    """Width 3x the spread's top: a lent file stays 'none' and says why; the same json at the batch's own
    scan range (campaign) is not gated."""
    rec = _json(tmp_path / "..", "c5.json", scan_range=(40.0, 600.0), sigma=1.05, spread=(0.25, 0.35))
    summ = _run(tmp_path / "a", monkeypatch, record=rec)
    lc = summ["levels_calibration"]
    assert lc["source"] == "none" and "outside 1.5x the lending runs' per-file sigma spread" in lc["file"]["reason"]
    assert "A calibration file was given and not used" in summ["levels_not_assessed_reason"]
    ok = _json(tmp_path / "..", "c6.json", scan_range=(40.0, 600.0), sigma=0.5, spread=(0.25, 0.35))
    assert _run(tmp_path / "b", monkeypatch, record=ok)["levels_calibration"]["source"] == "lent"


def test_a_polarity_or_an_unusable_json_lends_nothing():
    rec = dict(_record(), _sha256="x" * 64)
    kw = dict(reagent="Br", polarity="-", instrument_class="orbitrap", scan_range=[100.0, 139.0], days=[DAY, DAY])
    assert CF.choose(rec, **kw) == ("campaign", None)
    assert CF.choose(rec, **dict(kw, polarity="+"))[0] is None
    assert CF.choose(rec, **dict(kw, instrument_class="tof"))[0] is None
    assert CF.choose(dict(_record(usable=False), _sha256="y"), **kw)[1].endswith("not usable: too few ions")


def test_the_scan_range_match_reads_the_low_edge_and_the_high_edge_band():
    assert CF.scan_match([45.0, 383.0], [45.0, 540.0])          # one 40-600 method, sparse tops
    assert not CF.scan_match([128.0, 599.1], [45.0, 540.0])     # 128-600 vs 40-600
    assert not CF.scan_match([65.0, 118.0], [45.0, 383.0])      # 65-120 vs 40-600
    assert CF.scan_match([122.1, 579.2], [124.0, 579.2])        # low edge within 3 Da
    assert not CF.scan_match([45.0, 159.0], [45.0, 540.0])      # the top 3.4x apart


@pytest.mark.parametrize("calibrated", [True, False])
def test_a_file_that_calibrates_either_way_is_byte_identical_with_the_json(tmp_path, monkeypatch, calibrated):
    """Every file calibrated alone (per_file) or the batch's pool calibrates (16 isotope-backed rows:
    pooled): with a json the per-file ledgers, the merged ledger and every table are byte for byte the
    run's without it; only batch_summary gains the json's record."""
    rec = _json(tmp_path, "c7.json", sigma=0.9, spread=(0.8, 1.0))
    a = _run(tmp_path / "a", monkeypatch, calibrated=calibrated, iso_rows=16)
    b = _run(tmp_path / "b", monkeypatch, calibrated=calibrated, iso_rows=16, record=rec)
    want = "per_file" if calibrated else "pooled"
    assert a["levels_calibration"]["source"] == b["levels_calibration"]["source"] == want
    assert b["levels_calibration"]["file"]["used"] is False and b["levels_calibration"]["file"]["decided"] is False
    files = [os.path.relpath(os.path.join(d, f), tmp_path / "a") for d, _s, fs in os.walk(tmp_path / "a") for f in fs
             if f != "batch_summary.json"]
    assert any(f.startswith("per_file") for f in files) and "merged_ledger.csv" in files
    assert any(f.endswith("evidence_levels.csv") for f in files)
    for f in files:
        assert filecmp.cmp(tmp_path / "a" / f, tmp_path / "b" / f, shallow=False), f


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_workers_carry_the_file_and_keep_their_copies_while_it_can_lend(tmp_path, monkeypatch, n_jobs):
    """The json rides the cfg into every run (through the worker pool too); once the batch's own pool
    falls short on the cover files, the residual stage's runs still keep their tier-time copies because
    the file can lend, and every file is lent it."""
    rec = _json(tmp_path / "..", f"c8_{n_jobs}.json", scan_range=(100.0, 502.0))
    seen, keeps = [], {}
    summ = _run(tmp_path, monkeypatch, record=rec, staged=True, n_jobs=n_jobs, seen_cfg=seen, keeps=keeps)
    assert len(seen) == 4 and all(c.tier_calibration["_sha256"] == rec["_sha256"] for c in seen)
    assert keeps == {"a0": True, "a1": True, "z1": True, "z2": True}
    assert summ["levels_calibration"]["files"]["campaign"] == 4
    assert pickle.loads(pickle.dumps(PassConfig(tier_calibration=rec))).tier_calibration == rec


def test_the_manifest_records_the_json_by_its_sha256(tmp_path):
    from peaky import pipeline as PL
    from peaky.reporting import provenance as PV
    p = str(tmp_path / "c9.json")
    rec = _record()
    rec["core"]["ions"] = [{"ion": "C4H5O4-", "neutral_formula": "C4H6O4", "adduct": AD, "mz": 117.0, "ppm": 0.1,
                            "sid": "s"}]
    sha = CF.write(rec, p)
    cfg = PL.tier_calibration_config(PassConfig(), p, True, log=lambda *a: None)
    assert cfg.tier_calibration["_sha256"] == sha == CF.sha256_file(p) and cfg.tier_calibration_lent is True
    man = PV.build_manifest(run_dir=str(tmp_path), batch_name="b", dataset=None, sample_ids=["s"], reagent="Br",
                            cfg=cfg)
    assert man["config"]["tier_calibration"]["sha256"] == sha and "path" not in man["config"]["tier_calibration"]
    assert "core" not in man["config"]["tier_calibration"] and man["config"]["tier_calibration_lent"] is True
    assert man["input"]["tier_calibration"] == {"path": os.path.abspath(p), "sha256": sha, "lent_flag": True}
    plain = PV.build_manifest(run_dir=str(tmp_path), batch_name="b", dataset=None, sample_ids=["s"], reagent="Br",
                              cfg=PassConfig())
    assert plain["config"]["tier_calibration"] is None and "tier_calibration" not in plain["input"]
    with pytest.raises(ValueError, match="needs --tier-calibration"):
        PL.tier_calibration_config(PassConfig(), None, True)


def test_a_json_of_another_schema_or_without_a_width_is_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(dict(_record(), schema=99)))
    with pytest.raises(CF.CalibrationError, match="schema"):
        CF.load(str(p))
    p.write_text(json.dumps(dict(_record(), width=None)))
    with pytest.raises(CF.CalibrationError, match="no width"):
        CF.load(str(p))


def test_run_windows_marks_a_campaign_or_lent_file_as_borrowed(monkeypatch):
    monkeypatch.setattr(CX, "fit_windows", lambda *a, **k: pytest.fail("the fallback ran"))
    led = _file("w", [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:3]])
    summary = {"reagent": "NO3", "context": "ambient-air", "reflists_active": [],
               "per_file": [{"sample_id": "a", "degeneracy_cal": {"mu": 0.1, "sigma": 0.2},
                             "tier_calibration": T.CAL_SOURCE_PER_FILE},
                            {"sample_id": "c", "degeneracy_cal": {"mu": 0.0, "sigma": 0.9},
                             "tier_calibration": T.CAL_SOURCE_CAMPAIGN},
                            {"sample_id": "l", "degeneracy_cal": {"mu": 0.0, "sigma": 0.7},
                             "tier_calibration": T.CAL_SOURCE_LENT}]}
    win = CX.run_windows(summary, {"a": led, "c": led, "l": led})
    assert win["c"].get("pooled") is True and win["l"].get("pooled") is True and "pooled" not in win["a"]
    windows, run = CX.window_table(win)
    assert run == (0.1, 0.2) and windows["c"] == (0.0, 0.9) and windows["l"] == (0.0, 0.7)


# --------------------------------------------------------------------------- calibration pool
def _rows(n_cho=16, *, ppm=0.05):
    rows = [(nf, AD, ppm * ((k % 5) - 2), 1e5 * (1 + k), True, "Good") for k, nf in enumerate(NEUTRALS[:n_cho])]
    return rows


def _run_dir(path, sids, *, reagent="Br", own=None, extra=None, alias_ion=None, shadow_mz=None):
    """A finished run dir: per-file ledgers of 16 CHO [M-H]- readings with their 13C lines, `extra`
    rows [(neutral, adduct, iso label | None)] each with that isotope line, batch_summary, manifest, a
    series parquet, a label-twin table with `shadow_mz` as a shadow line. `own`: (mu, sigma) the files
    calibrated alone with, else uncalibrated. `alias_ion`: a neutral whose rows carry the alias-series
    hold's mark."""
    os.makedirs(os.path.join(path, "per_file"), exist_ok=True)
    os.makedirs(os.path.join(path, "tables"), exist_ok=True)
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    ts = []
    for s in sids:
        led = _file(s, _rows())
        for k, (nf, ad, lab) in enumerate(extra or []):
            mz = C.ion_mz(nf, ad)
            pid = f"{s}-x{k}"
            add = L.new_ledger(pd.DataFrame([(pid, mz, 2e5)] + ([(pid + "i", mz + 1.99, 1e4)] if lab else []),
                                            columns=["peak_id", "mz", "height"]))
            led = pd.concat([led, add], ignore_index=True)
            L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_formula=_ion(nf, ad), ion_score=0.95,
                                compound_score=0.95, ppm_error=0.02, pass_no=1, method="cheminfo+grid",
                                confidence="Good", commentary=f"Pass 1: {nf} {ad}")
            if lab:
                L.attach_isotopologue(led, pid + "i", pid, iso_label=lab, iso_match_score=0.9)
        if alias_ion:
            i = led.index[(led["role"] == "M0") & (led["neutral_formula"] == alias_ion)]
            led.loc[i, "tier_reason"] = led.loc[i, "tier_reason"].astype(str) + f"; {SERIES_ALIAS_MARK} (alias)"
        led.to_csv(os.path.join(path, "per_file", f"{s}_ledger.csv"), index=False)
        ts += [{"sample_item_id": s, "mz": m, "datetime_utc": pd.Timestamp(f"{DAY} 12:00", tz="UTC")}
               for m in pd.to_numeric(led["mz"]).tolist()]
    pd.DataFrame(ts).to_parquet(os.path.join(path, "data", "X_ts.parquet"))
    if shadow_mz is not None:
        pd.DataFrame([{"mz": shadow_mz, "adduct": "[M+NO3]-", "line_verdict": "shadow"}]).to_csv(
            os.path.join(path, "tables", "label_twins.csv"), index=False)
    from peaky.chem.resolution import Resolution
    per_file = [{"sample_id": s, "degeneracy_cal": ({"mu": own[0], "sigma": own[1]} if own else None),
                 "tier_calibration": "per_file" if own else "none"} for s in sids]
    json.dump({"reagent": reagent, "per_file": per_file, "resolution": Resolution.coerce(ORBI).as_dict(),
               "mass_axis": {}}, open(os.path.join(path, "batch_summary.json"), "w"))
    json.dump({"code": {"git": {"commit": "c0ffee"}}, "input": {}}, open(os.path.join(path, "run_manifest.json"), "w"))
    return str(path)


def test_the_pool_takes_distinct_ions_plus_own_line_inorganics_and_never_the_reagent(tmp_path):
    """Two runs, two files each: 16 CHO ions (13C), H2SO4 [M-H]- on its 34S line (in), HBr [M+Br]- on
    81Br (the reagent's identity: out), H2O [M+Br]- on 81Br (the adduct's own line: out), one CHO ion
    held by the alias-series mark (out), one on a 13C-shadow line of the label-twin table (out)."""
    extra = [("H2SO4", "[M-H]-", "34S"), ("HBr", "[M+Br]-", "81Br"), ("H2O", "[M+Br]-", "81Br")]
    shadow = C.ion_mz(NEUTRALS[3], AD)
    r1 = _run_dir(tmp_path / "r1", ["a0", "a1"], extra=extra, alias_ion=NEUTRALS[2], shadow_mz=shadow,
                  own=(0.02, 0.21))
    r2 = _run_dir(tmp_path / "r2", ["b0", "b1"], extra=extra, alias_ion=NEUTRALS[2], shadow_mz=shadow)
    rec = CF.pool_runs([r1, r2], log=lambda *a: None)
    ions = {i["neutral_formula"]: i for i in rec["core"]["ions"]}
    assert "H2SO4" in ions and "HBr" not in ions and "H2O" not in ions
    assert NEUTRALS[2] not in ions and NEUTRALS[3] not in ions
    assert rec["core"]["n_ions"] == 15 and rec["core"]["n_inorganic_ions"] == 1 and rec["usable"] is True
    assert rec["core"]["n_files"] == 4 and rec["core"]["n_files_read"] == 4
    assert rec["reagent"] == "Br" and rec["polarity"] == "-" and rec["instrument_class"] == "orbitrap"
    assert rec["days"] == [DAY, DAY] and rec["runs"][0]["commit"] == "c0ffee"
    w = rec["width"]
    assert w["basis"] == "run" and w["sigma"] == pytest.approx(0.21) and w["mu"] == pytest.approx(0.02)
    assert w["per_file_sigma"]["n"] == 2 and w["per_file_sigma"]["max"] == pytest.approx(0.21)
    # without the extension the inorganic row is not a core row
    led = pd.read_csv(os.path.join(r1, "per_file", "a0_ledger.csv"))
    assert "H2SO4" not in set(CP.core_rows(led)["neutral_formula"])
    assert "H2SO4" in set(CP.core_rows(led, inorganic=True)["neutral_formula"])


def test_the_pool_refuses_mixed_reagents_unless_asked(tmp_path):
    r1 = _run_dir(tmp_path / "r1", ["a0", "a1"])
    r2 = _run_dir(tmp_path / "r2", ["b0", "b1"], reagent="NO3")
    with pytest.raises(CF.CalibrationError, match="mix reagents"):
        CF.pool_runs([r1, r2], log=lambda *a: None)
    rec = CF.pool_runs([r1, r2], allow_mixed=True, log=lambda *a: None)
    assert rec["reagent"] == ["Br", "NO3"] and rec["width"]["basis"] == "pool"


def test_a_short_core_writes_an_unusable_json_that_says_why(tmp_path):
    r1 = _run_dir(tmp_path / "r1", ["a0"])
    rec = CF.pool_runs([r1], log=lambda *a: None)
    assert rec["usable"] is False and rec["width"] is None and "needs 15" in rec["reason"]


def test_the_cli_pools_shows_and_compares(tmp_path, capsys):
    from peaky import cli
    r1 = _run_dir(tmp_path / "r1", ["a0", "a1"], own=(0.0, 0.2))
    out = str(tmp_path / "c.json")
    assert cli.main(["calibration", "pool", r1, "--out", out]) == 0
    rec = CF.load(out)
    assert rec["usable"] and rec["core"]["n_ions"] == 16 and rec["_sha256"] == CF.sha256_file(out)
    assert cli.main(["calibration", "show", out]) == 0
    assert "16 distinct ions" in capsys.readouterr().out
    other = str(tmp_path / "d.json")
    CF.write(dict(_record(sigma=0.22)), other)
    assert cli.main(["calibration", "compare", out, other, "--json"]) == 0
    cmp = json.loads(capsys.readouterr().out)
    assert cmp["width_ratio"] == pytest.approx(1.1) and cmp["n_ions"] == [16, 0]


# --------------------------------------------------------------------------- the offline re-level
def test_relevel_lends_the_json_offline_and_changes_nothing_without_one(tmp_path, monkeypatch):
    """A finished run whose files no calibration reached: re-levelled with a matching json its pairs get
    levels ('campaign', the per-file windows centred on the files); with a json it may not read the
    level table is the one a re-level without a json writes. The run dir is never written."""
    run = tmp_path / "run"
    _run(run, monkeypatch)
    before = {f: os.path.getmtime(run / f) for f in os.listdir(run) if os.path.isfile(run / f)}
    rec = _json(tmp_path, "r.json")
    out = CF.relevel_run(str(run), rec, out_dir=str(tmp_path / "rl"), log=lambda *a: None)
    assert out["read_as"] == "campaign" and set(out["centres"].values()) == {"file"}
    s = json.load(open(tmp_path / "rl" / "batch_summary.json"))
    assert s["levels_calibration"]["source"] == "campaign" and s["levels_not_assessed_reason"] is None
    assert all(p["tier_calibration"] == "campaign" for p in s["per_file"])
    lv = pd.read_csv(tmp_path / "rl" / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert lv["evidence_level"].isin(["3c", "4a", "4b", "5a", "5b"]).any()
    m = pd.read_csv(tmp_path / "rl" / "merged_ledger.csv", keep_default_na=False)
    assert m["evidence_level"].isin(["3c", "4a", "4b", "5a", "5b"]).any()
    CF.relevel_run(str(run), None, out_dir=str(tmp_path / "none"), log=lambda *a: None)
    CF.relevel_run(str(run), _json(tmp_path, "x.json", reagent="NO3"), out_dir=str(tmp_path / "x"),
                   log=lambda *a: None)
    assert filecmp.cmp(tmp_path / "none" / "tables" / "evidence_levels.csv",
                       tmp_path / "x" / "tables" / "evidence_levels.csv", shallow=False)
    sx = json.load(open(tmp_path / "x" / "batch_summary.json"))
    assert "lending across reagents" in sx["levels_not_assessed_reason"]
    assert before == {f: os.path.getmtime(run / f) for f in before}
    with pytest.raises(CF.CalibrationError, match="never the run dir"):
        CF.relevel_run(str(run), rec, out_dir=str(run))
