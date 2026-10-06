"""Declared side channels: the adducts a reagent profile declares beside its
analyte channels, scored when the server resolves their mechanism.

Nothing opens by polarity. The uronium profile declares [M+NH4]+ (the batch's
amine gate then keeps or re-reads each reading); every other built-in profile
declares none; `--side-channels` (PassConfig.side_channels) opens exactly the
channels asked for, `none` closes them all.

Pins: the per-profile defaults and the config field; the resolution order
(explicit > the cfg's own tuple > the profile) and the compose union; the cfg
field survives the pickling that carries it into a spawned worker; an offline
run registers and opens its side channels and records them per file; the
composite de-blend reads the DECLARED channels (a nitrate run with an opted-in
[M+Br2]- skips it); a channel of the other polarity, without a server
mechanism, or on a labelled-ammonium run is skipped and said so; the batch
summary records what was asked for and what the files opened; the amine gate
acts on a side-channel [M+NH4]+ reading at the merge and on the evidence
scale; the CLI flag and the pipeline thread the choice; the scorecard's arms
open what the run recorded.

Offline: no server, no network. Run: pytest tests/test_side_channels.py -q
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky import pipeline as PL
from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment.levels import split as SPL
from peaky.assignment.passes import config as PCfg
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from peaky.io import io_mascope as IO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scorecard as SC  # noqa: E402

NH4 = "[M+NH4]+"


@pytest.fixture(autouse=True)
def _forget_offline_samples():
    yield
    for sid in list(IO._OFFLINE):
        if sid.startswith("side-"):
            IO.unregister_offline_sample(sid)


# --------------------------------------------------------------------------- the profiles
def test_only_the_uronium_profile_declares_a_side_channel():
    assert P.UR.side_channels == (NH4,)
    for prof in (P.BR, P.NO3, P.NO3_15N, P.IODIDE, P.EASYIC, P.NH4_15N):
        assert prof.side_channels == (), prof.name
    # every declared channel is one the server can score
    assert all(a in IO.ADDUCT_TO_MECH for p in P.PROFILES.values() for a in p.side_channels)


def test_a_config_profile_declares_its_own_side_channels():
    assert "side_channels" in P._CONFIG_FIELDS
    base = {"name": "X-side", "label": "X", "polarity": "-", "adducts": ["[M-H]-"], "normaliser": "tic",
            "reagent_ion_re": None, "ranges": "C0-10 H0-20 O0-5", "detect_adduct": None}
    assert P.from_dict(dict(base, side_channels=["[M+CO3]-", "[M+Br2]-", "[M+CO3]-"])).side_channels == \
        ("[M+CO3]-", "[M+Br2]-")                                   # a list, once each, in order
    assert P.from_dict(dict(base, side_channels="[M+CO3]-")).side_channels == ("[M+CO3]-",)   # one string = one
    assert P.from_dict(dict(base, side_channels=["none"])).side_channels == ()
    assert P.from_dict(base).side_channels == ()


def test_the_resolution_order_explicit_then_the_cfg_then_the_profile():
    cfg = PCfg.PassConfig()
    assert cfg.side_channels is None                                # unset, not ()
    lines = []
    assert P.apply_side_channels(cfg, P.UR, log=lines.append) == (NH4,) and cfg.side_channels == (NH4,)
    assert lines == ["[gate] side channels ['[M+NH4]+'] (from profile Ur)"]
    # an earlier call on the way down (pipeline -> assign_batch) keeps the profile's credit
    lines.clear()
    P.apply_side_channels(cfg, P.UR, log=lines.append)
    assert "from profile Ur" in lines[0]
    # an explicit () closes the uronium channel; a cfg that already carries () outranks the profile too
    off = PCfg.PassConfig()
    assert P.apply_side_channels(off, P.UR, explicit=()) == () and off.side_channels == ()
    assert P.apply_side_channels(PCfg.PassConfig(side_channels=()), P.UR) == ()
    assert P.apply_side_channels(PCfg.PassConfig(), P.UR, explicit="none") == ()
    # an explicit tuple opens a channel the profile does not declare
    lines.clear()
    no3 = PCfg.PassConfig()
    assert P.apply_side_channels(no3, P.NO3, explicit=["[M+CO3]-"], log=lines.append) == ("[M+CO3]-",)
    assert lines == ["[gate] side channels ['[M+CO3]-'] (from explicit config)"]
    # ... and the explicit value outranks a tuple the cfg carried
    assert P.apply_side_channels(PCfg.PassConfig(side_channels=(NH4,)), P.UR, explicit=()) == ()
    # no profile (a forced adduct list, a context-only entry point): closed
    lines.clear()
    assert P.apply_side_channels(PCfg.PassConfig(), None, log=lines.append) == ()
    assert lines == ["[gate] side channels closed (from no profile)"]
    assert P.apply_side_channels(PCfg.PassConfig(), P.NO3) == ()
    # a knob, fingerprinted with the configuration, not a runtime field
    assert "side_channels" not in PCfg.PassConfig.RUNTIME_FIELDS
    assert dataclasses.asdict(PCfg.PassConfig(side_channels=(NH4,)))["side_channels"] == (NH4,)


def test_compose_unions_the_declared_side_channels():
    assert P.compose([P.UR, P.EASYIC]).side_channels == (NH4,)
    assert P.compose([P.EASYIC, P.UR]).side_channels == (NH4,)
    assert P.compose([P.NO3, P.BR]).side_channels == ()
    base = {"label": "X", "polarity": "-", "adducts": ["[M-H]-"], "normaliser": "tic",
            "reagent_ion_re": None, "ranges": "C0-10 H0-20 O0-5", "detect_adduct": None}
    a = P.from_dict(dict(base, name="A-side", side_channels=["[M+CO3]-"]))
    b = P.from_dict(dict(base, name="B-side", side_channels=["[M+Br2]-", "[M+CO3]-"]))
    assert P.compose([a, b]).side_channels == ("[M+CO3]-", "[M+Br2]-")


# --------------------------------------------------------------------------- the spawn pool
def test_the_cfg_carries_the_side_channels_into_a_spawned_worker(monkeypatch):
    """The pool's initargs are pickled into each spawned worker; the PassConfig
    field is how the choice gets there (a patch of the parent never does)."""
    cfg = PCfg.PassConfig(side_channels=("[M+CO3]-",))
    assert pickle.loads(pickle.dumps(cfg)).side_channels == ("[M+CO3]-",)
    initargs = pickle.loads(pickle.dumps(("ambient-air", None, {"cfg": cfg, "adducts": ["[M+NO3]-"]}, None)))
    seen = []

    def fake_run(sid, context="ambient-air", **kw):
        seen.append(kw["cfg"])
        return {"ledger": pd.DataFrame(), "stats": {"side_channels": list(kw["cfg"].side_channels)}}

    monkeypatch.setattr(A, "run", fake_run)
    AB._worker_init(*initargs)
    out = AB._assign_one("s1")
    assert seen[0].side_channels == ("[M+CO3]-",) and seen[0] is not initargs[2]["cfg"]   # a per-sample copy
    assert out["stats"]["side_channels"] == ["[M+CO3]-"]


# --------------------------------------------------------------------------- assign.run offline
def _offline(sid, adducts, context, mzs, cfg):
    tbl = pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(mzs))], "mz": mzs,
                        "height": [5e4, 2e4, 1e4, 300.0, 200.0][:len(mzs)]})
    lines: list = []
    out = A.run(sid, context, cfg=cfg, peaks=tbl, adducts=list(adducts), use_cache=False,
                log=lines.append, resolving_power=None)
    return out, lines


UR_MZ = [C.ion_mz("C6H10O", "[M+H]+"), C.ion_mz("C6H10O", NH4), C.ion_mz("C7H12O2", "[M+(CH4N2O)H]+"),
         120.5, 150.3]
NO3_MZ = [C.ion_mz("C5H8O4", "[M+NO3]-"), C.ion_mz("C5H8O4", "[M-H]-"), 140.2]


def test_an_offline_run_registers_and_opens_the_declared_side_channel_and_records_it():
    cfg = PCfg.PassConfig()
    P.apply_side_channels(cfg, P.UR)
    out, lines = _offline("side-ur", P.UR.adducts, "uronium", UR_MZ, cfg)
    assert out["stats"]["side_channels"] == [NH4]
    assert IO.ADDUCT_TO_MECH[NH4] in IO._OFFLINE["side-ur"][1]      # the offline sample registered it
    assert "[run] side channels: opened ['[M+NH4]+']" in lines
    assert any("adducts=['[M+H]+', '[M+(CH4N2O)H]+', '[M+NH4]+']" in ln for ln in lines)
    # closed: nothing registered beyond the analyte channels, nothing opened, no side-channel line
    out, lines = _offline("side-ur-closed", P.UR.adducts, "uronium", UR_MZ, PCfg.PassConfig(side_channels=()))
    assert out["stats"]["side_channels"] == []
    assert IO.ADDUCT_TO_MECH[NH4] not in IO._OFFLINE["side-ur-closed"][1]
    assert not any("side channels" in ln for ln in lines)
    # an unset cfg (a library caller that never applied a profile) opens nothing either: no polarity default
    out, _ = _offline("side-ur-unset", P.UR.adducts, "uronium", UR_MZ, PCfg.PassConfig())
    assert out["stats"]["side_channels"] == []


def test_an_opted_in_channel_opens_and_one_of_the_other_polarity_or_without_a_mechanism_is_skipped():
    cfg = PCfg.PassConfig(side_channels=("[M+Br2]-", "[M+Na]+", "[M+K]+"))
    out, lines = _offline("side-no3", P.NO3.adducts, "ambient-air", NO3_MZ, cfg)
    assert out["stats"]["side_channels"] == ["[M+Br2]-"]
    assert IO.ADDUCT_TO_MECH["[M+Br2]-"] in IO._OFFLINE["side-no3"][1]
    assert IO.ADDUCT_TO_MECH["[M+Na]+"] not in IO._OFFLINE["side-no3"][1]
    side = [ln for ln in lines if ln.startswith("[run] side channels")]
    assert side == ["[run] side channels: opened ['[M+Br2]-']; [M+Na]+ skipped (not a negative channel); "
                    "[M+K]+ skipped (no server mechanism)"]
    assert out["stats"]["reagent_halogen"] is None          # declared, not opened: still no bromide reagent


def test_the_composite_de_blend_reads_the_declared_channels_not_an_opted_in_halogen_side_channel():
    """The even-shift composite test is the halide reagents'; an [M+Br2]- side
    channel opened on a nitrate run must not switch it on (a bromide run's
    declared channels already carry the halogen)."""
    out, _ = _offline("side-no3-br2", P.NO3.adducts, "ambient-air", NO3_MZ,
                      PCfg.PassConfig(side_channels=("[M+Br2]-",)))
    assert out["stats"]["side_channels"] == ["[M+Br2]-"]
    assert out["summaries"]["composite"] == {"flagged": 0, "skipped": "no halogen adduct"}
    br, _ = _offline("side-br", P.BR.adducts, "ambient-air",
                     [C.ion_mz("C5H8O4", "[M+Br]-"), C.ion_mz("C5H8O4", "[M-H]-"), 140.2], PCfg.PassConfig())
    assert br["summaries"]["composite"].get("skipped") is None          # a bromide run keeps the test


def test_a_labelled_ammonium_run_keeps_the_ammonium_and_sodium_channels_closed_even_when_asked():
    cfg = PCfg.PassConfig(side_channels=(NH4, "[M+Na]+"))
    mzs = [C.ion_mz("C6H10O", "[M+^NH4]+"), C.ion_mz("C6H10O", "[M+H]+"), 150.3]
    out, lines = _offline("side-nh4", P.NH4_15N.adducts, "ammonium-15n", mzs, cfg)
    assert out["stats"]["side_channels"] == []
    assert "[run] side channels: opened none; [M+NH4]+ skipped (labelled-ammonium run); " \
           "[M+Na]+ skipped (labelled-ammonium run)" in lines


# --------------------------------------------------------------------------- the batch: summary + amine gate
REAL, REJ = "C7H12O4", "C10H16O2"        # an ammonium adduct that tracks its parent / one that does not
_N = 24
_K = np.arange(_N)
_SHP = 10.0 ** (0.5 + 2.5 * _K / (_N - 1))          # a shaped trace (cv ~ 1.4)
_IONS = [(REAL, "[M+H]+", _SHP * 1e3), (REAL, "[M+(CH4N2O)H]+", _SHP * 2e3), (REAL, NH4, _SHP * 2e2),
         (REJ, "[M+H]+", _SHP * 1e3), (REJ, NH4, _SHP[::-1] * 3e2)]


def _ts():
    t0 = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")
    rows = []
    for k in range(_N):
        sid = f"u{k:02d}"
        t = t0 + pd.Timedelta(hours=2 * k)
        for n, a, tr in _IONS:
            rows.append(dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                             mz=float(C.ion_mz(n, a)), height=float(tr[k])))
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t, mz=float(m),
                      height=500.0) for m in range(300, 320)]
    return pd.DataFrame(rows)


def _ur_ledger(side):
    """A file's ledger: REAL and REJ on the uronium channels, and -- where the run
    opened the side channel -- their [M+NH4]+ readings, all Assigned."""
    ions = [(n, a) for n, a, _t in _IONS if a != NH4 or NH4 in side]
    peaks = pd.DataFrame([{"peak_id": f"{n}{a}", "mz": C.ion_mz(n, a), "height": 1e4} for n, a in ions])
    led = L.new_ledger(peaks)
    for n, a in ions:
        L.commit_assignment(led, f"{n}{a}", neutral_formula=n, adduct=a,
                            ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "+",
                            ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="Pass 1")
    led["tier"] = led["tier"].astype(object)
    led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
    led["below_assignability"] = False
    return led


def _run_ur_batch(tmp_path, monkeypatch, *, cfg=None):
    ts = _ts()
    seen = []

    def fake_assign(sid, context="ambient-air", **kw):
        side = tuple(kw["cfg"].side_channels or ())
        seen.append(side)
        return {"ledger": _ur_ledger(side),
                "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0, "side_channels": list(side),
                          "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [300.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    kw = {"cfg": cfg} if cfg is not None else {}
    res = AB.run(peaks=ts, ts_peaks=ts, reagent="Ur", batch="test batch", out_dir=str(tmp_path),
                 k_min=2, k_max=3, min_gain=0.0, n_jobs=1, residual=False, resolving_power=100_000,
                 log=lambda *a: None, **kw)
    return res, seen


def test_the_batch_records_the_side_channels_and_the_amine_gate_acts_on_their_readings(tmp_path, monkeypatch):
    res, seen = _run_ur_batch(tmp_path, monkeypatch)
    assert seen and all(s == (NH4,) for s in seen)                    # the profile's channel reached every file
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["side_channels"] == [NH4] and summ["side_channels_files"] == {NH4: len(seen)}
    assert summ["side_channels_requested"] == [NH4] and summ["side_channels_source"] == "profile Ur"
    assert all(pf["side_channels"] == [NH4] for pf in summ["per_file"])
    # THE AMINE GATE on the side channel's readings, at the merge: the reading that tracks its own
    # [M+H]+ / urea parent stays [M+NH4]+; the one that does not is re-read as the protonated amine
    assert summ["merge_gates"]["amine"] == {"relabeled": 1, "kept_covary": 1, "kept_protected": 0,
                                            "kept_si": 0, "forced_nh4": 0}
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    real = merged[(merged.neutral_formula == REAL) & (merged.adduct == NH4)]
    assert len(real) == 1 and real.tier.iloc[0] == "Assigned"
    amine = merged[(merged.neutral_formula == "C10H19NO2") & (merged.adduct == "[M+H]+")]
    assert len(amine) == 1 and amine.tier.iloc[0] == "Candidate"
    assert "not confirmed (independent time trace" in str(amine.tier_reason.iloc[0])
    assert merged[(merged.neutral_formula == REJ) & (merged.adduct == NH4)].empty
    # ... and on the evidence scale (the same gate, AmineGate, over the pooled per-file pairs)
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    by = {(r.neutral_formula, r.adduct): r for r in ev.itertuples()}
    assert str(by[(REAL, NH4)].nh4_gate_detail).startswith(f"{REAL}: keep (tracks its own [M+H]+/urea parent")
    assert str(by[(REJ, NH4)].nh4_gate_detail).startswith(f"{REJ}: reject (independent time trace")
    assert "amine default" in str(by[(REJ, NH4)].split_how)


def test_an_explicit_empty_choice_closes_the_uronium_channel_and_says_so(tmp_path, monkeypatch):
    res, seen = _run_ur_batch(tmp_path, monkeypatch, cfg=PCfg.PassConfig(side_channels=()))
    assert seen and all(s == () for s in seen)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["side_channels"] == [] and summ["side_channels_files"] == {}
    assert summ["side_channels_requested"] == [] and summ["side_channels_source"] == "explicit config"
    assert summ["merge_gates"]["amine"]["relabeled"] == 0 and summ["merge_gates"]["amine"]["kept_covary"] == 0


def test_the_nh4_admissibility_rule_reads_a_side_channel_run_by_ion_composition():
    """A side-channel [M+NH4]+ reading of Y is not a uronium adduct ion of Y: the
    rule asks for Y's [M+H]+ or urea ion, under any reading, in any ledger."""
    merged = pd.DataFrame(dict(neutral_formula=["C9H16O", "C10H16O2"], adduct=[NH4, NH4]))
    per_file = {"f1": pd.DataFrame(dict(neutral_formula=["C9H16O"], adduct=["[M+H]+"], role=["M0"],
                                        mz=[141.13], height=[1e4]))}
    g = SPL.AmineGate(merged, None, set(), scan=(100.0, 400.0), per_file=per_file)
    assert g.nh4_admissibility("C9H16O")["ok"] and g.nh4_admissibility("C9H16O")["present"] == ["[M+H]+"]
    bad = g.nh4_admissibility("C10H16O2")              # only its own NH4 reading: inadmissible
    assert not bad["ok"] and bad["why"].startswith("no uronium adduct ion of Y present")


# --------------------------------------------------------------------------- the CLI and the pipeline
@pytest.mark.parametrize("cmd", [["assign", "--sample-id", "S"], ["batch", "--batch", "B"],
                                 ["pool", "--batches", "B.*"]])
def test_the_flag_parses_on_assign_batch_and_pool(cmd):
    ap = cli.build_parser()
    assert ap.parse_args(cmd).side_channels is None
    assert ap.parse_args(cmd + ["--side-channels", "[M+CO3]-", "[M+Br2]-"]).side_channels == ["[M+CO3]-", "[M+Br2]-"]
    assert cli._side_channels_arg(ap.parse_args(cmd + ["--side-channels", "none"]).side_channels) == ()


def test_the_flag_value_is_checked():
    assert cli._side_channels_arg(None) is None
    assert cli._side_channels_arg(["None"]) == ()
    assert cli._side_channels_arg(["[M+CO3]-", "[M+CO3]-"]) == ("[M+CO3]-",)
    with pytest.raises(SystemExit, match="no server mechanism for \\[M\\+K\\]\\+"):
        cli._side_channels_arg(["[M+K]+"])
    with pytest.raises(SystemExit, match="give it alone"):
        cli._side_channels_arg(["none", "[M+CO3]-"])


def test_batch_and_pool_hand_the_flag_to_the_pipeline(monkeypatch):
    got = {}
    res = {"ctx": types.SimpleNamespace(out_dir="out", run_id="rid"), "assign": {"summary": {}}}
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(PL, "run_batch", lambda **kw: (got.__setitem__("batch", kw), res)[1])
    monkeypatch.setattr(PL, "run_pooled_batches", lambda **kw: (got.__setitem__("pool", kw), res)[1])
    cli.cmd_batch(cli.build_parser().parse_args(["batch", "--batch", "B", "--side-channels", "[M+CO3]-"]))
    cli.cmd_pool(cli.build_parser().parse_args(["pool", "--batches", "B.*", "--side-channels", "none"]))
    assert got["batch"]["side_channels"] == ("[M+CO3]-",) and got["pool"]["side_channels"] == ()
    got.clear()
    cli.cmd_batch(cli.build_parser().parse_args(["batch", "--batch", "B"]))
    assert got["batch"]["side_channels"] is None                      # unset: the profile decides


def test_assign_stamps_the_flag_on_the_cfg(monkeypatch, tmp_path):
    class _Stop(Exception):
        pass

    got = {}

    def stop(sid, context, **kw):
        got["cfg"] = kw["cfg"]
        raise _Stop

    monkeypatch.setattr(cli, "_require_creds", lambda: None)
    monkeypatch.setattr(A, "run", stop)
    base = ["assign", "--sample-id", "X", "--output-dir", str(tmp_path)]
    with pytest.raises(_Stop):
        cli.cmd_assign(cli.build_parser().parse_args(base + ["--reagent", "Ur"]))
    assert got["cfg"].side_channels == (NH4,)
    with pytest.raises(_Stop):
        cli.cmd_assign(cli.build_parser().parse_args(base + ["--reagent", "Ur", "--side-channels", "none"]))
    assert got["cfg"].side_channels == ()
    with pytest.raises(_Stop):           # a forced adduct list has no profile: closed unless asked for
        cli.cmd_assign(cli.build_parser().parse_args(base + ["--adducts", "[M+H]+"]))
    assert got["cfg"].side_channels == ()


def test_the_pipeline_stamps_the_choice_before_the_manifest_snapshot(monkeypatch, tmp_path):
    from peaky.reporting import provenance as PV

    class _Batches:
        def list(self, dataset=None):   # noqa: A001
            return pd.DataFrame({"sample_batch_id": ["B-id"], "sample_batch_name": ["B"]})

    got = {}
    ts = pd.DataFrame({"sample_item_id": ["s1", "s2"], "mz": [100.0, 100.0], "height": [5.0, 5.0],
                       "sample_batch_name": ["b1", "b2"]})
    monkeypatch.setattr(IO, "connect", lambda *a, **k: types.SimpleNamespace(batches=_Batches()))
    monkeypatch.setattr(AB, "run", lambda **kw: got.__setitem__("cfg", kw["cfg"]) or {"summary": {}, "sample_ids": []})
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **kw: {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: got.__setitem__("rec", kw))
    PL.run_batch(batch="B", dataset="D", reagent="Ur", base_out=str(tmp_path), ts=ts, do_report=False,
                 log=lambda *a: None)
    assert got["cfg"].side_channels == (NH4,) and got["rec"]["cfg"].side_channels == (NH4,)
    PL.run_batch(batch="B", dataset="D", reagent="Ur", base_out=str(tmp_path / "closed"), ts=ts,
                 do_report=False, side_channels=(), log=lambda *a: None)
    assert got["cfg"].side_channels == () and got["rec"]["cfg"].side_channels == ()


# --------------------------------------------------------------------------- the scorecard's arms
def test_the_decoy_arms_open_what_the_run_recorded_and_the_wrong_adducts_arm_does_not(monkeypatch):
    got = []

    def fake_run(sid, context="ambient-air", **kw):
        got.append(kw["cfg"].side_channels)
        return {"ledger": pd.DataFrame()}

    monkeypatch.setattr(A, "run", fake_run)
    run = types.SimpleNamespace(profile=P.UR, adducts=list(P.UR.adducts),
                                summary={"side_channels": [NH4], "side_channels_requested": [NH4]})
    peaks = pd.DataFrame({"peak_id": ["a"], "mz": [100.0], "height": [1.0]})
    assert SC.run_side_channels(run) == (NH4,)
    SC.run_engine_offline(run, peaks, "side-arm", list(P.UR.adducts))
    SC.run_engine_offline(run, peaks, "side-arm", ["[M+Na]+"])          # the wrong-adducts arm
    assert got == [(NH4,), ()]
    # a run that recorded none (made before the record existed) opens none, whatever its profile declares
    got.clear()
    old = types.SimpleNamespace(profile=P.UR, adducts=list(P.UR.adducts), summary={})
    assert SC.run_side_channels(old) == ()
    SC.run_engine_offline(old, peaks, "side-arm", list(P.UR.adducts))
    assert got == [()]
