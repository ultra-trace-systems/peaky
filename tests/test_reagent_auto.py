"""`--reagent auto` (the CLI default) reads the reagent from the server's own
ionization mechanisms and STOPS when none of them is a reagent's diagnostic adduct.

Pinned here, offline and on synthetic tables:
  * polarity comes from the mechanisms' own charge (standard or legacy spelling),
    then from a `polarity` column's words -- never from a batch or sample name;
  * no diagnostic adduct -> ValueError naming the mechanisms seen (no guess of the
    first registered profile of the polarity);
  * `peaky pool` resolves on the full pooled table, not the 4-column trim;
  * `peaky assign` and the MCP `assign_sample` tool stop with that reason instead
    of handing assign.run adducts=None (whose per-sample default is [M-H]-).
"""
import time
from datetime import datetime

import pandas as pd
import pytest

from peaky import cli
from peaky import mcp_server as M
from peaky import pipeline as PL
from peaky.assignment import assign as A
from peaky.batch import assign_batch as AB
from peaky.chem import profiles as P
from peaky.io import io_mascope as IO
from peaky.reporting import provenance as PV

# a batch name full of hyphens (as dated, instrument-coded names are), and one carrying a "+"
HYPHENS = "Instrument-A uronium run - zone-2 - zone-3"
PLUS = "PTR H3O+ run"


def _table(mechs, *, batch=HYPHENS, polarity=None, n=12):
    """A peak table shaped like a batch time series: `mechs` fill the first rows of
    `ionization_mechanism`, the rest are unmatched peaks."""
    d = pd.DataFrame({
        "sample_item_id": [f"s{i % 3}" for i in range(n)],
        "sample_item_name": [f"file-{i % 3}-2021-03-04-10-00" for i in range(n)],
        "sample_batch_name": batch,
        "datetime_utc": pd.Timestamp("2021-03-04"),
        "mz": [100.0 + 7.3 * i for i in range(n)],
        "height": 1e4,
        "ionization_mechanism": (list(mechs) + [None] * n)[:n],
    })
    if polarity is not None:
        d["polarity"] = polarity
    return d


# --------------------------------------------------------------------------- #
# polarity: the mechanisms' own charge, then a polarity column, never a name
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechs, sign", [
    (["[M+H]+"], "+"),
    (["+H+"], "+"),                     # legacy protonation
    (["-H+"], "-"),                     # legacy DEprotonation: the last char is not the charge
    (["+Br-", "-H+"], "-"),
    (["+"], "+"),                       # legacy bare molecular cation = [M]+.
    (["[M]-."], "-"),
    (["[M+(CH4N2O)H]+", "+H+"], "+"),
])
def test_polarity_is_read_from_the_mechanisms_charge(mechs, sign):
    assert P._detect_polarity(_table(mechs)) == sign


def test_both_polarities_stamped_is_no_polarity():
    assert P._detect_polarity(_table(["[M+H]+", "[M-H]-"])) is None


def test_the_polarity_column_is_read_when_no_mechanism_says():
    assert P._detect_polarity(_table([], polarity="Positive")) == "+"
    assert P._detect_polarity(_table([], polarity="negative")) == "-"
    assert P._detect_polarity(_table([], polarity="-")) == "-"
    # the mechanisms outrank the column
    assert P._detect_polarity(_table(["-H+"], polarity="positive")) == "-"


@pytest.mark.parametrize("batch", [HYPHENS, PLUS, "Iodide CIMS 2021-03-04"])
def test_the_batch_and_sample_names_are_never_read(batch):
    t = _table([], batch=batch)
    assert P._detect_polarity(t) is None
    with pytest.raises(ValueError, match="could not auto-detect reagent"):
        P.resolve("auto", t)


# --------------------------------------------------------------------------- #
# diagnostic adducts still resolve; anything less raises, naming what it saw
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechs, name", [
    (["+(CH4N2O)H+", "+H+"], "Ur"),
    (["[M+CH4N2O+H]+", "[M+H]+"], "Ur"),
    (["+NO3-", "+^NO3-", "-H+"], "NO3+NO3_15N"),
    (["+Br-", "+NO3-", "-H+"], "Br+NO3"),
    (["[M+I]-", "[M-H]-"], "I"),
    (["+"], "EasyIC"),                  # a weak signature, believed when alone
])
def test_a_diagnostic_adduct_resolves_whatever_the_name(mechs, name):
    assert P.resolve("auto", _table(mechs, batch=HYPHENS)).name == name


@pytest.mark.parametrize("mechs, polarity, seen, pol_word", [
    (["[M+H]+", "[M+Na]+"], None, "[M+H]+, [M+Na]+", "positive"),
    (["+H+"], None, "[M+H]+", "positive"),
    (["-H+"], None, "[M-H]-", "negative"),
    ([], "positive", "none", "positive"),
    ([], None, "none", "unknown"),
])
def test_no_diagnostic_adduct_raises_instead_of_guessing(mechs, polarity, seen, pol_word):
    with pytest.raises(ValueError) as e:
        P.resolve("auto", _table(mechs, polarity=polarity))
    msg = str(e.value)
    assert f"mechanisms seen: {seen};" in msg
    assert f"polarity {pol_word}" in msg
    assert "--reagent" in msg and "Ur" in msg and "Br" in msg
    # the CLI boundary maps 'not found' / 'no peaks' texts to a stale-id hint
    assert cli._friendly_server_error(e.value) is None


def test_a_generic_signature_cannot_fire_on_a_table_with_no_matches(monkeypatch):
    """detect_adducts' [M-H]- default is not a match: a profile declaring [M-H]- as
    its signature must not be picked for a table the server matched nothing in."""
    monkeypatch.setattr(P, "PROFILES", dict(P.PROFILES))
    monkeypatch.setattr(P, "_BY_ALIAS", dict(P._BY_ALIAS))
    P.register(P.ReagentProfile(
        name="Deprot", label="deprotonation only", polarity="-", adducts=["[M-H]-"],
        normaliser="tic", reagent_ion_re=None, ranges="C0-10 H0-20",
        detect_adduct="[M-H]-"))
    with pytest.raises(ValueError, match="mechanisms seen: none"):
        P.resolve("auto", _table([]))
    assert P.resolve("auto", _table(["-H+"])).name == "Deprot"


def test_recognised_adducts_has_no_default():
    assert IO.recognised_adducts(_table([])) == []
    assert IO.recognised_adducts(pd.DataFrame({"mz": [1.0]})) == []
    assert IO.detect_adducts(_table([])) == ["[M-H]-"]          # unchanged
    assert IO.recognised_adducts(_table(["+Br-", "-H+", "+Br-"])) == ["[M+Br]-", "[M-H]-"]


# --------------------------------------------------------------------------- #
# peaky pool: resolve on the full pooled table, then trim
# --------------------------------------------------------------------------- #
class _FakeClient:
    pass


def _pool(monkeypatch, tmp_path, ts):
    got = {}

    def fake_ab(**kw):
        got["ab"] = kw
        return {"summary": {}, "sample_ids": []}

    monkeypatch.setattr(AB, "run", fake_ab)
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **kw: {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: None)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: _FakeClient())
    PL.run_pooled_batches(batches="b.*", dataset="D", reagent="auto",
                          base_out=str(tmp_path), ts=ts, when=datetime(2021, 3, 4, 12),
                          do_report=False, per_group_reports=False,
                          log=lambda *a: None)
    return got


def test_pool_auto_resolves_on_the_untrimmed_table(monkeypatch, tmp_path):
    got = _pool(monkeypatch, tmp_path, _table(["+(CH4N2O)H+", "+H+"] * 3))
    assert got["ab"]["reagent"] == "Ur"
    # the workers still get the trimmed table
    assert "ionization_mechanism" not in got["ab"]["peaks"].columns


def test_pool_auto_stops_before_assigning_when_nothing_is_diagnostic(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="mechanisms seen: \\[M\\+H\\]\\+"):
        _pool(monkeypatch, tmp_path, _table(["+H+"] * 4))


# --------------------------------------------------------------------------- #
# peaky assign / MCP assign_sample: stop, never adducts=None
# --------------------------------------------------------------------------- #
def _stub_sample(monkeypatch, raw):
    calls = []
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: _FakeClient())
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: raw.copy())

    def fake_run(sample_id, context="ambient-air", **kw):
        calls.append(kw)
        return {"ledger": pd.DataFrame({"mz": [100.0], "height": [1.0], "role": ["M0"],
                                        "neutral_formula": ["C5H8O"], "adduct": ["[M+H]+"]}),
                "stats": {}}

    monkeypatch.setattr(A, "run", fake_run)
    return calls


def _assign_args(tmp_path, *extra):
    return cli.build_parser().parse_args(
        ["assign", "--sample-id", "S1", "--output-dir", str(tmp_path), *extra])


def test_cli_assign_auto_resolves_from_the_sample(monkeypatch, tmp_path):
    _stub_sample(monkeypatch, _table(["+(CH4N2O)H+", "+H+"]))
    ad, ctx, note, prof = cli._resolve_reagent(_assign_args(tmp_path), with_profile=True)
    assert prof.name == "Ur" and ad == list(P.UR.adducts) and "auto-detected Ur" in note


def test_cli_assign_auto_stops_instead_of_the_deprotonation_default(monkeypatch, tmp_path, capsys):
    calls = _stub_sample(monkeypatch, _table(["[M+H]+", "[M+Na]+"]))
    with pytest.raises(ValueError, match="could not auto-detect reagent"):
        cli._resolve_reagent(_assign_args(tmp_path))
    rc = cli._run_guarded(lambda: cli.cmd_assign(_assign_args(tmp_path)))
    err = capsys.readouterr().err
    assert rc == 1 and "could not auto-detect reagent" in err and "--reagent" in err
    assert calls == [], "assign.run must not run on a guessed reagent"


def test_cli_explicit_reagent_is_unchanged(monkeypatch, tmp_path):
    _stub_sample(monkeypatch, _table([]))          # nothing diagnostic: irrelevant here
    ad, ctx, note = cli._resolve_reagent(_assign_args(tmp_path, "--reagent", "Ur"))
    assert ad == list(P.UR.adducts) and ctx == P.UR.context
    ad, ctx, note = cli._resolve_reagent(_assign_args(tmp_path, "--adducts", "[M+H]+"))
    assert ad == ["[M+H]+"]


def _wait(jid, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = M.JOBS.get(jid)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish")


def test_mcp_assign_sample_auto(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "JOBS", M.JobManager())
    calls = _stub_sample(monkeypatch, _table(["+(CH4N2O)H+", "+H+"]))
    job = _wait(M.assign_sample("S1", output_dir=str(tmp_path))["job_id"])
    assert job.status == "done", job.view()
    assert calls[-1]["adducts"] == list(P.UR.adducts)

    calls = _stub_sample(monkeypatch, _table(["+H+"]))
    job = _wait(M.assign_sample("S2", output_dir=str(tmp_path))["job_id"])
    assert job.status == "error" and "could not auto-detect reagent" in job.error
    assert calls == []


# --------------------------------------------------------------------------- #
# assign.run's own per-sample default is no longer silent
# --------------------------------------------------------------------------- #
def _run_engine(monkeypatch, raw, adducts):
    from peaky.assignment import passes as PA

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: raw.copy())
    monkeypatch.setattr(IO, "resolve_mechanism_ids", lambda client, names: {})
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(IO, "score_candidates", lambda *a, **k: pd.DataFrame())
    lines = []
    A.run("SID", "ambient-air", cfg=PA.PassConfig(height_cutoff_cps=1.0), adducts=adducts,
          use_cache=False, log=lines.append)
    return [str(x) for x in lines if "[reagent] WARNING" in str(x)]


def test_assign_run_warns_when_it_falls_back_to_deprotonation(monkeypatch):
    raw = pd.DataFrame({"peak_id": [f"p{i}" for i in range(20)],
                        "mz": [120.0 + 9.1 * i for i in range(20)],
                        "height": [50.0 + 100 * i for i in range(20)]})
    assert len(_run_engine(monkeypatch, raw, None)) == 1
    assert _run_engine(monkeypatch, raw, ["[M-H]-"]) == []
