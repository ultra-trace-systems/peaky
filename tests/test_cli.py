"""Offline tests for the CLI (cli.py): parser wiring, reagent resolution for
explicit profiles (no network), friendly server-error hints, the --env override,
and the offline `gka` subcommand. Run: python3 tests/test_cli.py"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import cli, gka_widget, profiles  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---- gka_widget is importable FROM THE PACKAGE (it moved out of scripts/) ----
check("gka_widget moved into the package", hasattr(gka_widget, "build_points"))

# ---- parser wiring -----------------------------------------------------------
P = cli.build_parser()
a = P.parse_args(["list", "datasets"])
check("parse `list datasets`", a.what == "datasets" and a.func is cli.cmd_list)
a = P.parse_args(["list", "batches", "--dataset", "Test Workspace"])
check("parse `list batches --dataset`", a.what == "batches" and a.dataset == "Test Workspace")
a = P.parse_args(["assign", "--sample-id", "XYZ"])
check("parse `assign` defaults reagent=auto", a.func is cli.cmd_assign and a.reagent == "auto"
      and a.sample_id == "XYZ")
a = P.parse_args(["assign", "--sample-id", "X", "--reagent", "Br",
                  "--adducts", "[M+Br]-", "[M-H]-"])
check("parse `assign --adducts` (multi)", a.adducts == ["[M+Br]-", "[M-H]-"] and a.reagent == "Br")
a = P.parse_args(["--env", "/tmp/x.env", "list", "datasets"])
check("top-level --env is parsed", a.env == "/tmp/x.env")

# ---- batch selection knobs (presence set-cover; one selector, no strategy flag) ----
from peaky.batch import sampling as _SS  # noqa: E402
a = P.parse_args(["batch", "--batch", "B"])
check("batch selection defaults mirror sampling.K_MIN/K_MAX/MIN_GAIN",
      a.func is cli.cmd_batch and a.k_min == _SS.K_MIN and a.k_max == _SS.K_MAX
      and a.min_gain == _SS.MIN_GAIN, vars(a))
a = P.parse_args(["batch", "--batch", "B", "--k-max", "12", "--k-min", "4", "--min-gain", "0.01"])
check("parse `batch --k-max/--k-min/--min-gain`",
      a.k_max == 12 and a.k_min == 4 and a.min_gain == 0.01)
for flag in ("--select", "--coverage-target", "--height-floor"):
    try:
        P.parse_args(["batch", "--batch", "B", flag, "x"])
        check(f"removed flag {flag} is rejected", False, "parsed")
    except SystemExit:
        check(f"removed flag {flag} is rejected", True)
a = P.parse_args(["assign", "--sample-id", "X"])
# BOTH default to None: the absolute gate is opt-in, and an unset x-edge is what
# lets the resolution fall through to the reagent profile / the package default.
check("assign: --height-cutoff and --height-cutoff-x-edge both default to None",
      a.height_cutoff is None and a.height_cutoff_x_edge is None)
a = P.parse_args(["assign", "--sample-id", "X", "--height-cutoff-x-edge", "5"])
check("assign: --height-cutoff-x-edge parses as a float", a.height_cutoff_x_edge == 5.0)

# ---- one flag, two parsers: `peaky assign` and assign.main (the module entry
# point) both expose --height-cutoff-x-edge, and PassConfig.height_cutoff returns
# the absolute height_cutoff_cps whenever it is set -- so BOTH help texts have to
# tell the reader that passing --height-cutoff makes the multiple inert. They had
# drifted: only assign.main said it.
import argparse as _argparse  # noqa: E402
import contextlib as _contextlib  # noqa: E402
import io as _io  # noqa: E402

from peaky.assignment import assign as _assign_mod  # noqa: E402

_sub = next(x for x in P._actions if isinstance(x, _argparse._SubParsersAction))
_cli_x_help = next(x.help for x in _sub.choices["assign"]._actions
                   if x.dest == "height_cutoff_x_edge")
_buf = _io.StringIO()
try:
    with _contextlib.redirect_stdout(_buf):
        _assign_mod.main(["--help"])
except SystemExit:
    pass
_mod_help = " ".join(_buf.getvalue().split())
_INERT = "ignored when --height-cutoff is given"
check("both --height-cutoff-x-edge help texts say the absolute cutoff wins",
      _INERT in " ".join((_cli_x_help or "").split()) and _INERT in _mod_help,
      {"cli": _cli_x_help, "assign.main says it": _INERT in _mod_help})

# subcommand is required
try:
    P.parse_args([])
    check("no subcommand -> error", False, "did not raise")
except SystemExit:
    check("no subcommand -> error", True)

# ---- setup command + output-dir resolution ----------------------------------
a = P.parse_args(["setup"])
check("parse `setup`", a.func is cli.cmd_setup)
check("resolve_out_dir: --out-dir wins", cli.resolve_out_dir("/x/y") == os.path.expanduser("/x/y"))
os.environ["PEAKY_OUTPUT_DIR"] = "/tmp/peaky_test_out"
check("resolve_out_dir: $PEAKY_OUTPUT_DIR honored when no --out-dir",
      cli.resolve_out_dir(None) == "/tmp/peaky_test_out")
os.environ.pop("PEAKY_OUTPUT_DIR", None)
# cmd_setup scaffolds a workspace (no network: clear creds so it skips the connect check)
import tempfile  # noqa: E402
_saved = {k: os.environ.pop(k, None) for k in ("MASCOPE_URL", "MASCOPE_ACCESS_TOKEN")}
_orig_root = cli._workspace_root
try:
    with tempfile.TemporaryDirectory() as _d:
        open(os.path.join(_d, ".env.example"), "w").write("MASCOPE_URL=\nMASCOPE_ACCESS_TOKEN=\n")
        cli._workspace_root = lambda: _d
        cli.cmd_setup(SimpleNamespace())
        _env = open(os.path.join(_d, ".env")).read()
        check("setup creates .env + output/ + sets PEAKY_OUTPUT_DIR to the workspace output/",
              os.path.isdir(os.path.join(_d, "output"))
              and f"PEAKY_OUTPUT_DIR={os.path.join(_d, 'output')}" in _env)
finally:
    cli._workspace_root = _orig_root
    for _k, _v in _saved.items():
        if _v is not None:
            os.environ[_k] = _v

# ---- reagent resolution: explicit profile name needs NO network --------------
ns = SimpleNamespace(adducts=None, reagent="Br", context=None, sample_id="X", no_cache=False)
ad, ctx, note = cli._resolve_reagent(ns)
check("resolve --reagent Br -> Br adducts", ad == list(profiles.BR.adducts), ad)
check("resolve --reagent Br -> Br context", ctx == profiles.BR.context, ctx)
check("resolve --reagent Br -> labelled note", "Br" in note, note)

ns = SimpleNamespace(adducts=None, reagent="uronium", context=None, sample_id="X", no_cache=False)
ad, ctx, note = cli._resolve_reagent(ns)
check("resolve alias 'uronium' -> Ur context", ctx == profiles.UR.context, ctx)

# explicit --adducts overrides reagent, no network
ns = SimpleNamespace(adducts=["[M+Na]+"], reagent="auto", context="chamber", sample_id="X", no_cache=False)
ad, ctx, note = cli._resolve_reagent(ns)
check("explicit --adducts wins", ad == ["[M+Na]+"] and ctx == "chamber", (ad, ctx))

# with_profile= is opt-in: the 3-tuple callers above are untouched, and the 4th
# item is the profile the gate multiple is read from (None when it was forced).
ns = SimpleNamespace(adducts=None, reagent="Br", context=None, sample_id="X", no_cache=False)
_r4 = cli._resolve_reagent(ns, with_profile=True)
check("with_profile=True appends the resolved profile",
      len(_r4) == 4 and _r4[:3] == cli._resolve_reagent(ns)
      and _r4[3] is profiles.BR, _r4)
ns = SimpleNamespace(adducts=["[M+Na]+"], reagent="auto", context=None, sample_id="X", no_cache=False)
check("forced --adducts surfaces NO profile (nothing to read a multiple from)",
      cli._resolve_reagent(ns, with_profile=True)[3] is None)

# the labelled-reagent isotopic purity rides on that same profile -- cmd_assign
# reads it off the 4th item rather than from a 4-tuple of its own.
ns = SimpleNamespace(adducts=None, reagent="NO3_15N", context=None, sample_id="X", no_cache=False)
_pr = cli._resolve_reagent(ns, with_profile=True)[3]
check("a labelled profile still surfaces its purity through with_profile=",
      _pr is not None and _pr.purity == profiles.NO3_15N.purity, getattr(_pr, "purity", None))

# ---- friendly server-error hints ---------------------------------------------
check("403 -> WAF hint", "WAF" in (cli._friendly_server_error(RuntimeError("HTTP 403 Attention Required")) or ""))
check("401 -> token hint", "token" in (cli._friendly_server_error(RuntimeError("401 Unauthorized")) or "").lower())
check("no-peaks -> list hint", "list" in (cli._friendly_server_error(RuntimeError("no peaks returned for sample Z")) or ""))
check("unknown error -> no hint", cli._friendly_server_error(ValueError("boom")) is None)

# ---- offline `gka` subcommand + --env override -------------------------------
with tempfile.TemporaryDirectory() as d:
    led = pd.DataFrame({"mz": [200.1, 214.1, 99.9], "height": [1e5, 5e4, 30.0],
                        "role": ["M0", "M0", "unexplained"],
                        "tier": ["Assigned", "Candidate", ""]})
    csv = os.path.join(d, "led.csv"); led.to_csv(csv, index=False)
    out = os.path.join(d, "w.html")
    rc = cli.main(["--env", os.path.join(d, "creds.env"), "gka", csv, "-o", out])
    check("`gka` subcommand returns 0", rc == 0, rc)
    check("`gka` writes an HTML file",
          os.path.exists(out) and "<html" in Path(out).read_text(encoding="utf-8"))
    check("--env sets MASCOPE_ENV", os.environ.get("MASCOPE_ENV") == os.path.join(d, "creds.env"))

# ---- reach-through: the parsed flags actually reach the pipeline ------------
# The checks above are parse-only, so swapping two flags in cmd_batch would pass.
# Drive the real cmd_* with the credential check and the pipeline stubbed, and
# read back the kwargs the command handed over.
from peaky import pipeline as PL  # noqa: E402
from peaky.assignment import assign as _A  # noqa: E402

_seen = {}
_saved = {"require": cli._require_creds, "run_batch": PL.run_batch,
          "run_pooled": PL.run_pooled_batches, "assign_run": _A.run}


def _rec(name):
    def f(**kw):
        _seen[name] = kw
        return {"ctx": SimpleNamespace(out_dir="/tmp/peaky-test-run", run_id="rid")}
    return f


class _StopAssign(Exception):
    """Cut cmd_assign off right after it builds the config."""


def _capture_cfg(sample_id, context="ambient-air", *, cfg=None, **kw):
    _seen["assign"] = {"sample_id": sample_id, "context": context, "cfg": cfg, **kw}
    raise _StopAssign


cli._require_creds = lambda: None
PL.run_batch, PL.run_pooled_batches = _rec("batch"), _rec("pool")
_A.run = _capture_cfg
try:
    cli.cmd_batch(P.parse_args(["batch", "--batch", "B", "--dataset", "D",
                                "--k-max", "12", "--k-min", "4", "--min-gain", "0.02"]))
    check("cmd_batch forwards k_max / k_min / min_gain to pipeline.run_batch",
          (_seen["batch"]["k_max"], _seen["batch"]["k_min"], _seen["batch"]["min_gain"])
          == (12, 4, 0.02), _seen.get("batch"))
    cli.cmd_batch(P.parse_args(["batch", "--batch", "B"]))
    check("cmd_batch forwards the sampling DEFAULTS when no flag is given",
          (_seen["batch"]["k_max"], _seen["batch"]["k_min"], _seen["batch"]["min_gain"])
          == (_SS.K_MAX, _SS.K_MIN, _SS.MIN_GAIN), _seen.get("batch"))
    cli.cmd_pool(P.parse_args(["pool", "--batches", "chamber.*", "--k-max", "9",
                               "--k-min", "5", "--min-gain", "0.03"]))
    check("cmd_pool forwards k_max / k_min / min_gain to run_pooled_batches",
          (_seen["pool"]["k_max"], _seen["pool"]["k_min"], _seen["pool"]["min_gain"])
          == (9, 5, 0.03), _seen.get("pool"))
    check("cmd_pool forwards --group-by too (the knobs are not positional luck)",
          _seen["pool"]["group_by"] == "sample_batch_name", _seen.get("pool"))
    with tempfile.TemporaryDirectory() as _d:
        try:
            cli.cmd_assign(P.parse_args(["assign", "--sample-id", "X", "--reagent", "Br",
                                         "--height-cutoff-x-edge", "2.5",
                                         "--output-dir", _d]))
            check("cmd_assign reaches assign.run", False, "no call")
        except _StopAssign:
            _cfg = _seen["assign"]["cfg"]
            check("cmd_assign passes --height-cutoff-x-edge into the PassConfig",
                  _cfg.height_cutoff_x_edge == 2.5 and _cfg.height_cutoff_cps is None,
                  vars(_cfg) if _cfg is not None else None)
        try:
            cli.cmd_assign(P.parse_args(["assign", "--sample-id", "X", "--reagent", "Br",
                                         "--height-cutoff", "250", "--output-dir", _d]))
        except _StopAssign:
            _cfg = _seen["assign"]["cfg"]
            check("cmd_assign passes --height-cutoff as the ABSOLUTE override",
                  _cfg.height_cutoff_cps == 250.0 and _cfg.height_cutoff == 250.0,
                  vars(_cfg) if _cfg is not None else None)
        # no flag + a bundled profile -> the package default, unchanged
        try:
            cli.cmd_assign(P.parse_args(["assign", "--sample-id", "X", "--reagent", "Br",
                                         "--output-dir", _d]))
        except _StopAssign:
            check("cmd_assign with no x-edge flag keeps the package default",
                  _seen["assign"]["cfg"].height_cutoff_x_edge == 1.0)
        # ... but a profile that carries its own multiple supplies it
        _pick = profiles.ReagentProfile(
            name="TofPick", label="tof picker", polarity="-", adducts=["[M-H]-"],
            normaliser="tic", reagent_ion_re=None, ranges="C0-10 H0-20",
            detect_adduct=None, height_cutoff_x_edge=5.0)
        _snap = (dict(profiles.PROFILES), dict(profiles._BY_ALIAS))
        profiles.register(_pick)
        try:
            try:
                cli.cmd_assign(P.parse_args(["assign", "--sample-id", "X",
                                             "--reagent", "TofPick", "--output-dir", _d]))
            except _StopAssign:
                check("cmd_assign takes the multiple from the reagent profile",
                      _seen["assign"]["cfg"].height_cutoff_x_edge == 5.0,
                      vars(_seen["assign"]["cfg"]))
            try:
                cli.cmd_assign(P.parse_args(["assign", "--sample-id", "X",
                                             "--reagent", "TofPick",
                                             "--height-cutoff-x-edge", "2.5",
                                             "--output-dir", _d]))
            except _StopAssign:
                check("an explicit --height-cutoff-x-edge outranks the profile",
                      _seen["assign"]["cfg"].height_cutoff_x_edge == 2.5)
        finally:
            profiles.PROFILES.clear(); profiles.PROFILES.update(_snap[0])
            profiles._BY_ALIAS.clear(); profiles._BY_ALIAS.update(_snap[1])
finally:
    cli._require_creds = _saved["require"]
    PL.run_batch, PL.run_pooled_batches = _saved["run_batch"], _saved["run_pooled"]
    _A.run = _saved["assign_run"]


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
