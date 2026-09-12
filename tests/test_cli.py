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

# ---- batch selection strategy flags -----------------------------------------
a = P.parse_args(["batch", "--batch", "B"])
check("batch defaults to representative selection",
      a.func is cli.cmd_batch and a.select == "representative"
      and a.coverage_target == 0.85 and a.k_max == 10 and a.height_floor == 1000.0)
a = P.parse_args(["batch", "--batch", "B", "--select", "brightest",
                  "--coverage-target", "0.9", "--k-max", "8", "--height-floor", "500"])
check("parse `batch --select brightest` + knobs",
      a.select == "brightest" and a.coverage_target == 0.9
      and a.k_max == 8 and a.height_floor == 500.0)

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
ad, ctx, note, pur = cli._resolve_reagent(ns)
check("resolve --reagent Br -> Br adducts", ad == list(profiles.BR.adducts), ad)
check("resolve --reagent Br -> Br context", ctx == profiles.BR.context, ctx)
check("resolve --reagent Br -> labelled note", "Br" in note, note)

ns = SimpleNamespace(adducts=None, reagent="uronium", context=None, sample_id="X", no_cache=False)
ad, ctx, note, pur = cli._resolve_reagent(ns)
check("resolve alias 'uronium' -> Ur context", ctx == profiles.UR.context, ctx)

# explicit --adducts overrides reagent, no network
ns = SimpleNamespace(adducts=["[M+Na]+"], reagent="auto", context="chamber", sample_id="X", no_cache=False)
ad, ctx, note, pur = cli._resolve_reagent(ns)
check("explicit --adducts wins", ad == ["[M+Na]+"] and ctx == "chamber", (ad, ctx))

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

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
