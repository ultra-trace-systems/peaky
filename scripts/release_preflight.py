"""Am I about to release the code I think I am?

Answers the one question a release-prep checklist cannot answer by reading files:
*which* peaky does this interpreter actually import, and is that tree current?

The trap this exists for: the shared `.venv` has peaky installed EDITABLE, and an
editable install records ONE source path, fixed at `pip install -e` time. That path
is the main checkout, which is also where long-lived feature branches get parked. A
worktree's own code wins only because `''` (the cwd) leads `sys.path` -- so the same
command run one directory over imports a different, often far older, tree. It does
not fail; it just runs other code. Measured while preparing 0.8.0, the main checkout
was 124 commits behind and still reporting 0.5.0.

So this checks the LIVE import first and derives everything else from wherever that
landed -- never from the cwd, and never from the file you happen to be editing.

    python scripts/release_preflight.py              # fetches, then reports
    python scripts/release_preflight.py --no-fetch   # offline; "behind" is stale
    python scripts/release_preflight.py --expect 0.8.0

Exit status is 0 only when every check passes, so it can gate a release script.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

OK, WARN, BAD = "ok  ", "WARN", "FAIL"
_verdict: list[str] = []


def _say(status: str, label: str, detail: str = "") -> None:
    _verdict.append(status)
    print(f"  {status}  {label:<34} {detail}")


def _git(root: Path, *args: str) -> str:
    """git in `root`, never raising -- a preflight must report, not explode."""
    try:
        r = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:
        return ""


def _resolves_to(python: str, cwd: Path, *, isolated: bool) -> tuple[str, str]:
    """(path, version) of the peaky a real invocation would import.

    `isolated=True` reproduces a CONSOLE SCRIPT: `-P` stops python prepending the
    cwd (or the script's dir) to sys.path, which is exactly why `peaky ...` never
    sees the worktree you are standing in and always falls through to the editable
    install's recorded path. `isolated=False` is `python -c` / `-m` / pytest, where
    the cwd leads sys.path and the local tree wins.
    """
    flags = ["-P"] if isolated else []
    code = "import peaky; print(peaky.__file__); print(peaky.__version__)"
    try:
        r = subprocess.run([python, *flags, "-c", code], capture_output=True,
                           text=True, cwd=str(cwd), timeout=60)
        out = r.stdout.strip().splitlines()
        return (out[0], out[1]) if len(out) >= 2 else ("<not importable>", "?")
    except Exception as e:
        return (f"<{type(e).__name__}>", "?")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expect", metavar="X.Y.Z",
                    help="fail unless the resolved version is exactly this")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip `git fetch`; 'behind origin' then reflects the last "
                         "fetch, which before a release is usually not what you want")
    args = ap.parse_args(argv)

    print("\npeaky release preflight\n" + "=" * 23)

    # ---- 1. the tree you MEAN to release ----------------------------------- #
    # Anchored on the cwd, not on this file: a script run from elsewhere must audit
    # the checkout the operator is standing in, which is what a tag would capture.
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel") or "")
    if not root or not (root / "pyproject.toml").exists():
        print(f"\n  {BAD}  cwd is not a peaky checkout: {Path.cwd()}\n")
        return 1
    root = root.resolve()

    # ---- 2. does every entry point actually RUN that tree? ----------------- #
    print("\nIMPORT  (which tree each entry point really runs)")
    print(f"        cwd / release target       {root}")
    py = sys.executable
    for label, isolated in (("python -c / -m / pytest", False),
                            ("`peaky` console script", True)):
        path, version = _resolves_to(py, root, isolated=isolated)
        inside = path.startswith(str(root) + os.sep)
        _say(OK if inside else BAD, label, f"{version}  {path}"
             + ("" if inside else "  <- NOT this checkout"))
    if BAD in _verdict:
        print("\n        A console script puts its BIN dir on sys.path[0], never the")
        print("        cwd, so it resolves through the editable install's recorded")
        print("        path -- the checkout `pip install -e` was run in, whatever")
        print("        branch that happens to be parked on today. To drive THIS tree:")
        print(f"            python -m peaky.cli <args>        # from {root}")
        print(f"            PYTHONPATH={root} peaky <args>")
        print("        Or give the worktree its own venv: uv sync --frozen --extra dev")

    # ---- 3. the state of that tree ----------------------------------------- #
    print(f"\nTREE  ({root})")
    if not args.no_fetch:
        _git(root, "fetch", "-q", "origin", "main")

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(root, "rev-parse", "--short", "HEAD")
    print(f"        branch                     {branch}  @ {head}")

    dirty = [ln for ln in _git(root, "status", "--porcelain").splitlines()
             if not ln.startswith("??")]
    untracked = [ln for ln in _git(root, "status", "--porcelain").splitlines()
                 if ln.startswith("??")]
    _say(OK if not dirty else BAD, "no uncommitted changes",
         "clean" if not dirty else f"{len(dirty)} modified -- a release must be "
         f"reproducible from the tag: {', '.join(l[3:] for l in dirty[:4])}")
    if untracked:
        _say(WARN, "untracked files present",
             f"{len(untracked)} (not in the tag): "
             f"{', '.join(l[3:] for l in untracked[:4])}")

    behind = _git(root, "rev-list", "--count", "HEAD..origin/main")
    ahead = _git(root, "rev-list", "--count", "origin/main..HEAD")
    _say(OK if behind == "0" else BAD, "not behind origin/main",
         f"{behind} behind, {ahead} ahead"
         + ("" if behind == "0" else "  <- the release would MISS those commits"))

    up = _git(root, "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}")
    if up:
        unpushed = _git(root, "rev-list", "--count", f"{up}..HEAD")
        _say(OK if unpushed == "0" else BAD, f"pushed to {up}",
             "in sync" if unpushed == "0"
             else f"{unpushed} local commits not on the remote -- the tag would "
                  "point at code nobody else has")

    # ---- 4. every version string, read from THAT tree ---------------------- #
    print("\nVERSION  (all read from the tree above)")
    try:
        proj = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    except Exception as e:
        _say(BAD, "pyproject readable", f"{type(e).__name__}: {e}")
        return 1
    version = proj["version"]

    sources: dict[str, str | None] = {"pyproject": version}
    m = re.search(r'^name = "%s"\nversion = "([^"]+)"' % re.escape(proj["name"]),
                  (root / "uv.lock").read_text(), re.M) \
        if (root / "uv.lock").exists() else None
    sources["uv.lock"] = m.group(1) if m else None
    init_txt = (root / "peaky" / "__init__.py").read_text()
    m = re.search(r'^_FALLBACK_VERSION = "([^"]+)"', init_txt, re.M)
    sources["_FALLBACK_VERSION"] = m.group(1) if m else None
    cff = (root / "CITATION.cff").read_text() if (root / "CITATION.cff").exists() else ""
    m = re.search(r"^version:\s*(\S+)\s*$", cff, re.M)
    sources["CITATION.cff"] = m.group(1).strip('"') if m else None

    for name, got in sources.items():
        _say(OK if got == version else BAD, f"{name} agrees", str(got))

    # what the package REPORTS at runtime, resolved peaky's own way (pyproject ->
    # installed metadata -> literal). Run in a subprocess rooted at the tree, so a
    # broken resolution shows up here as a mismatch instead of crashing the check.
    out = subprocess.run([sys.executable, "-c",
                          "import peaky; print(peaky.__version__)"],
                         capture_output=True, text=True, cwd=str(root))
    reported = out.stdout.strip() or "<error>"
    _say(OK if reported == version else BAD, "peaky.__version__ at runtime", reported)

    if args.expect:
        _say(OK if version == args.expect else BAD,
             f"version is the expected {args.expect}", version)

    # ---- 5. release-day specifics ------------------------------------------ #
    print("\nRELEASE")
    released = dict(re.findall(
        r"^## \[(\d+\.\d+\.\d+)\]\s*[—-]\s*(\d{4}-\d{2}-\d{2})",
        (root / "CHANGELOG.md").read_text(), re.M))
    cl_date = released.get(version)
    _say(OK if cl_date else BAD, "CHANGELOG has a dated section",
         f"## [{version}] — {cl_date}" if cl_date
         else f"no dated `## [{version}]` heading -- still under [Unreleased]?")

    m = re.search(r'^date-released:\s*"?([\d-]+)"?\s*$', cff, re.M)
    cff_date = m.group(1) if m else None
    today = _dt.date.today().isoformat()
    if cff_date and cl_date:
        _say(OK if cff_date == cl_date else BAD, "CITATION.cff date matches CHANGELOG",
             f"{cff_date} vs {cl_date}")
    _say(OK if cff_date == today else WARN, "release date is today",
         f"{cff_date} (today is {today})"
         + ("" if cff_date == today else "  <- bump both if you ship on a later day"))

    tag = f"v{version}"
    local_tag = _git(root, "tag", "-l", tag)
    remote_tag = _git(root, "ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    _say(WARN if (local_tag or remote_tag) else OK, f"{tag} not yet tagged",
         "already exists -- this release was cut" if (local_tag or remote_tag)
         else "as expected; publishing mints a permanent Zenodo DOI")

    # ---- verdict ----------------------------------------------------------- #
    bad, warn = _verdict.count(BAD), _verdict.count(WARN)
    print("\n" + "=" * 23)
    if bad:
        print(f"NOT READY -- {bad} failed"
              + (f", {warn} warning(s)" if warn else "") + "\n")
        return 1
    print(f"READY to release {version}"
          + (f" -- {warn} warning(s), read them" if warn else "") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
