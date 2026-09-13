"""Every version string in the repo, and the ONE source of truth behind each.

peaky states its version in five places, and they do not all mean the same thing:

  pyproject.toml [project].version   the version UNDER DEVELOPMENT. The source of
                                     truth; everything below derives from it.
  peaky.__version__                  resolved FROM pyproject at first access
                                     (peaky/__init__.py `_resolve_version`), so it
                                     cannot be restated wrong -- which it once was,
                                     stamping `code.package_version` 0.5.0 on every
                                     run folder produced by 0.7.0 code.
  peaky._FALLBACK_VERSION            last-resort literal for a copy with neither
                                     pyproject nor installed metadata.
  uv.lock                            the pinned graph; CI runs it with --frozen.
  CITATION.cff version/date-released the last release actually TAGGED, published
                                     and archived. It LAGS pyproject on purpose --
                                     a citation must resolve to something Zenodo
                                     holds -- so it is pinned to a *released*
                                     CHANGELOG heading, not to pyproject.

Run: python3 tests/test_versioning.py
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


import peaky  # noqa: E402

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
VERSION = PYPROJECT["version"]


def _parts(v):
    """(1, 2, 3) from '1.2.3' -- comparable, and tolerant of a suffix like -rc1."""
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", v).groups())


# --- the package derives from pyproject, and so does its fallback ------------
check("peaky.__version__ is pyproject's [project].version",
      peaky.__version__ == VERSION, f"{peaky.__version__} / {VERSION}")
check("the _FALLBACK_VERSION literal still matches pyproject",
      peaky._FALLBACK_VERSION == VERSION,
      f"fallback {peaky._FALLBACK_VERSION} / pyproject {VERSION}")
# the literal is a LAST resort: with pyproject present it must never be consulted,
# or a stale literal could mask the real version instead of merely standing in.
check("_resolve_version reads pyproject, not the literal, from a checkout",
      peaky._resolve_version() == VERSION)

# --- the lockfile pins the same version -------------------------------------
# CI runs `uv run --frozen`, which fails if uv.lock is out of date with pyproject.
# Catch it here too, so a version bump that forgot `uv lock` fails offline first.
_lock = (ROOT / "uv.lock").read_text()
_m = re.search(r'^name = "%s"\nversion = "([^"]+)"' % re.escape(PYPROJECT["name"]),
               _lock, re.M)
check("uv.lock pins the same version for this package",
      _m is not None and _m.group(1) == VERSION,
      f"uv.lock {_m.group(1) if _m else None} / pyproject {VERSION}")

# --- the CHANGELOG has a section for the version under development ----------
CHANGELOG = (ROOT / "CHANGELOG.md").read_text()
# `## [x.y.z] - YYYY-MM-DD` or `## [x.y.z] — YYYY-MM-DD` (both dashes are in use)
RELEASED = {v: d for v, d in re.findall(
    r"^## \[(\d+\.\d+\.\d+)\]\s*[—-]\s*(\d{4}-\d{2}-\d{2})", CHANGELOG, re.M)}
check("the CHANGELOG has released sections with parseable dates", len(RELEASED) >= 2,
      sorted(RELEASED))
check("an [Unreleased] section is open for work in progress",
      "## [Unreleased]" in CHANGELOG)

# --- CITATION.cff names a REAL, archived release ----------------------------
CFF = (ROOT / "CITATION.cff").read_text()
_cff_version = re.search(r"^version:\s*(\S+)\s*$", CFF, re.M).group(1).strip('"')
_cff_date = re.search(r'^date-released:\s*"?([\d-]+)"?\s*$', CFF, re.M).group(1)

check("CITATION.cff names a version the CHANGELOG records as released",
      _cff_version in RELEASED,
      f"{_cff_version} not among {sorted(RELEASED)}")
check("CITATION.cff's date-released is that release's own CHANGELOG date",
      RELEASED.get(_cff_version) == _cff_date,
      f"cff {_cff_date} / changelog {RELEASED.get(_cff_version)}")
# It may lag pyproject (a release not yet cut) but must never RUN AHEAD of it:
# that would cite a version whose code does not exist.
check("CITATION.cff never runs ahead of pyproject",
      _parts(_cff_version) <= _parts(VERSION),
      f"cff {_cff_version} > pyproject {VERSION}")
check("CITATION.cff carries the concept DOI the README badge points at",
      re.search(r'^doi:\s*"10\.5281/zenodo\.\d+"', CFF, re.M) is not None)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
