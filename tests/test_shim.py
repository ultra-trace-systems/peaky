"""The `mascope_assign` -> `peaky` back-compat shim.

The package was renamed peaky; `mascope_assign` must remain importable as an alias
so existing code/notebooks keep working. Run: python3 tests/test_shim.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
import mascope_assign  # noqa: E402

# the alias IS the real package object (sys.modules swap)
check("import mascope_assign is peaky", mascope_assign is peaky,
      f"{mascope_assign!r} vs {peaky!r}")

# submodule access through the old path resolves to peaky's submodule
ma_cli = importlib.import_module("mascope_assign.cli")
pk_cli = importlib.import_module("peaky.cli")
check("mascope_assign.cli is peaky.cli", ma_cli is pk_cli)

# `from mascope_assign import X` goes through peaky's lazy __getattr__
from mascope_assign import PassConfig  # noqa: E402
from peaky import PassConfig as _PC  # noqa: E402
check("from mascope_assign import PassConfig works", PassConfig is _PC)

# version is the renamed package's version, and that is the INSTALLED one: a
# literal here went stale against pyproject and got stamped on published runs
# as the engine version, making two releases indistinguishable in the store.
from importlib.metadata import version as _dist_version  # noqa: E402

check("mascope_assign.__version__ == peaky.__version__",
      mascope_assign.__version__ == peaky.__version__,
      f"{mascope_assign.__version__} / {peaky.__version__}")
check("peaky.__version__ is the installed distribution's",
      peaky.__version__ == _dist_version("mascope-peaky"),
      f"{peaky.__version__} / {_dist_version('mascope-peaky')}")


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
