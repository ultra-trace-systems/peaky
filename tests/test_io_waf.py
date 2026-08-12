"""Offline tests for the Cloudflare-WAF retry wrapper. Run: python3 tests/test_io_waf.py

No network: the wrapper is a pure control-flow helper. Tests pass base_delay=0 so
there are no real sleeps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.io import io_mascope as IO  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


class _Boom(Exception):
    pass


# --- transient 521 twice, then succeeds ---
calls = {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise _Boom("ServerError [HTTP 521] <!DOCTYPE html> origin down")
    return "ok"
res = IO._with_waf_retry(flaky, tries=4, base_delay=0)
check("retries a transient 521 twice then returns", res == "ok" and calls["n"] == 3, calls)

# --- non-transient (404 missing endpoint) re-raises immediately, unmasked ---
calls2 = {"n": 0}
def missing_endpoint():
    calls2["n"] += 1
    raise ValueError("404 no such endpoint on this server")
try:
    IO._with_waf_retry(missing_endpoint, tries=4, base_delay=0)
    ok = False
except ValueError:
    ok = True
check("non-transient 404 re-raises immediately (1 call, no retry)",
      ok and calls2["n"] == 1, calls2)

# --- persistent WAF challenge exhausts retries then raises ---
calls3 = {"n": 0}
def always_waf():
    calls3["n"] += 1
    raise _Boom("HTTP 403 Attention Required (Cloudflare)")
try:
    IO._with_waf_retry(always_waf, tries=3, base_delay=0)
    ok3 = False
except _Boom:
    ok3 = True
check("exhausts retries then raises the last error", ok3 and calls3["n"] == 3, calls3)

# --- classifier ---
check("_is_transient: 521", IO._is_transient(Exception("HTTP 521")))
check("_is_transient: 403 WAF", IO._is_transient(Exception("403 Attention Required")))
check("_is_transient: read timeout", IO._is_transient(Exception("Read timed out (30s)")))
check("_is_transient: 502 gateway", IO._is_transient(Exception("502 Bad Gateway")))
check("_is_transient: NOT 404", not IO._is_transient(Exception("HTTP 404 not found")))
check("_is_transient: NOT a plain ValueError",
      not IO._is_transient(ValueError("bad formula C3H5ClO17")))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
