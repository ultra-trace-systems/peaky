"""stdio MCP: stdout carries JSON-RPC frames only.

Each check runs a child Python process and reads its real stdout and stderr,
because what matters is the file descriptor, not `sys.stdout`. The guard
check needs no `mcp` package; the end-to-end check runs a real stdio server
and is skipped when `mcp` is not installed. Run:
    python3 tests/test_mcp_stdio.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS = FAIL = 0
# unbuffered, as a stdio host may run us: a stray line is flushed at once, not lost
_ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def _run(code, stdin=None, timeout=60):
    return subprocess.run([sys.executable, "-c", code], input=stdin, capture_output=True,
                          text=True, timeout=timeout, cwd=ROOT,
                          env=_ENV)


# ---------- the guard alone: every kind of stray write goes to stderr ----------
_GUARD = r'''
import os, subprocess, sys
from peaky import mcp_server as M
with M.stdout_reserved_for_protocol():
    print("noise: print")
    sys.stdout.write("noise: sys.stdout.write\n")
    os.write(1, b"noise: os.write fd 1\n")
    subprocess.run([sys.executable, "-c", "print('noise: child process')"])
    os.system("echo noise: shell")
    sys.stdout.buffer.write(b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n')
    sys.stdout.buffer.flush()
print("after the block")
'''
r = _run(_GUARD)
check("guard: the child ran", r.returncode == 0, r.stderr[-400:])
check("guard: stdout holds the frame only, then the restored stdout",
      r.stdout == '{"jsonrpc": "2.0", "id": 1, "result": {}}\nafter the block\n', repr(r.stdout))
for what in ("print", "sys.stdout.write", "os.write fd 1", "child process", "shell"):
    check(f"guard: '{what}' lands on stderr", f"noise: {what}" in r.stderr, r.stderr[-400:])

# ---------- end to end: a real stdio server with a tool that prints ----------
_SERVER = r'''
import os, subprocess, sys
from peaky import mcp_server as M
def noisy() -> str:
    """Print on every channel, then answer."""
    print("noise: tool print")
    subprocess.run([sys.executable, "-c", "print('noise: tool child')"])
    return "quiet answer"
M._TOOLS.append(noisy)
M.serve(transport="stdio")
'''


def _stdio_session(timeout=60):
    p = subprocess.Popen([sys.executable, "-c", _SERVER], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
                         env=_ENV)
    killer = threading.Timer(timeout, p.kill)
    killer.start()
    lines = []

    def send(msg):
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()

    def read_reply(rid):
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            lines.append(line)
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == rid:
                return msg

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                         "clientInfo": {"name": "test", "version": "0"}}})
        init = read_reply(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "noisy", "arguments": {}}})
        call = read_reply(2)
        p.stdin.close()
        rest, err = p.stdout.read(), p.stderr.read()
        lines += rest.splitlines(keepends=True)
    finally:
        killer.cancel()
        p.kill()
        p.wait()
    return init, call, lines, err


if importlib.util.find_spec("mcp") is None:
    print("skip  end-to-end stdio server (the `mcp` extra is not installed)")
else:
    init, call, lines, err = _stdio_session()
    check("stdio: initialize answered", bool(init and "result" in init), (init, err[-400:]))
    text = json.dumps((call or {}).get("result", {}))
    check("stdio: the tool call answered", "quiet answer" in text, (call, err[-400:]))
    bad = []
    for line in lines:
        try:
            json.loads(line)
        except ValueError:
            bad.append(line)
    check("stdio: every stdout line is a JSON-RPC frame", lines and not bad, bad[:3])
    check("stdio: the tool's print and child output land on stderr",
          "noise: tool print" in err and "noise: tool child" in err, err[-400:])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
