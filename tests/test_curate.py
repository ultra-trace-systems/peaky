"""Offline tests for curate.py (the Mascope write/curation engine).

No network: a fake transport captures every (method, path, body, params) the client
would send, so we assert the reverse-engineered endpoint contract exactly. Dry-run
and the delete confirm-gate are tested without any transport at all.

Run: python3 tests/test_curate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.io import curate as CU  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


class FakeResp:
    """Minimal requests.Response stand-in."""
    def __init__(self, data=None, status=200):
        self._data = data
        self.status_code = status
        self.content = b"{}" if data is None else b'{"data":{}}'
        self.ok = status < 400

        class _Req:
            method = "?"
            url = "?"
        self.request = _Req()
        self.url = "?"

    def json(self):
        return {"data": self._data}


def client(monkeypatch_calls, *, data=None, dry_run=False, cookie="session=test"):
    """A CurationClient whose HTTP layer records calls into `monkeypatch_calls`.
    Defaults to carrying a session cookie so write endpoints are exercised (the
    server gates writes behind a cookie; the no-cookie guard is tested separately)."""
    c = CU.CurationClient(url="https://x.test", token="t", dry_run=dry_run,
                          cookie=cookie)

    def fake_request(method, full, headers=None, json=None, params=None,
                     timeout=None, verify=None):
        monkeypatch_calls.append({"method": method, "url": full, "body": json,
                                  "params": params, "headers": headers or {}})
        return FakeResp(data=data)

    import peaky.io.curate as mod
    mod.requests.request = fake_request  # type: ignore[attr-defined]
    return c


# ---------- envelope + path shape ----------
calls = []
c = client(calls, data={"workspace_id": "W1"})
res = c.create_workspace("Zero-air campaign", "blanks")
check("create_workspace method+path",
      calls[-1]["method"] == "POST" and calls[-1]["url"].endswith("/api/workspaces"),
      calls[-1])
check("create_workspace body",
      calls[-1]["body"] == {"workspace_name": "Zero-air campaign",
                            "workspace_description": "blanks"}, calls[-1]["body"])
check("envelope unwrap -> data", res == {"workspace_id": "W1"}, res)
check("action recorded + executed",
      c.actions[-1].op == "create_workspace" and c.actions[-1].executed)

# ---------- dataset create / update / delete / move ----------
calls = []
c = client(calls)
c.create_dataset("W1", "Ambient CIMS 40-600", "campaign subset")
check("create_dataset path",
      calls[-1]["url"].endswith("/api/workspaces/W1/datasets") and
      calls[-1]["method"] == "POST", calls[-1])
check("create_dataset body",
      calls[-1]["body"] == {"dataset_name": "Ambient CIMS 40-600",
                            "dataset_description": "campaign subset"})

c.update_dataset("W1", "D9", name="renamed")
check("update_dataset PATCH path+partial-body",
      calls[-1]["method"] == "PATCH" and
      calls[-1]["url"].endswith("/api/workspaces/W1/datasets/D9") and
      calls[-1]["body"] == {"dataset_id": "D9", "dataset_name": "renamed"}, calls[-1])

c.move_dataset("D9", "W1", "W2")
check("move_dataset path+body",
      calls[-1]["method"] == "POST" and
      calls[-1]["url"].endswith("/api/workspaces/W1/datasets/D9/move") and
      calls[-1]["body"] == {"source_workspace_id": "W1", "target_workspace_id": "W2"},
      calls[-1])

# ---------- batch create (field names + polarity guard) ----------
calls = []
c = client(calls)
c.create_batch("D1", "Blanks", "+", description="zero air")
check("create_batch uses sample_batch_polarity + target_collection_ids",
      calls[-1]["body"] == {"sample_batch_name": "Blanks",
                            "sample_batch_description": "zero air",
                            "sample_batch_polarity": "+", "dataset_id": "D1",
                            "target_collection_ids": []}, calls[-1]["body"])
try:
    c.create_batch("D1", "Bad", "positive")
    check("bad polarity rejected", False)
except CU.CurationError:
    check("bad polarity rejected", True)

c.copy_batch("B1", "D2", name="Blanks copy")
check("copy_batch path+body",
      calls[-1]["url"].endswith("/api/sample/batches/B1/copy") and
      calls[-1]["body"] == {"dataset_id": "D2", "sample_batch_name": "Blanks copy",
                            "sample_batch_description": ""}, calls[-1])

# ---------- sample copy / move / update / delete ----------
calls = []
c = client(calls)
c.copy_samples(["s1", "s2"], "B2")
check("copy_samples body",
      calls[-1]["url"].endswith("/api/sample/items/copy") and
      calls[-1]["body"] == {"sample_batch_id": "B2", "sample_item_ids": ["s1", "s2"]})
c.move_samples(["s3"], "B3")
check("move_samples body",
      calls[-1]["url"].endswith("/api/sample/items/move") and
      calls[-1]["body"] == {"sample_batch_id": "B3", "sample_item_ids": ["s3"]})
c.update_sample("s1", sample_item_name="ZA-01", sample_item_type="blank")
check("update_sample wraps in sample_item + PATCH",
      calls[-1]["method"] == "PATCH" and
      calls[-1]["url"].endswith("/api/sample/items/s1") and
      calls[-1]["body"] == {"sample_item": {"sample_item_name": "ZA-01",
                                            "sample_item_type": "blank"}}, calls[-1])

# ---------- rematch (params, not body) ----------
calls = []
c = client(calls)
c.rematch_batch("B1", full_remove=True, force=False)
check("rematch_batch uses query params",
      calls[-1]["url"].endswith("/api/match/rematch/batch/B1") and
      calls[-1]["params"] == {"full_remove": True, "force": False} and
      calls[-1]["body"] == {}, calls[-1])

# ---------- delete confirm-gate ----------
calls = []
c = client(calls)
try:
    c.delete_batch("B1")
    check("unconfirmed delete refused", False)
except CU.CurationError:
    check("unconfirmed delete refused", True)
check("refused delete sent NOTHING", len(calls) == 0, calls)
c.delete_batch("B1", confirm=True)
check("confirmed delete fires DELETE",
      len(calls) == 1 and calls[-1]["method"] == "DELETE" and
      calls[-1]["url"].endswith("/api/sample/batches/B1"))

# ---------- session-cookie auth (writes require it) ----------
calls = []
c = client(calls, cookie="mascope_session=abc123")
c.create_workspace("WithCookie")
check("write sends the session Cookie header",
      calls[-1]["headers"].get("Cookie") == "mascope_session=abc123", calls[-1]["headers"])

# a write WITHOUT a cookie fails fast (clear message), sends nothing
calls = []
c = client(calls, cookie=None)
try:
    c.create_workspace("NoCookie")
    check("write without cookie refused", False)
except CU.CurationError as e:
    check("write without cookie refused", "session cookie" in str(e).lower(), str(e))
check("no-cookie write sent nothing", len(calls) == 0)
# reads never need a cookie (they don't go through the mutating guard)
check("reads don't require a cookie",
      CU.CurationClient(url="https://x.test", token="t").cookie is None)

# ---------- dry-run plans everything, sends nothing ----------
calls = []
c = client(calls, dry_run=True, cookie=None)   # dry-run needs no cookie
c.create_workspace("Preview WS")
c.create_batch("D1", "PreviewBatch", "-")
c.delete_batch("B9")           # dry-run: no confirm needed, still not sent
planned = c.plan()
check("dry-run sent zero requests", len(calls) == 0, calls)
check("dry-run recorded 3 planned actions",
      len(planned) == 3 and all(not a.executed for a in planned), planned)
check("dry-run delete needs no confirm", planned[-1].op == "delete_batch")
check("summary renders PLAN lines", "[PLAN] create_workspace" in c.summary())

# ---------- name -> id resolution ----------
import pandas as pd  # noqa: E402
ws = pd.DataFrame([{"workspace_id": "W1", "workspace_name": "Alpha workspace"},
                   {"workspace_id": "W2", "workspace_name": "Beta Workspace"}])
check("resolve exact id",
      CU.CurationClient._resolve(ws, "W2", id_col="workspace_id",
                                 name_col="workspace_name", kind="workspace") == "W2")
check("resolve exact name (case-insensitive)",
      CU.CurationClient._resolve(ws, "beta workspace", id_col="workspace_id",
                                 name_col="workspace_name", kind="workspace") == "W2")
check("resolve substring",
      CU.CurationClient._resolve(ws, "Alpha", id_col="workspace_id",
                                 name_col="workspace_name", kind="workspace") == "W1")
try:
    CU.CurationClient._resolve(ws, "workspace", id_col="workspace_id",
                               name_col="workspace_name", kind="workspace")
    check("ambiguous match raises", False)
except CU.CurationError:
    check("ambiguous match raises", True)
try:
    CU.CurationClient._resolve(ws, "nope", id_col="workspace_id",
                               name_col="workspace_name", kind="workspace")
    check("no match raises", False)
except CU.CurationError:
    check("no match raises", True)

# ---------- server error surfaces the message ----------
c = CU.CurationClient(url="https://x.test", token="t", cookie="session=test")


def _err_request(*a, **k):
    r = FakeResp(status=422)
    r.content = b'{"detail":{"error_message":"bad polarity"}}'
    r.json = lambda: {"detail": {"error_message": "bad polarity"}}
    r.request.method = "POST"
    r.url = "https://x.test/api/sample/batches"
    return r


CU.requests.request = _err_request  # type: ignore[attr-defined]
try:
    c.create_batch("D1", "X", "+")
    check("HTTP error raised", False)
except CU.CurationError as e:
    check("HTTP error carries server message", "bad polarity" in str(e), str(e))


# ---------- token writes (Mascope >= 1.13 opened these routes to API tokens) ----------
calls = []
c = client(calls, cookie=None)
c.copy_samples(["s1"], "B2")
c.copy_batch("B1", "D2", name="Campaign")
c.update_sample("s1", sample_item_name="x")
check("token-writable routes send without a cookie",
      len(calls) == 3 and all("Cookie" not in k["headers"] for k in calls), calls)
check("token write carries bearer + service name",
      calls[0]["headers"].get("Authorization") == "Bearer t"
      and calls[0]["headers"].get("X-Service-Name") == "mascope_sdk", calls[0]["headers"])
check("token_writable: dataset create yes, batch create no, sample delete no",
      CU.token_writable("POST", "workspaces/W1/datasets")
      and not CU.token_writable("POST", "sample/batches")
      and not CU.token_writable("POST", "sample/items/delete")
      and not CU.token_writable("POST", "sample/batches/B1/import"))
calls = []
c = client(calls, cookie=None)
try:
    c.create_batch("D1", "NoCookieBatch", "-")
    check("cookie-only write refused without cookie", False)
except CU.CurationError as e:
    check("cookie-only write refused without cookie", "session cookie" in str(e).lower(), str(e))
check("refused cookie-only write sent nothing", len(calls) == 0)

# an older server refuses a token write with 401 -> the error names the cause
c = CU.CurationClient(url="https://x.test", token="t")


def _r401(*a, **k):
    r = FakeResp(status=401)
    r.request.method = "POST"
    return r


CU.requests.request = _r401  # type: ignore[attr-defined]
try:
    c.move_samples(["s1"], "B9")
    check("401 on a token write explained", False)
except CU.CurationError as e:
    check("401 on a token write explained", "1.13" in str(e) and "cookie" in str(e), str(e))

# the service name comes from $MASCOPE_SERVICE_NAME (an AI Agent token is 'ai-agent')
import os  # noqa: E402
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    envp = Path(td) / ".env"
    envp.write_text("MASCOPE_URL=https://x.test\nMASCOPE_ACCESS_TOKEN=t\nMASCOPE_SERVICE_NAME=ai-agent\n")
    saved = {k: os.environ.pop(k, None) for k in ("MASCOPE_URL", "MASCOPE_ACCESS_TOKEN", "MASCOPE_SERVICE_NAME")}
    try:
        cc = CU.CurationClient.from_env(str(envp))
        check("from_env takes MASCOPE_SERVICE_NAME", cc.service_name == "ai-agent", cc.service_name)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

# ---------- batch rename keeps the batch's target collections ----------
calls = []
c = client(calls, data=[{"target_collection_id": "TC1"}, {"target_collection_id": "TC2"}])
c.update_batch("B5", name="renamed")
check("update_batch reads the batch's collections first",
      calls[0]["method"] == "GET"
      and calls[0]["url"].endswith("/api/target/associations/target_collections_in_sample_batch")
      and calls[0]["params"] == {"sample_batch_id": "B5"}, calls[0])
check("update_batch sends them back with the new name",
      calls[-1]["method"] == "PATCH" and calls[-1]["url"].endswith("/api/sample/batches/B5")
      and calls[-1]["body"] == {"target_collection_ids": ["TC1", "TC2"], "sample_batch_name": "renamed"},
      calls[-1]["body"])
calls = []
c = client(calls)
c.update_batch("B6", description="d", target_collection_ids=["TC9"])
check("update_batch with explicit collections sends no read",
      len(calls) == 1 and calls[0]["body"] == {"target_collection_ids": ["TC9"], "sample_batch_description": "d"},
      calls)


assert FAIL == 0, f"{FAIL} checks failed"

if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
