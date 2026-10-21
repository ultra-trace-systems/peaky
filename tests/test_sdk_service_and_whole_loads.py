"""mascope-sdk 2026.10.20 / Mascope 1.13: the service name a token is presented
under, and batch loads the SDK marks incomplete.

Offline: fake clients stand in for the SDK. The SDK reads a batch's samples
concurrently and, when some reads fail, returns the rest with
``attrs['complete']`` False and the failures in ``attrs['failures']``; peaky
re-reads the transiently failed samples and refuses a load that stays partial.
"""
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from peaky.io import io_mascope as IO


# ---------------------------------------------------------------- service name
def test_service_name_defaults_to_the_notebook_token(monkeypatch):
    monkeypatch.delenv("MASCOPE_SERVICE_NAME", raising=False)
    assert IO.service_name() == "mascope_sdk"


def test_service_name_follows_the_env(monkeypatch):
    monkeypatch.setenv("MASCOPE_SERVICE_NAME", " ai-agent ")
    assert IO.service_name() == "ai-agent"


def test_transport_kwargs_carry_the_clients_settings():
    c = SimpleNamespace(_verify_ssl=False, _service_name="ai-agent")
    assert IO.transport_kwargs(c) == {"verify_ssl": False, "service_name": "ai-agent"}
    assert IO.transport_kwargs(SimpleNamespace()) == {}


def test_connect_presents_the_service_name(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("MASCOPE_URL=https://example.test\nMASCOPE_ACCESS_TOKEN=t\n"
                   "MASCOPE_SERVICE_NAME=ai-agent\n")
    for k in ("MASCOPE_URL", "MASCOPE_ACCESS_TOKEN", "MASCOPE_SERVICE_NAME", "MASCOPE_WORKSPACE"):
        monkeypatch.delenv(k, raising=False)
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

    import mascope_sdk
    monkeypatch.setattr(mascope_sdk, "MascopeClient", FakeClient)
    IO.connect(str(env), workspace="WS")
    assert seen["service_name"] == "ai-agent" and seen["workspace"] == "WS"


def test_publish_posts_under_the_clients_service_name(monkeypatch):
    from peaky.io import publish as P
    seen = {}

    class Resp:
        def json(self):
            return {"data": [{"state": "running"}]}

    def fake_post(url, path, token, body, timeout=None, **kw):
        seen.update(kw, path=path)
        return Resp()

    import mascope_sdk._http as H
    monkeypatch.setattr(H, "http_post", fake_post)
    c = SimpleNamespace(url="https://example.test", access_token="t", _service_name="ai-agent")
    P._post(c, "SIDxxxxxxxxxxxxx", {"rows": []}, timeout=5)
    assert seen["service_name"] == "ai-agent"
    assert seen["path"].endswith("/runs/import")


# ---------------------------------------------------------------- whole loads
def _frame(ids, *, failures=(), complete=None):
    f = pd.DataFrame({"sample_item_id": list(ids), "sample_item_name": [f"n-{i}" for i in ids],
                      "mz": [100.0] * len(ids), "height": [1.0] * len(ids)})
    f.attrs = {"complete": (not failures) if complete is None else complete,
               "failures": list(failures), "segments": {i: None for i in ids}}
    return f


def _fail(sid, error):
    return {"sample_batch_name": "B", "sample_item_id": sid, "sample_item_name": f"n-{sid}",
            "error_type": "AuthenticationError", "error": error}


class LoadSeq:
    """A client whose load_peaks answers from a queue and records each call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def load_peaks(self, **kw):
        self.calls.append(kw)
        return self.answers.pop(0)


WAF = "[HTTP 403] <html>Attention Required! | Cloudflare</html>"


def test_a_transient_failure_is_re_read_for_that_sample_only():
    c = LoadSeq(_frame(["s1", "s3"], failures=[_fail("s2", WAF)]), _frame(["s2"]))
    out = IO._load_peaks_whole(c, dataset="DS", batches="Batch (a)", what="t", base_delay=0)
    assert sorted(out.sample_item_id) == ["s1", "s2", "s3"]
    assert out.attrs["complete"] is True and out.attrs["failures"] == []
    first, second = c.calls
    assert first["batches"] == "Batch (a)" and first["exact"] is True and first["confirm_above"] is None
    assert isinstance(second["batches"], re.Pattern) and second["batches"].fullmatch("batch (a)")
    assert isinstance(second["samples"], re.Pattern) and second["samples"].fullmatch("n-s2")
    assert "exact" not in second                         # the SDK refuses exact with a compiled pattern


def test_a_non_transient_failure_raises_at_once():
    c = LoadSeq(_frame(["s1"], failures=[_fail("s2", "[HTTP 404] not found")]))
    with pytest.raises(IO.IncompleteLoad, match="n-s2"):
        IO._load_peaks_whole(c, dataset="DS", batches="B", what="t", base_delay=0)
    assert len(c.calls) == 1


def test_a_failure_that_persists_raises_after_the_re_reads():
    stuck = [_frame(["s1"], failures=[_fail("s2", WAF)])] + [
        _frame([], failures=[_fail("s2", WAF)]) for _ in range(3)]
    c = LoadSeq(*stuck)
    with pytest.raises(IO.IncompleteLoad, match="1 sample"):
        IO._load_peaks_whole(c, dataset="DS", batches="B", what="t", tries=3, base_delay=0)
    assert len(c.calls) == 4


def test_a_whole_load_passes_through():
    c = LoadSeq(_frame(["s1", "s2"]))
    out = IO._load_peaks_whole(c, dataset="DS", batches=re.compile("x"), what="t")
    assert len(out) == 2 and len(c.calls) == 1 and "exact" not in c.calls[0]


def test_an_old_sdk_frame_without_attrs_counts_as_whole():
    f = pd.DataFrame({"sample_item_id": ["s1"], "mz": [1.0]})
    c = LoadSeq(f)
    assert IO._load_peaks_whole(c, dataset="DS", batches="B", what="t") is f


def test_the_waf_retry_leaves_an_incomplete_load_alone():
    n = []

    def boom():
        n.append(1)
        raise IO.IncompleteLoad("batch: 1 sample(s) could not be read -- x: [HTTP 403] page")

    with pytest.raises(IO.IncompleteLoad):
        IO._with_waf_retry(boom, base_delay=0)
    assert len(n) == 1


def test_load_workers_env(monkeypatch):
    monkeypatch.delenv("PEAKY_LOAD_WORKERS", raising=False)
    assert IO._load_workers() == {}
    monkeypatch.setenv("PEAKY_LOAD_WORKERS", "3")
    assert IO._load_workers() == {"max_workers": 3}
    monkeypatch.setenv("PEAKY_LOAD_WORKERS", "lots")
    assert IO._load_workers() == {}
