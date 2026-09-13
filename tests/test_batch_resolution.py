"""Batch resolution: ONE rule for both server calls a batch run makes.

The SDK matches a plain string as a case-insensitive literal SUBSTRING, but
applies it differently on the two calls: load_peaks (the time series) keeps
every batch the string occurs in, samples.list (the roster) demands a unique
match. On a dataset where one batch's name is a prefix of its siblings'
("Site A Ur 122-600" beside "Site A Ur 122-600 11 " and "Site A Ur 122-600 05-30 -
06-02 wind zone 1"), `peaky batch --batch "Site A Ur 122-600"` used to POOL all
three batches' peaks into one time series and then die at the roster with
"Multiple batchs matching"; addressed by id it resolved the roster and found no
time series at all.

io_mascope.resolve_batch settles the batch ONCE (exact id > exact casefolded
name > unique literal substring; ambiguity raises) and both fetchers go through
it. Sections 1-4 pin that offline, over a fake client shaped like the SDK
(whose substring rules are reproduced so the calls the OLD code made still
misbehave here -- the controls). Section 5 runs the INSTALLED SDK's own matching
code, so a contract change fails here instead of on a live run. Section 6 pins
that the pipeline carries the resolved display name + id into the run folder,
the assign stage and the manifest.

Run: python3 tests/test_batch_resolution.py   (or pytest)
"""
import inspect
import os
import re
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

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


def raises(fn, exc=Exception):
    """The `exc` instance `fn()` raises, or None if it returns."""
    try:
        fn()
    except exc as e:
        return e
    return None


# The bates.mascope.app "Emmy's Workspace" shape (2026-09-13): the batch under
# study is a PREFIX of two siblings' names, one of which even ends in a space.
DATASET = "Emmy's Workspace"
DATASET_ID = "DSxxxxxxxxxxxxxx"
BATCH_ID = "BkKhjKMnAFVXFCUm"
BATCH = "Site A Ur 122-600"
SIB_11 = "Site A Ur 122-600 11 "
SIB_WZ = "Site A Ur 122-600 05-30 - 06-02 wind zone 1"
LISTING = pd.DataFrame({
    "sample_batch_id": [BATCH_ID, "SIB11xxxxxxxxxxx", "SIBWZxxxxxxxxxxx", "NITRATExxxxxxxxx"],
    "sample_batch_name": [BATCH, SIB_11, SIB_WZ, "Nitrate plain batch"],
    "dataset_id": [DATASET_ID] * 4,
})


class FakeClient:
    """The SDK surfaces a batch run touches, recording every call.

    Shaped like MascopeClient: `batches.list(dataset)` -> the listing,
    `datasets.list()` -> the datasets (when given), `samples.list(batch=...)` ->
    the roster under the SDK's exact-id-else-UNIQUE-substring rule, and
    `load_peaks(...)` -> the time series under the SDK's substring-or-exact name
    mask. Both rules are reproduced from the SDK so that what tripped the old
    code trips this fake too."""

    def __init__(self, listing=LISTING, datasets=None):
        self.listing = listing
        self.datasets_df = datasets
        self.calls = []                       # (surface, kwargs) in call order
        c = self

        class _Batches:
            def list(self, dataset=None):     # noqa: A001
                c.calls.append(("batches.list", {"dataset": dataset}))
                if c.datasets_df is not None:  # per-dataset listings
                    return c.listing[c.listing["dataset_id"] == dataset].reset_index(drop=True)
                return c.listing

        class _Datasets:
            def list(self):                   # noqa: A001
                c.calls.append(("datasets.list", {}))
                return c.datasets_df

        class _Samples:
            def list(self, *, batch=None, batches=None, dataset=None,   # noqa: A001
                     drop_columns=None):
                c.calls.append(("samples.list", {"batch": batch, "dataset": dataset}))
                bs = c.listing
                if batch in bs["sample_batch_id"].values:                 # exact id
                    hit = bs[bs["sample_batch_id"] == batch]
                else:                                                     # unique substring
                    hit = bs[bs["sample_batch_name"].str.contains(re.escape(batch), case=False)]
                    if len(hit) > 1:
                        raise ValueError(f"Multiple batchs matching '{batch}'")
                if not len(hit):
                    raise ValueError(f"No batch matching '{batch}'")
                bid = hit.iloc[0]["sample_batch_id"]
                return pd.DataFrame({"sample_item_id": [f"{bid}-s1", f"{bid}-s2"],
                                     "sample_item_name": ["s1", "s2"]})

        self.batches, self.datasets, self.samples = _Batches(), _Datasets(), _Samples()

    def load_peaks(self, *, dataset, batches, exact=False, confirm_above=100, **kw):
        self.calls.append(("load_peaks", {"dataset": dataset, "batches": batches,
                                          "exact": exact, "confirm_above": confirm_above}))
        names = self.listing["sample_batch_name"]
        if isinstance(batches, re.Pattern):
            mask = names.str.contains(batches)
        elif exact:
            mask = names.str.casefold() == batches.casefold()
        else:
            mask = names.str.contains(re.escape(batches), case=False)
        sel = self.listing[mask]
        if not len(sel):
            return None
        return pd.DataFrame({"sample_batch_name": sel["sample_batch_name"].tolist(),
                             "sample_item_id": [f"{i}-s1" for i in sel["sample_batch_id"]],
                             "mz": [100.0] * len(sel), "height": [1.0] * len(sel)})

    def only(self, surface):
        return [kw for s, kw in self.calls if s == surface]


# ---------------------------------------------------------------------------
# 1. resolve_batch: exact id > exact name > unique substring; ambiguity raises
# ---------------------------------------------------------------------------
print("-- resolve_batch --")
c = FakeClient()
rb = IO.resolve_batch(c, BATCH, dataset=DATASET)
check("exact name that is a prefix of two siblings -> that ONE batch, by name",
      rb == (BATCH_ID, BATCH, "name", 1), rb)
rb = IO.resolve_batch(c, BATCH_ID, dataset=DATASET)
check("the batch id -> the same batch, carrying its DISPLAY name",
      rb.id == BATCH_ID and rb.name == BATCH and rb.how == "id", rb)
rb = IO.resolve_batch(c, BATCH.upper(), dataset=DATASET)
check("the exact-name step is case-insensitive",
      rb.id == BATCH_ID and rb.how == "name", rb)
rb = IO.resolve_batch(c, "wind zone", dataset=DATASET)
check("a UNIQUE substring still resolves (the SDK's own rule, as the last resort)",
      rb.id == "SIBWZxxxxxxxxxxx" and rb.name == SIB_WZ and rb.how == "substring", rb)
rb = IO.resolve_batch(c, SIB_11.strip(), dataset=DATASET)
check("the trailing-space sibling is reachable WITHOUT its trailing space (unique substring)",
      rb.name == SIB_11 and rb.how == "substring", rb)
e = raises(lambda: IO.resolve_batch(c, "Site A Ur", dataset=DATASET), ValueError)
check("an ambiguous substring RAISES and names every candidate with its id",
      e is not None and "Multiple" in str(e)
      and all(n in str(e) for n in (BATCH, SIB_11, SIB_WZ, BATCH_ID)), e)
e = raises(lambda: IO.resolve_batch(c, "Uronium", dataset=DATASET), ValueError)
check("no match raises, listing what exists",
      e is not None and "No batch matching" in str(e) and BATCH in str(e), e)
e = raises(lambda: IO.resolve_batch(c, re.compile("Site A"), dataset=DATASET), TypeError)
check("a compiled pattern is refused here (that is the pool path)",
      e is not None and "pool" in str(e), e)
e = raises(lambda: IO.resolve_batch(FakeClient(listing=LISTING.iloc[0:0]), BATCH,
                                    dataset=DATASET), RuntimeError)
check("an empty listing raises (no fallback)", e is not None and "no batches" in str(e), e)
check("the listing is asked for by the caller's dataset string (the SDK settles it)",
      c.only("batches.list")[0] == {"dataset": DATASET}, c.calls[:1])

# no dataset: every dataset's batches are searched, as the SDK's samples.list does
DS = pd.DataFrame({"dataset_id": [DATASET_ID, "DS2xxxxxxxxxxxxx"],
                   "dataset_name": [DATASET, "Other"]})
L2 = pd.concat([LISTING, pd.DataFrame({"sample_batch_id": ["OTHERxxxxxxxxxxx"],
                                       "sample_batch_name": ["Other batch"],
                                       "dataset_id": ["DS2xxxxxxxxxxxxx"]})],
               ignore_index=True)
c = FakeClient(listing=L2, datasets=DS)
rb = IO.resolve_batch(c, "Other batch")
check("no dataset given: resolves across every dataset's listing",
      rb.id == "OTHERxxxxxxxxxxx"
      and [s for s, _ in c.calls] == ["datasets.list", "batches.list", "batches.list"],
      (rb, c.calls))
e = raises(lambda: IO.resolve_batch(FakeClient(listing=L2, datasets=DS), "Site A Ur"),
           ValueError)
check("no dataset given: an ambiguous name is still refused",
      e is not None and "Multiple" in str(e) and "any dataset" in str(e), e)

# ---------------------------------------------------------------------------
# 2. fetch_batch_peaks: the time series is asked for by the EXACT resolved name
# ---------------------------------------------------------------------------
print("-- fetch_batch_peaks --")
c = FakeClient()
ts = IO.fetch_batch_peaks(c, DATASET, BATCH)
lp = c.only("load_peaks")
check("TS by exact name: ONE load_peaks call -- the resolved name, exact=True, no prompt",
      lp == [{"dataset": DATASET, "batches": BATCH, "exact": True, "confirm_above": None}],
      lp)
check("TS by exact name: exactly ONE batch's peaks come back (the siblings are not pooled)",
      set(ts["sample_batch_name"]) == {BATCH}, set(ts["sample_batch_name"]))
c = FakeClient()
ts = IO.fetch_batch_peaks(c, DATASET, BATCH_ID)
lp = c.only("load_peaks")
check("TS by id: the loader (which has no id filter) is asked for the resolved name, exactly",
      lp == [{"dataset": DATASET, "batches": BATCH, "exact": True, "confirm_above": None}]
      and set(ts["sample_batch_name"]) == {BATCH}, lp)
c = FakeClient()
e = raises(lambda: IO.fetch_batch_peaks(c, DATASET, "Site A Ur"), ValueError)
check("TS by an ambiguous substring: raises BEFORE the loader is called -- never pools",
      e is not None and "Multiple" in str(e) and not c.only("load_peaks"), (e, c.calls))
c = FakeClient()
pooled = c.load_peaks(dataset=DATASET, batches=BATCH, confirm_above=None)
check("(control) the bare substring call the OLD code made pools all three siblings",
      set(pooled["sample_batch_name"]) == {BATCH, SIB_11, SIB_WZ},
      set(pooled["sample_batch_name"]))
check("(control) ... and the bare call by id finds nothing (the id is not a name)",
      c.load_peaks(dataset=DATASET, batches=BATCH_ID, confirm_above=None) is None)


class _NoPeaks(FakeClient):
    def load_peaks(self, **kw):
        return None


e = raises(lambda: IO.fetch_batch_peaks(_NoPeaks(), DATASET, BATCH_ID), RuntimeError)
check("no peaks -> RuntimeError that names the batch by its display name",
      e is not None and "no peaks" in str(e) and BATCH in str(e), e)

# ---------------------------------------------------------------------------
# 3. fetch_batch_samples: the roster is asked for by ID
# ---------------------------------------------------------------------------
print("-- fetch_batch_samples --")
c = FakeClient()
sl = IO.fetch_batch_samples(c, BATCH, dataset=DATASET)
check("roster by exact name: the SDK is asked by ID (its unique-substring rule refuses the prefix)",
      c.only("samples.list") == [{"batch": BATCH_ID, "dataset": DATASET}] and len(sl) == 2,
      c.calls)
c = FakeClient()
sl = IO.fetch_batch_samples(c, BATCH_ID, dataset=DATASET)
check("roster by id: asked by id, two samples back",
      c.only("samples.list") == [{"batch": BATCH_ID, "dataset": DATASET}] and len(sl) == 2,
      c.calls)
c = FakeClient()
e = raises(lambda: IO.fetch_batch_samples(c, "Site A Ur", dataset=DATASET), ValueError)
check("roster by an ambiguous substring: raises before samples.list is called",
      e is not None and "Multiple" in str(e) and not c.only("samples.list"), (e, c.calls))
e = raises(lambda: FakeClient().samples.list(batch=BATCH, dataset=DATASET), ValueError)
check("(control) the bare samples.list call the OLD code made refuses the exact prefix name",
      e is not None and "Multiple" in str(e), e)

# ---------------------------------------------------------------------------
# 4. duplicate names: an id still resolves; the NAME-addressed loader is refused
# ---------------------------------------------------------------------------
print("-- duplicate names --")
DUP = pd.DataFrame({"sample_batch_id": ["D1xxxxxxxxxxxxxx", "D2xxxxxxxxxxxxxx"],
                    "sample_batch_name": ["Chamber run", "chamber run"],
                    "dataset_id": [DATASET_ID] * 2})
c = FakeClient(listing=DUP)
rb = IO.resolve_batch(c, "D2xxxxxxxxxxxxxx", dataset=DATASET)
check("duplicate names: the id resolves, and the clash is reported",
      rb.name == "chamber run" and rb.how == "id" and rb.n_same_name == 2, rb)
e = raises(lambda: IO.resolve_batch(c, "Chamber run", dataset=DATASET), ValueError)
check("duplicate names: the name is ambiguous -> raises, pointing at the id",
      e is not None and "named" in str(e) and "batch id" in str(e), e)
c = FakeClient(listing=DUP)
IO.fetch_batch_samples(c, "D2xxxxxxxxxxxxxx", dataset=DATASET)
check("duplicate names: the roster (asked by id) is fine",
      c.only("samples.list") == [{"batch": "D2xxxxxxxxxxxxxx", "dataset": DATASET}], c.calls)
c = FakeClient(listing=DUP)
e = raises(lambda: IO.fetch_batch_peaks(c, DATASET, "D2xxxxxxxxxxxxxx"), RuntimeError)
check("duplicate names: the TS (a NAME-addressed loader) is refused rather than pooled",
      e is not None and "cannot be isolated" in str(e) and not c.only("load_peaks"),
      (e, c.calls))

# ---------------------------------------------------------------------------
# 5. REAL-SDK tripwire (offline): the installed SDK's own matching code, over the
#    same names. Pins WHY peaky resolves itself and WHAT it then hands the SDK.
# ---------------------------------------------------------------------------
print("-- real SDK --")
import mascope_sdk._loaders as _sdk_loaders                       # noqa: E402
from mascope_sdk import MascopeClient                             # noqa: E402
from mascope_sdk._resolve import resolve_id as _sdk_resolve_id    # noqa: E402
_sdk_name_mask = getattr(_sdk_loaders, "name_mask", None) or _sdk_loaders._name_mask
_names = LISTING["sample_batch_name"]
with warnings.catch_warnings():
    warnings.simplefilter("error")     # a DeprecationWarning here = contract drift
    _m_sub = _sdk_name_mask(_names, BATCH, exact=False)
    _m_ex = _sdk_name_mask(_names, BATCH, exact=True)
check("REAL SDK: the substring mask keeps all three siblings (why a bare name must not be sent)",
      list(_m_sub) == [True, True, True, False], list(_m_sub))
check("REAL SDK: the exact mask keeps exactly the one batch (what fetch_batch_peaks sends)",
      list(_m_ex) == [True, False, False, False], list(_m_ex))
_kw = dict(id_column="sample_batch_id", name_column="sample_batch_name", entity_label="batch")
e = raises(lambda: _sdk_resolve_id(BATCH, LISTING, **_kw), ValueError)
check("REAL SDK: resolve_id refuses the exact prefix name (why the roster is asked by id)",
      e is not None and "Multiple" in str(e), e)
check("REAL SDK: resolve_id takes the id straight",
      _sdk_resolve_id(BATCH_ID, LISTING, **_kw) == BATCH_ID)
check("REAL SDK: load_peaks still takes exact= (peaky relies on it)",
      "exact" in inspect.signature(MascopeClient.load_peaks).parameters
      and "exact" in inspect.signature(_sdk_loaders.load_peaks).parameters)

# ---------------------------------------------------------------------------
# 6. the pipeline carries the resolved DISPLAY name + id: run folder, cover
#    context, assign stage, manifest. Assign / report / provenance are stubbed;
#    the resolution itself is real, over the fake client.
# ---------------------------------------------------------------------------
print("-- pipeline.run_batch --")
from peaky import pipeline as PL  # noqa: E402
from peaky import progress as PG  # noqa: E402
from peaky.batch import assign_batch as _AB  # noqa: E402
from peaky.reporting import provenance as _PV  # noqa: E402

WHEN = datetime(2026, 9, 13, 10, 0, 0)
_TS = pd.DataFrame({"sample_item_id": ["s1", "s1", "s2", "s2"],
                    "mz": [100.0, 200.0, 100.0, 300.0], "height": [5.0] * 4,
                    "sample_batch_name": [BATCH] * 4})
_saved = {"connect": IO.connect, "ab": _AB.run, "gen": PL.generate_report,
          "rec": _PV.record_run}
_got: dict = {}
_client = FakeClient()
_connects: list = []                  # one entry per IO.connect() call


def _fake_ab(**kw):
    _got["ab"] = kw
    return {"summary": {}, "sample_ids": []}


def _fake_connect(*a, **k):
    _connects.append(1)
    return _client


IO.connect = _fake_connect
_AB.run = _fake_ab
PL.generate_report = lambda ctx, ts, **kw: {}
_PV.record_run = lambda **kw: _got.__setitem__("rec", kw)
try:
    _logs: list = []
    with tempfile.TemporaryDirectory() as d:
        res = PL.run_batch(batch=BATCH_ID, dataset=DATASET, reagent="Br", base_out=d,
                           ts=_TS, when=WHEN, do_report=False, log=_logs.append)
        ctx = res["ctx"]
        check("run_batch by id: the run folder is named after the batch's DISPLAY name, not the id",
              ctx.batch_name == BATCH
              and os.path.basename(ctx.out_dir).startswith(PL.slugify(BATCH) + "_")
              and BATCH_ID not in ctx.out_dir, ctx.out_dir)
        check("run_batch by id: the context carries the id too", ctx.batch_id == BATCH_ID, ctx)
        check("run_batch by id: the assign stage is addressed by id",
              _got["ab"]["batch"] == BATCH_ID and _got["ab"]["dataset"] == DATASET,
              _got["ab"].get("batch"))
        check("run_batch by id: provenance records the display name AND the id",
              _got["rec"]["batch_name"] == BATCH and _got["rec"]["batch_id"] == BATCH_ID,
              {k: _got["rec"].get(k) for k in ("batch_name", "batch_id")})
        check("run_batch given a frame: no time series is fetched, the batch is only resolved",
              not _client.only("load_peaks") and len(_client.only("batches.list")) == 1,
              _client.calls)
        _line = [ln for ln in _logs if ln.startswith("[batch] batch ")]
        check("run_batch by id: logs the resolution once, name and id",
              len(_line) == 1 and BATCH in _line[0] and BATCH_ID in _line[0], _logs)
        check("... on a line the progress window cannot mistake for the run-dir marker",
              _line and PG.RE_RUNDIR.match(_line[0]) is None, _line)
    _client.calls.clear(); _got.clear(); _logs.clear()
    with tempfile.TemporaryDirectory() as d:
        res = PL.run_batch(batch=BATCH, dataset=DATASET, reagent="Br", base_out=d,
                           ts=_TS, when=WHEN, do_report=False, log=_logs.append)
        check("run_batch by exact name: the one batch; the assign stage still gets the id",
              res["ctx"].batch_name == BATCH and res["ctx"].batch_id == BATCH_ID
              and _got["ab"]["batch"] == BATCH_ID and _got["rec"]["batch_id"] == BATCH_ID,
              (res["ctx"].batch_name, res["ctx"].batch_id, _got["ab"].get("batch")))
        check("run_batch by exact name: nothing to announce (the name IS the display name)",
              not [ln for ln in _logs if ln.startswith("[batch] batch ")], _logs)
    _client.calls.clear(); _got.clear()
    with tempfile.TemporaryDirectory() as d:
        e = raises(lambda: PL.run_batch(batch="Site A Ur", dataset=DATASET, reagent="Br",
                                        base_out=d, ts=_TS, when=WHEN, do_report=False,
                                        log=lambda *a: None), ValueError)
        check("run_batch by an ambiguous substring: refused before any run folder or assign",
              e is not None and "Multiple" in str(e) and os.listdir(d) == []
              and "ab" not in _got, (e, os.listdir(d)))
    _client.calls.clear(); _got.clear(); _connects.clear()
    with tempfile.TemporaryDirectory() as d:
        res = PL.run_batch(batch=BATCH, dataset=DATASET, reagent="Br", base_out=d,
                           ts=None, when=WHEN, do_report=False, log=lambda *a: None)
        lp = _client.only("load_peaks")
        check("run_batch live fetch: the TS is asked for by the exact resolved name, once",
              lp == [{"dataset": DATASET, "batches": BATCH, "exact": True,
                      "confirm_above": None}], lp)
        # the fetch re-settles the id against the listing (the SDK caches it per
        # client, so that is free) -- on the SAME connection, opened once
        check("run_batch live fetch: ONE connection serves the resolution and the fetch",
              len(_connects) == 1 and len(_client.only("batches.list")) == 2,
              (len(_connects), _client.calls))
finally:
    IO.connect, _AB.run, PL.generate_report, _PV.record_run = (
        _saved["connect"], _saved["ab"], _saved["gen"], _saved["rec"])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
