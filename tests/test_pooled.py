"""Offline tests for the multi-batch POOLING path: one presence set-cover over
the pooled table with per-group coverage (sampling.select_cover_samples with
group_col), the pool-name helper, the un-escaped regex loader
(io_mascope.fetch_pooled_peaks), and `peaky pool` CLI parsing. No network.
Run: python3 tests/test_pooled.py"""
import re as _re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import pipeline as PL           # noqa: E402
from peaky.batch import sampling as SS     # noqa: E402
from peaky.io import io_mascope as IO      # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


def make_pool(groups, *, n_unique=3):
    """Build a pooled peak frame across groups. `groups` maps a batch name to a
    list of per-sample intensity scales; every sample holds 6 shared bins plus
    `n_unique` bins exclusive to its group (so each group has its own chemistry),
    and each sample appears TWICE (a twin) so every bin passes the prevalence gate."""
    t0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    rows, s = [], 0
    for gi, (gname, scales) in enumerate(groups.items()):
        for j, sc in enumerate(scales):
            for twin in ("", "t"):
                sid = f"{gname[:2]}_s{s:02d}{twin}"
                t = t0 + pd.Timedelta(minutes=30 * s)
                for k in range(6):
                    rows.append(dict(sample_item_id=sid, sample_batch_name=gname,
                                     datetime_utc=t, mz=100.0 + k, height=sc * (1 + 0.1 * k)))
                for k in range(n_unique):
                    rows.append(dict(sample_item_id=sid, sample_batch_name=gname,
                                     datetime_utc=t, mz=200.0 + 10 * gi + k + 0.1 * j,
                                     height=sc))
            s += 1
    return pd.DataFrame(rows)


# --- a LOUD group and a QUIET group -----------------------------------------
# Group A intensities dwarf group B. A brightness-driven selector would let A win
# every bin and under-represent B; a PRESENCE cover is blind to intensity, so B's
# own bins count exactly as much as A's.
LOUD, QUIET = "HR-CIMS 100-500 zone 1", "HR-CIMS 100-500 zone 2"
pool = make_pool({
    LOUD: list(np.linspace(5e5, 9e5, 4)),    # loud group
    QUIET: list(np.linspace(1e4, 3e4, 4)),   # quiet group
})

prov = SS.select_cover_samples(pool, group_col="sample_batch_name", k_min=2, min_gain=0.0)
meta = prov.attrs["selection"]
union = prov["sample_item_id"].tolist()
per_group = prov.groupby("sample_batch_name")["sample_item_id"].nunique().to_dict()

check("pool cover: both groups' bins fully covered (coverage_by_group == 1.0)",
      all(np.isclose(v, 1.0) for v in meta["coverage_by_group"].values()),
      meta["coverage_by_group"])
check("pool cover: the QUIET group gets picks (its bins are unique to it)",
      per_group.get(QUIET, 0) >= 1, per_group)
check("pool cover: the LOUD group gets picks too", per_group.get(LOUD, 0) >= 1, per_group)
check("pool cover: ids are de-duplicated", len(union) == len(set(union)), len(union))
check("pool cover: every id belongs to the pool", set(union) <= set(pool["sample_item_id"]))
check("pool cover: picks_by_group agrees with the provenance rows",
      meta["picks_by_group"] == per_group, (meta["picks_by_group"], per_group))
check("pool cover: k recorded == rows", meta["k"] == len(prov), meta)

# missing group column -> clear error
try:
    SS.select_cover_samples(pool, group_col="no_such_col")
    check("pool cover: missing group_col raises", False, "no error")
except KeyError:
    check("pool cover: missing group_col raises KeyError", True)

# --- pool_name: regex -> readable label -------------------------------------
check("pool_name: strips metachars + tags pooled",
      PL.pool_name("HR-CIMS 100-500.*zone") == "HR-CIMS 100-500 zone (pooled)",
      PL.pool_name("HR-CIMS 100-500.*zone"))
check("pool_name: collapses runs of metachars/space",
      PL.pool_name("A.*(B)+  C") == "A B C (pooled)", PL.pool_name("A.*(B)+  C"))
check("pool_name: strips quantifier braces too",
      PL.pool_name("a{2,3} zone") == "a 2,3 zone (pooled)", PL.pool_name("a{2,3} zone"))
check("pool_name: empty-ish -> fallback",
      PL.pool_name(".*") == "pooled-batches", PL.pool_name(".*"))

# --- staging helpers: a per-group child folder gets everything a report needs ---
import tempfile, os as _os  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    pool = _os.path.join(d, "pool"); child = _os.path.join(d, "child")
    _os.makedirs(_os.path.join(pool, "per_file")); _os.makedirs(child)
    for n in ("merged_ledger.csv", "batch_summary.json"):
        Path(pool, n).write_text("x")
    Path(pool, "per_file", "s1_ledger.csv").write_text("y")
    gprov = pd.DataFrame({"sample_item_id": ["s1"], "role": ["cover"], "pick": [1],
                          "bins_new": [3], "coverage": [0.5], "sample_batch_name": ["zone 1"]})
    PL._write_selected_samples(pool, gprov)
    check("_write_selected_samples: writes tables/selected_samples.csv",
          _os.path.exists(_os.path.join(pool, "tables", "selected_samples.csv")))
    PL._stage_pool_child(pool, child, gprov)
    check("_stage_pool_child: copies merged_ledger.csv",
          _os.path.exists(_os.path.join(child, "merged_ledger.csv")))
    check("_stage_pool_child: copies per_file/",
          _os.path.exists(_os.path.join(child, "per_file", "s1_ledger.csv")))
    check("_stage_pool_child: writes the group's selected_samples.csv",
          _os.path.exists(_os.path.join(child, "tables", "selected_samples.csv")))

# --- fetch_pooled_peaks compiles the regex UN-escaped (the whole point): the
# SDK treats a plain string as a literal substring, so only a compiled pattern
# keeps its regex meaning ----------
class _FakeClient:
    def __init__(self): self.seen = {}
    def load_peaks(self, *, dataset, batches, confirm_above):
        self.seen = dict(dataset=dataset, batches=batches, confirm_above=confirm_above)
        return pd.DataFrame({"sample_item_id": ["a"], "sample_batch_name": ["b"],
                             "mz": [100.0], "height": [1e4]})

fc = _FakeClient()
out = IO.fetch_pooled_peaks(fc, "DS", "HR-CIMS 100-500.*zone")
check("fetch_pooled_peaks: regex compiled un-escaped (keeps its regex meaning)",
      isinstance(fc.seen["batches"], _re.Pattern)
      and fc.seen["batches"].pattern == "HR-CIMS 100-500.*zone"
      and (fc.seen["batches"].flags & _re.IGNORECASE), fc.seen)
check("fetch_pooled_peaks: never prompts (confirm_above=None)",
      fc.seen["confirm_above"] is None, fc.seen)
check("fetch_pooled_peaks: returns the pooled frame", len(out) == 1, len(out))

class _NoBatchCol:
    def load_peaks(self, **k):
        return pd.DataFrame({"sample_item_id": ["a"], "mz": [1.0]})   # no group col
try:
    IO.fetch_pooled_peaks(_NoBatchCol(), "DS", "x")
    check("fetch_pooled_peaks: missing sample_batch_name raises", False)
except RuntimeError:
    check("fetch_pooled_peaks: missing sample_batch_name raises RuntimeError", True)

class _Empty:
    def load_peaks(self, **k): return pd.DataFrame()
try:
    IO.fetch_pooled_peaks(_Empty(), "DS", "nomatch")
    check("fetch_pooled_peaks: no match raises", False)
except RuntimeError:
    check("fetch_pooled_peaks: empty match raises RuntimeError", True)

# --- CLI: `peaky pool` parses to cmd_pool with the pooling defaults ----------
from peaky import cli  # noqa: E402
ns = cli.build_parser().parse_args(
    ["pool", "--batches", "HR-CIMS .*zone", "--dataset", "DS", "--reagent", "Ur"])
check("cli pool: dispatches to cmd_pool", ns.func is cli.cmd_pool, getattr(ns, "func", None))
check("cli pool: --batches captured", ns.batches == "HR-CIMS .*zone", ns.batches)
check("cli pool: k_max default == sampling.K_MAX (one cover over the pool)",
      ns.k_max == SS.K_MAX, ns.k_max)
check("cli pool: k_min / min_gain defaults", ns.k_min == SS.K_MIN and ns.min_gain == SS.MIN_GAIN)
check("cli pool: group_by default sample_batch_name",
      ns.group_by == "sample_batch_name", ns.group_by)
check("cli pool: --no-group-reports flag present", ns.no_group_reports is False, ns.no_group_reports)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
