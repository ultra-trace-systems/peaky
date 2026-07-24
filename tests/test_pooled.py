"""Offline tests for the multi-batch POOLING path: per-group brightest union
(sampling.select_pooled_union), the pool-name helper, the unescaped regex loader
(io_mascope.fetch_pooled_peaks), and `peaky pool` CLI parsing. No network.
Run: python3 tests/test_pooled.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import pipeline as PL           # noqa: E402
from peaky import sampling as SS           # noqa: E402
from peaky.io import io_mascope as IO      # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


def make_pool(groups):
    """Build a pooled peak frame across groups. `groups` maps a batch name to a
    list of per-sample TICs; each sample gets 6 peaks so it can be selective."""
    t0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    rows, s = [], 0
    for gname, tics in groups.items():
        for j, tic in enumerate(tics):
            sid = f"{gname[:2]}_s{s:02d}"; s += 1
            t = t0 + pd.Timedelta(minutes=30 * s)
            for k in range(6):
                rows.append(dict(sample_item_id=sid, sample_batch_name=gname,
                                 datetime_utc=t, mz=100.0 + k,
                                 height=tic / 6 * (1 + 0.1 * k)))
    return pd.DataFrame(rows)


# --- a LOUD group and a QUIET group -----------------------------------------
# Group A intensities dwarf group B: a naive pooled brightest pass would let A win
# most bins and under-represent B. The union must still cover BOTH.
LOUD, QUIET = "HR-CIMS 100-500 zone 1", "HR-CIMS 100-500 zone 2"
pool = make_pool({
    LOUD: list(np.linspace(5e5, 9e5, 8)),    # loud group
    QUIET: list(np.linspace(1e4, 3e4, 8)),   # quiet group
})

union, prov = SS.select_pooled_union(pool, k_max=4, height_floor=1000.0)
groups_in_prov = set(prov["sample_batch_name"])
per_group = prov.groupby("sample_batch_name")["sample_item_id"].nunique().to_dict()

check("union: both groups represented in provenance",
      groups_in_prov == set(pool["sample_batch_name"].unique()), groups_in_prov)
check("union: the QUIET group still gets picks (not starved)",
      per_group.get(QUIET, 0) >= 1, per_group)
check("union: the LOUD group gets picks too",
      per_group.get(LOUD, 0) >= 1, per_group)
check("union: ids are de-duplicated", len(union) == len(set(union)), len(union))
check("union: every id belongs to some group's samples",
      set(union) <= set(pool["sample_item_id"].unique()), None)
check("union: per group capped at k_max (+ <=2 time-grid endpoints)",
      all(v <= 4 + 2 for v in per_group.values()), per_group)
check("union: count == sum of distinct per-group picks (disjoint groups)",
      len(union) == sum(per_group.values()), (len(union), per_group))

# naive pooled brightest DOES starve the quiet group -> motivates the union
naive = set(SS.select_brightest_coverage_sample_ids(pool, k_max=4, height_floor=1000.0))
quiet_ids = set(pool.loc[pool.sample_batch_name == QUIET, "sample_item_id"])
check("motivation: naive pooled brightest under-covers the quiet group vs union",
      len(naive & quiet_ids) <= per_group.get(QUIET, 0),
      (len(naive & quiet_ids), per_group))

# missing group column -> clear error
try:
    SS.select_pooled_union(pool.drop(columns=["sample_batch_name"]))
    check("union: missing group_col raises", False, "no error")
except KeyError:
    check("union: missing group_col raises KeyError", True)

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
    gprov = pd.DataFrame({"sample_item_id": ["s1"], "role": ["coverage-winner"],
                          "bins_won": [3], "sample_batch_name": ["zone 1"]})
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

# --- fetch_pooled_peaks passes the regex UNescaped (the whole point) ----------
class _FakeClient:
    def __init__(self): self.seen = {}
    def load_peaks(self, *, dataset, batches, confirm_above):
        self.seen = dict(dataset=dataset, batches=batches, confirm_above=confirm_above)
        return pd.DataFrame({"sample_item_id": ["a"], "sample_batch_name": ["b"],
                             "mz": [100.0], "height": [1e4]})

fc = _FakeClient()
out = IO.fetch_pooled_peaks(fc, "DS", "HR-CIMS 100-500.*zone")
check("fetch_pooled_peaks: regex passed through UNescaped",
      fc.seen["batches"] == "HR-CIMS 100-500.*zone", fc.seen)
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
check("cli pool: k_max default 6 (per group)", ns.k_max == 6, ns.k_max)
check("cli pool: coverage-target default 0.90", ns.coverage_target == 0.90, ns.coverage_target)
check("cli pool: group_by default sample_batch_name",
      ns.group_by == "sample_batch_name", ns.group_by)
check("cli pool: --no-group-reports flag present", ns.no_group_reports is False, ns.no_group_reports)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
