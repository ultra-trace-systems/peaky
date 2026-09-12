"""Offline tests for batch/sampling.py -- greedy presence set-cover selection.
Run: python3 tests/test_sampling.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.batch import sampling as SS  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


T0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")


def make_batch(spec: dict, *, minutes: int = 10, height: float = 500.0):
    """spec: sample_id -> iterable of m/z values present in that sample (each at
    `height` cps). Samples are stamped `minutes` apart in the given order."""
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = T0 + pd.Timedelta(minutes=minutes * i)
        for mz in mzs:
            rows.append(dict(sample_item_id=sid, sample_item_name=str(t),
                             datetime_utc=t, mz=float(mz), height=height))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sample_table basics
# ---------------------------------------------------------------------------
bg = list(range(100, 120))                     # 20 shared background bins
pk = make_batch({f"s{i:02d}": bg for i in range(6)})
tab = SS.sample_table(pk)
check("sample_table: one row per sample", len(tab) == 6, len(tab))
check("sample_table: tic = sum of heights", np.isclose(tab["tic"].iloc[0], 20 * 500.0))
check("sample_table: n_peaks counted", set(tab["n_peaks"]) == {20}, set(tab["n_peaks"]))
check("is_per_peak: per-peak table", SS.is_per_peak(pk))
check("is_per_peak: per-sample table is not", not SS.is_per_peak(tab))

# ---------------------------------------------------------------------------
# redundancy: two near-identical RICH samples + one SPARSE sample holding unique
# bins. Arg-max-by-brightness would take rich-rich; the cover takes rich then
# sparse because the second rich sample adds nothing new.
# ---------------------------------------------------------------------------
rich = list(range(200, 260))                    # 60 bins
spec = {"richA": rich, "richB": rich, "sparse": [300, 301, 302], "dupS": [300, 301, 302]}
spec.update({f"bgA{i}": bg for i in range(4)})  # background-only samples
spec["richA"] = spec["richA"] + bg              # rich samples also hold the background
spec["richB"] = spec["richB"] + bg
sel = SS.select_cover_samples(make_batch(spec), k_min=2, min_gain=0.5)
meta = sel.attrs["selection"]
order = sel["sample_item_id"].tolist()
check("cover: first pick is the richest sample", order[0] == "richA", order)
check("cover: second pick is a SPARSE sample (unique bins), not the twin rich one",
      order[1] in ("sparse", "dupS") and "richB" not in order[:2], order)
check("cover: bins_new records the marginal gain (80 then 3)",
      sel["bins_new"].tolist()[:2] == [80, 3], sel["bins_new"].tolist())
check("cover: coverage is cumulative and reaches 1.0",
      sel["coverage"].is_monotonic_increasing and np.isclose(sel["coverage"].iloc[-1], 1.0),
      sel["coverage"].tolist())
check("cover: table is in pick order with a 1-based pick column",
      sel["pick"].tolist() == list(range(1, len(sel) + 1)), sel["pick"].tolist())
check("cover: meta records method/k/n_bins/achieved_coverage/stop_reason",
      meta["method"] == "presence-cover" and meta["k"] == len(sel)
      and meta["n_bins"] == 83 and np.isclose(meta["achieved_coverage"], 1.0)
      and meta["stop_reason"] in (SS.STOP_GAIN, SS.STOP_EXHAUSTED), meta)
check("cover: meta records the binning tolerance = BATCH_TOL_PPM (the merge's too)",
      meta["tol_ppm"] == SS.BATCH_TOL_PPM == 6.0, meta.get("tol_ppm"))
from peaky.batch import assign_batch as _AB  # noqa: E402
check("assign_batch.DEFAULT_TOL_PPM IS sampling.BATCH_TOL_PPM (selection == merge binning)",
      _AB.DEFAULT_TOL_PPM == SS.BATCH_TOL_PPM)
check("cover: schema has sample_item_id/role/pick/bins_new/coverage",
      {"sample_item_id", "role", "pick", "bins_new", "coverage"} <= set(sel.columns),
      list(sel.columns))
check("select_cover_sample_ids == table ids",
      SS.select_cover_sample_ids(make_batch(spec), k_min=2, min_gain=0.5) == order)

# ---------------------------------------------------------------------------
# prevalence gate: a bin seen in ONE sample only never enters the universe, and
# coverage is computed over the gated universe.
# ---------------------------------------------------------------------------
spec2 = {f"s{i}": bg for i in range(8)}
spec2["s3"] = bg + [999.0]                      # a singleton bin in s3 only
spec2["s5"] = bg + [500.0]; spec2["s6"] = bg + [500.0]   # a 2-sample bin
sel2 = SS.select_cover_samples(make_batch(spec2), k_min=1, min_gain=0.5)
m2 = sel2.attrs["selection"]
check("gate: singleton bin excluded from the universe (n_bins = 21, 1 gated)",
      m2["n_bins"] == 21 and m2["n_bins_gated"] == 1 and m2["n_bins_total"] == 22, m2)
check("gate: the singleton's sample is NOT preferred (s5 or s6 picked first)",
      sel2["sample_item_id"].iloc[0] in ("s5", "s6"), sel2["sample_item_id"].tolist())
check("gate: coverage 1.0 over the gated universe after one pick",
      np.isclose(sel2["coverage"].iloc[0], 1.0), sel2["coverage"].tolist())
sel2b = SS.select_cover_samples(make_batch(spec2), k_min=1, min_gain=0.5, min_prevalence=1)
check("gate: min_prevalence=1 keeps every bin",
      sel2b.attrs["selection"]["n_bins"] == 22, sel2b.attrs["selection"])

# ---------------------------------------------------------------------------
# stop rules
# ---------------------------------------------------------------------------
# (a) gain-floor: 3 samples each holding a big exclusive block, then a long tail
# of samples each adding ONE tiny 2-sample bin (< 0.5 % of the universe).
rng = np.random.default_rng(0)
big = {"b0": list(range(1000, 1400)), "b1": list(range(2000, 2300)), "b2": list(range(3000, 3200))}
spec3 = {k: v + bg for k, v in big.items()}
spec3.update({k + "x": v + bg for k, v in big.items()})   # twins: the blocks pass the gate
for i in range(40):                             # tail: each pair shares one unique bin
    spec3[f"t{i:02d}a"] = bg + [5000 + i]
    spec3[f"t{i:02d}b"] = bg + [5000 + i]
sel3 = SS.select_cover_samples(make_batch(spec3), k_min=3, k_max=30, min_gain=0.005)
m3 = sel3.attrs["selection"]
check("stop: gain-floor after k_min (3 big picks, tail rejected)",
      m3["stop_reason"] == SS.STOP_GAIN and m3["k"] == 3
      and sel3["sample_item_id"].tolist() == ["b0", "b1", "b2"], m3)
check("stop: next_gain recorded as the rejected pick's fraction",
      0 < m3["next_gain"] < 0.005, m3)
check("stop: achieved coverage < 1 (the tail bins are uncovered)",
      m3["achieved_coverage"] < 1.0, m3)
check("k_max_warning: None on a gain-floor stop", SS.k_max_warning(m3) is None)

# (b) k_min defers the gain floor: with k_min=6 the tail IS taken up to 6
sel3b = SS.select_cover_samples(make_batch(spec3), k_min=6, k_max=30, min_gain=0.005)
check("stop: k_min=6 takes 3 big + 3 tail picks before the gain floor applies",
      sel3b.attrs["selection"]["k"] == 6 and sel3b.attrs["selection"]["stop_reason"] == SS.STOP_GAIN,
      sel3b.attrs["selection"])
check("stop: tail picks are 'cover' role with bins_new = 1",
      sel3b["role"].tolist() == ["cover"] * 6 and sel3b["bins_new"].tolist()[3:] == [1, 1, 1],
      sel3b[["role", "bins_new"]].to_dict("records"))

# (c) k_max binds while still gaining -> 'k_max' + a warning
spec4 = {f"u{i:02d}": bg + [7000 + i, 7000 + i + 0.0] for i in range(20)}
for i in range(20):                             # every exclusive bin shared by a twin
    spec4[f"v{i:02d}"] = bg + [7000 + i]
sel4 = SS.select_cover_samples(make_batch(spec4), k_min=2, k_max=5, min_gain=0.0)
m4 = sel4.attrs["selection"]
check("stop: k_max binds -> stop_reason 'k_max', k == k_max",
      m4["stop_reason"] == SS.STOP_KMAX and m4["k"] == 5, m4)
check("k_max_warning: text names k_max and the achieved coverage",
      (SS.k_max_warning(m4) or "").startswith("selection hit k_max=5"), SS.k_max_warning(m4))
check("describe: one line mentioning the stop reason",
      "stop=k_max" in SS.describe(m4), SS.describe(m4))

# (d) exhausted: every universe bin covered before k_min -> pad to k_min with the
# richest remaining samples, role 'pad', bins_new 0
spec5 = {"a": bg + [50, 51], "b": bg + [50, 51], "c": bg, "d": bg, "e": bg, "f": bg,
         "g": bg, "h": bg}
pk5 = make_batch(spec5)
pk5.loc[pk5.sample_item_id == "h", "height"] = 5000.0      # h is the richest pad
sel5 = SS.select_cover_samples(pk5, k_min=3, min_gain=0.005)
m5 = sel5.attrs["selection"]
check("stop: exhausted after 1 pick (a or b covers everything)",
      m5["stop_reason"] == SS.STOP_EXHAUSTED and sel5["sample_item_id"].iloc[0] in ("a", "b"), m5)
check("pad: padded to k_min=3, pads are role 'pad' with bins_new 0",
      len(sel5) == 3 and sel5["role"].tolist() == ["cover", "pad", "pad"]
      and sel5["bins_new"].tolist()[1:] == [0, 0], sel5[["sample_item_id", "role", "bins_new"]].to_dict("records"))
check("pad: richest remaining sample (h) is the first pad",
      sel5["sample_item_id"].iloc[1] == "h", sel5["sample_item_id"].tolist())

# (e) n <= k_min -> all samples taken
tiny = make_batch({"a": bg, "b": bg + [60], "c": bg + [60]})
selt = SS.select_cover_samples(tiny)
check("tiny batch (n=3 <= k_min): all 3 taken", len(selt) == 3
      and selt.attrs["selection"]["k"] == 3, selt.attrs["selection"])
one = make_batch({"only": bg})
sel1 = SS.select_cover_samples(one)
check("1-sample batch: the sample is taken (gate relaxed to prevalence >= 1)",
      len(sel1) == 1 and sel1.attrs["selection"]["n_bins"] == 20, sel1.attrs["selection"])

# ---------------------------------------------------------------------------
# determinism + input validation
# ---------------------------------------------------------------------------
a1 = SS.select_cover_samples(make_batch(spec3))
a2 = SS.select_cover_samples(make_batch(spec3))
check("deterministic: identical picks on re-run",
      a1["sample_item_id"].tolist() == a2["sample_item_id"].tolist()
      and a1.attrs["selection"] == a2.attrs["selection"])
try:
    SS.select_cover_samples(tab)                # per-sample table -> no mz/height
    check("per-sample table raises ValueError", False, "no error")
except ValueError:
    check("per-sample table raises ValueError", True)
empty = pd.DataFrame(columns=["sample_item_id", "datetime_utc", "height", "mz"])
esel = SS.select_cover_samples(empty)
check("empty peaks -> empty selection with role column + meta",
      len(esel) == 0 and "role" in esel.columns and esel.attrs["selection"]["k"] == 0,
      list(esel.columns))
nc = make_batch(spec3).drop(columns=["datetime_utc", "sample_item_name"])
check("no clock / name columns -> still selects", len(SS.select_cover_samples(nc)) >= 3)

# ---------------------------------------------------------------------------
# pooled groups: one greedy over the pool, per-group coverage recorded, and a
# quiet group is NOT starved because presence (not brightness) is the objective
# ---------------------------------------------------------------------------
LOUD, QUIET = "zone 1", "zone 2"
spec6 = {}
for i in range(8):
    spec6[f"L{i}"] = bg + list(range(400 + 5 * i, 400 + 5 * i + 5))   # loud: 5 unique bins each
for i in range(8):
    spec6[f"Q{i}"] = bg + list(range(600 + 5 * i, 600 + 5 * i + 5))   # quiet: same structure
pool = make_batch(spec6)
pool["sample_batch_name"] = np.where(pool.sample_item_id.str.startswith("L"), LOUD, QUIET)
pool.loc[pool.sample_batch_name == LOUD, "height"] *= 100           # loud group 100x brighter
for i in range(8):                                                   # every bin in >= 2 samples
    twin = pool[pool.sample_item_id == f"L{i}"].assign(sample_item_id=f"L{i}x")
    twin2 = pool[pool.sample_item_id == f"Q{i}"].assign(sample_item_id=f"Q{i}x")
    pool = pd.concat([pool, twin, twin2], ignore_index=True)
psel = SS.select_cover_samples(pool, group_col="sample_batch_name", k_min=2, min_gain=0.0, k_max=16)
pm = psel.attrs["selection"]
check("pool: group column carried on the selected rows",
      "sample_batch_name" in psel.columns and set(psel["sample_batch_name"]) == {LOUD, QUIET},
      list(psel.columns))
check("pool: per-group coverage recorded for both groups",
      set(pm["coverage_by_group"]) == {LOUD, QUIET}, pm)
check("pool: the QUIET (100x dimmer) group is fully covered too",
      np.isclose(pm["coverage_by_group"][QUIET], 1.0), pm["coverage_by_group"])
check("pool: picks_by_group counts sum to k",
      sum(pm["picks_by_group"].values()) == pm["k"], pm)
try:
    SS.select_cover_samples(pool, group_col="nope")
    check("pool: missing group_col raises KeyError", False, "no error")
except KeyError:
    check("pool: missing group_col raises KeyError", True)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
