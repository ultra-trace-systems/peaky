"""Offline batch harness over the episode fixture: `peaky batch`'s own `assign_batch.run`
with the server and the per-file engine stubbed. Every file's ledger is its own spectrum's
peaks (the fixture's peak ids) with the fixture's known ions committed Assigned; the rest
stay unexplained. Used by tests/test_episodes.py and by `python harness.py <out_dir>`,
which wrote the goldens on the parent of the episode stage (the code before it):
golden_no_episode/ (the batch without the plume) and golden_plume_base/ (the full batch)."""
from __future__ import annotations

import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import make_episodes as MK  # noqa: E402
from peaky.assignment import assign as A  # noqa: E402
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.assignment import ledger as L  # noqa: E402
from peaky.batch import assign_batch as AB  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402

GOLDEN_DIR = os.path.join(HERE, "golden_no_episode")          # the no-episode batch, base code
GOLDEN_OFF_DIR = os.path.join(HERE, "golden_plume_base")       # the plume batch, base code (= episodes off)
# run artifact -> its golden's file name (a `*_ledger.csv` name is git-ignored repo-wide)
GOLDEN_FILES = {"tables/selected_samples.csv": "selected_samples.csv",
                "merged_ledger.csv": "merged.golden.csv"}


def _known(comp: dict) -> dict:
    """{line key: (neutral, adduct)} of the lines a file commits."""
    return {x["key"]: (x["neutral_formula"], x["adduct"]) for x in comp["lines"]
            if x.get("neutral_formula")}


def fake_engine(ts: pd.DataFrame, comp: dict, *, tiers: dict | None = None, alternatives: dict | None = None,
                ppm: dict | None = None, commit: dict | None = None):
    """An `assign.run` stand-in over the fixture table. The episode-evidence tests shape what
    a file commits: `tiers` {line key: tier} (a per-file cap, e.g. Candidate for an open
    rival), `alternatives` {line key: [{formula, adduct}]} (the engine's rivals),
    `ppm` {line key: ppm_error}, `commit` {line key: (neutral, adduct, files or None)} (a
    reading of a line the fixture names none for, in those files only). None of them:
    the stand-in the goldens were cut with."""
    known = _known(comp)
    tiers, alternatives, ppm, commit = tiers or {}, alternatives or {}, ppm or {}, commit or {}

    def run(sid, context="ambient-air", **kw):
        pk = ts[ts["sample_item_id"] == sid]
        led = L.new_ledger(pk[["peak_id", "mz", "height"]].reset_index(drop=True))
        tier_of = {}
        for pid, key in zip(pk["peak_id"], pk["key"]):
            na = known.get(key)
            if key in commit and (commit[key][2] is None or sid in commit[key][2]):
                na = commit[key][:2]
            if na is not None:
                n, a = na
                L.commit_assignment(led, pid, neutral_formula=n, adduct=a,
                                    ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "+",
                                    ion_score=0.95, compound_score=0.95, ppm_error=ppm.get(key, 0.1), pass_no=1,
                                    method="cheminfo+grid", confidence="High", commentary="Pass 1",
                                    alternatives=alternatives.get(key))
                tier_of[pid] = tiers.get(key, "Assigned")
        led["tier"] = led["tier"].astype(object)
        led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
        for pid, t in tier_of.items():
            if t != "Assigned":
                led.loc[led["peak_id"] == pid, "tier"] = t
        led["below_assignability"] = False
        return {"ledger": led,
                "stats": {"noise_edge_cps": 80.0, "height_gate_cps": 240.0, "side_channels": [],
                          "degeneracy_cal": {"mu": 0.0, "sigma": 0.2}},
                "plausibility_audit": [], "summaries": {}, "problems": []}
    return run


def patch(ts: pd.DataFrame, comp: dict, setattr_=setattr, **engine) -> None:
    """Stub the server and the engine (`setattr_` = a pytest monkeypatch's setattr;
    `engine` = fake_engine's shaping switches)."""
    setattr_(IO, "connect", lambda *a, **k: "CLIENT")
    setattr_(IO, "fetch_peaks", lambda client, sid, use_cache=True:
             ts[ts["sample_item_id"] == sid][["peak_id", "mz", "height"]].reset_index(drop=True))
    setattr_(IO, "estimate_offset", lambda raw: 0.0)
    setattr_(A, "run", fake_engine(ts, comp, **engine))


def run_batch(out_dir: str, ts: pd.DataFrame, **kw) -> dict:
    """One offline batch over `ts` (the fixture table, `key` column dropped) into
    `out_dir`; returns batch_summary.json. The cover's own defaults unless `kw` says."""
    t = ts.drop(columns=["key"], errors="ignore")
    AB.run(peaks=t, ts_peaks=t, reagent="Ur", batch="episode test batch", out_dir=out_dir,
           n_jobs=1, resolving_power=100_000, log=lambda *a, **k: None, **kw)
    with open(os.path.join(out_dir, "batch_summary.json")) as fh:
        return json.load(fh)


if __name__ == "__main__":     # cut a golden: harness.py <scratch out_dir> no_episode|plume_base
    out, which = sys.argv[1], sys.argv[2]
    ts, comp = MK.build(plume=(which != "no_episode"))
    patch(ts, comp)
    run_batch(out, ts)
    dest = GOLDEN_DIR if which == "no_episode" else GOLDEN_OFF_DIR
    os.makedirs(dest, exist_ok=True)
    for f, g in GOLDEN_FILES.items():
        shutil.copyfile(os.path.join(out, f), os.path.join(dest, g))
        print("golden", dest, g)
