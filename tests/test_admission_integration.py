"""Integration tests for the admission gate: the batch reach-through.

tests/test_admission.py covers the gate's arithmetic in isolation. This file
covers the WIRING, which the unit tests cannot see:

  * `assign_batch.run` builds ONE occurrence table from the batch time series
    and hands it to every per-sample `assign.run`, stamped with the spectra
    count and the binning tolerance a per-peak lookup needs;
  * the derived threshold it resolves lands in `batch_summary.json`'s
    `admission` block, inside the clamp, with a real persistent-bin count;
  * the per-file admission stamp survives the merge into `merged_ledger.csv`;
  * `peaky batch` / `peaky pool` forward all three gate knobs to the pipeline.

No network and no real assignment: `assign.run` is replaced by a stub that does
exactly the admission part of the real thing (stamp, then commit), and the
Mascope client is stubbed out. Run: python3 tests/test_admission_integration.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.assignment import admission as ADM   # noqa: E402
from peaky.assignment import assign as A        # noqa: E402
from peaky.assignment import ledger as L        # noqa: E402
from peaky.assignment import passes as P        # noqa: E402
from peaky.batch import assign_batch as AB      # noqa: E402
from peaky.io import io_mascope as IO           # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# a synthetic batch: 40 spectra. 4 recurrent-weak ions (2 cps, in ~95 % of
# spectra -> the population the persistence path exists to recover), 3 bright
# ions (500 cps, in half the spectra), 200 transient noise bins. Bimodal by
# construction, so "auto" has a real split to find.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
N_SPECTRA = 40
RECUR = [264.0361, 278.0519, 294.0832, 310.0781]
BRIGHT = [201.0243, 250.5121, 333.3331]
rows = []
for s in range(N_SPECTRA):
    sid = f"s{s:03d}"
    for mz in RECUR:
        if rng.random() < 0.95:
            rows.append((sid, mz * (1 + rng.normal(0, 2e-6)), 2.0 + rng.random()))
    for mz in BRIGHT:
        if rng.random() < 0.5:
            rows.append((sid, mz * (1 + rng.normal(0, 2e-6)), 500.0 + 50 * rng.random()))
for mz in rng.uniform(200, 400, 200):
    for s in rng.choice(N_SPECTRA, size=rng.integers(1, 3), replace=False):
        rows.append((f"s{s:03d}", mz * (1 + rng.normal(0, 2e-6)), 1.0 + 2 * rng.random()))
TS = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])
TS["peak_id"] = [f"p{i}" for i in range(len(TS))]
TS["datetime_utc"] = pd.Timestamp("2026-06-20T00:00:00Z") + pd.to_timedelta(
    TS["sample_item_id"].str[1:].astype(int) * 60, unit="s")
REPS = ["s000", "s001"]          # the two "selected" samples

# ---------------------------------------------------------------------------
# the stub: everything assign.run does to the ledger for admission, and nothing
# else. It RECORDS what it was handed so the test can inspect the table that
# actually crossed the boundary.
# ---------------------------------------------------------------------------
seen: list = []


def fake_assign_run(sample_id, context="ambient-air", *, cfg=None, occurrence=None,
                    log=print, **kw):
    seen.append({"sample_id": sample_id, "cfg": cfg, "occurrence": occurrence,
                 "adducts": kw.get("adducts"), "has_ts": kw.get("ts_peaks") is not None})
    sub = TS[TS["sample_item_id"] == sample_id]
    led = L.new_ledger(sub[["peak_id", "mz", "height"]].reset_index(drop=True))
    cfg = cfg or P.PassConfig()
    cfg.noise_edge_cps = 100.0                     # a gate the weak ions sit under
    adm = ADM.stamp_admission(led, cfg, occurrence)
    # commit every ADMITTED peak, exactly as a pass would: only what the gate let
    # through can ever reach the ledger as an assignment.
    for pid in led.loc[ADM.admissible(led, cfg), "peak_id"]:
        L.commit_assignment(led, pid, neutral_formula="C8H10O6", adduct="[M-H]-",
                            ion_score=0.80, compound_score=0.80, eff_score=0.78,
                            eff_margin=0.2, tied=False, ppm_error=0.4, pass_no=1,
                            method="cheminfo+grid", confidence="Good",
                            commentary="stub", alternatives=[])
    # cfg.height_cutoff is the RESOLVED gate (x_edge x edge), and assign.run
    # stamps it under that name -- `height_cutoff_cps` is the absolute-override
    # KNOB, which is not what this is.
    st = {"noise_edge_cps": cfg.noise_edge_cps, "height_gate_cps": cfg.height_cutoff,
          "admitted": {k: adm[k] for k in ("height", "occurrence", "rejected")}}
    return {"ledger": led, "stats": st, "plausibility_audit": []}


_real = {"assign_run": A.run, "connect": IO.connect, "fetch_peaks": IO.fetch_peaks}
A.run = fake_assign_run
IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
IO.fetch_peaks = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network in tests"))
RUN_DIR = tempfile.mkdtemp(prefix="peaky-admission-")
try:
    res = AB.run(peaks=TS, reagent="Br", out_dir=RUN_DIR, ts_peaks=TS,
                 sample_ids=list(REPS), n_jobs=1, log=lambda *a: None)
finally:
    A.run, IO.connect, IO.fetch_peaks = (_real["assign_run"], _real["connect"],
                                         _real["fetch_peaks"])

# --- the table that actually reached assign ---------------------------------
check("assign.run was called once per selected sample",
      [s["sample_id"] for s in seen] == REPS, [s["sample_id"] for s in seen])
tabs = [s["occurrence"] for s in seen]
check("every per-sample run got an occurrence table (built once, shared)",
      all(t is not None for t in tabs) and all(t is tabs[0] for t in tabs))
t0 = tabs[0]
check("the table carries the SPECTRA COUNT of the whole batch, not of the rep sample",
      int(t0.attrs.get("n_samples", 0)) == N_SPECTRA, t0.attrs)
check("the table carries the binning TOLERANCE the per-peak lookup needs",
      float(t0.attrs.get("tol_ppm")) == AB.DEFAULT_TOL_PPM, t0.attrs)
check("...and that tolerance is the one batch tolerance",
      float(t0.attrs["tol_ppm"]) == 6.0, t0.attrs)
check("the table is usable for a lookup as handed over (no extra arguments)",
      np.all(np.isfinite(ADM.lookup_occurrence(RECUR, t0))),
      ADM.lookup_occurrence(RECUR, t0))
check("the cfg handed to assign carries the resolved threshold after stamping",
      seen[0]["cfg"].occurrence_threshold is not None, seen[0]["cfg"].occurrence_threshold)

# --- the summary block ------------------------------------------------------
adm = res["summary"]["admission"]
check("batch_summary admission block carries all seven keys",
      {"occurrence_min", "occurrence_threshold", "n_peaks", "n_persistent_peaks",
       "n_persistent_traces", "n_spectra", "tol_ppm"} == set(adm), sorted(adm))
check("admission.occurrence_min is the KNOB as given ('auto' by default)",
      adm["occurrence_min"] == "auto", adm)
check("admission.occurrence_threshold is a resolved number inside the clamp",
      adm["occurrence_threshold"] is not None
      and ADM.AUTO_MIN <= adm["occurrence_threshold"] <= ADM.AUTO_MAX, adm)
check("admission.n_persistent_peaks is non-zero (the recurrent population was found)",
      adm["n_persistent_peaks"] >= len(RECUR) and adm["n_persistent_peaks"] < adm["n_peaks"], adm)
# the trace-weighted count reads as IONS: the 4 recurrent ions (+ at most the 3
# half-time bright ones, which sit right at the split), never their ~150 peaks
check("admission.n_persistent_traces counts persistent IONS, not their peaks",
      len(RECUR) <= adm["n_persistent_traces"] <= len(RECUR) + len(BRIGHT), adm)
check("admission.n_spectra is the whole batch, tol_ppm the shared tolerance",
      adm["n_spectra"] == N_SPECTRA and adm["tol_ppm"] == 6.0, adm)

# --- the merged ledger ------------------------------------------------------
merged = pd.read_csv(os.path.join(res["out_dir"], "merged_ledger.csv"))
check("merged_ledger.csv has the admitted_by column", "admitted_by" in merged.columns,
      list(merged.columns))
check("merged_ledger.csv has the occurrence column", "occurrence" in merged.columns,
      list(merged.columns))
_occ_rows = merged[merged["admitted_by"] == "occurrence"]
check("the merged ledger really contains persistence-only peaks (the point of the path)",
      len(_occ_rows) >= 1, merged["admitted_by"].value_counts().to_dict())
check("every persistence-only merged peak sits above the resolved threshold",
      bool((_occ_rows["occurrence"] >= adm["occurrence_threshold"]).all()),
      _occ_rows[["mz", "occurrence"]].to_dict("records"))
check("per-file stats record the admitted counts by path",
      all(set(s["admitted"]) == {"height", "occurrence", "rejected"}
          and s["admitted"]["occurrence"] > 0 for s in res["summary"]["per_file"]),
      [s.get("admitted") for s in res["summary"]["per_file"]])
# assign_batch copies the per-sample stats dict into the summary verbatim, so a
# stub that spells the gate wrong ships a dead key in batch_summary.json. Same
# contract tests/test_assign_batch.py pins for a real run.
check("per-file stats carry the RESOLVED gate as height_gate_cps, not the knob's name",
      all("height_gate_cps" in s and "height_cutoff_cps" not in s
          for s in res["summary"]["per_file"]),
      [sorted(s) for s in res["summary"]["per_file"]])
shutil.rmtree(RUN_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI reach-through: `batch` and `pool` must forward all three gate knobs to the
# pipeline. The credential check is stubbed (CI runs with no .env and no creds).
# ---------------------------------------------------------------------------
from peaky import cli   # noqa: E402
from peaky import pipeline as PL   # noqa: E402

_creds, _rb, _rp = cli._require_creds, PL.run_batch, PL.run_pooled_batches
calls: list = []


def _fake_run(**kw):
    calls.append(kw)
    return {"ctx": SimpleNamespace(out_dir="/dev/null/run"), "group_runs": []}


cli._require_creds = lambda: None
PL.run_batch = _fake_run
PL.run_pooled_batches = _fake_run
PARSER = cli.build_parser()
try:
    with tempfile.TemporaryDirectory() as d:
        a = PARSER.parse_args(["batch", "--batch", "B", "--out-dir", d, "--no-report",
                               "--occurrence-min", "0.55",
                               "--height-cutoff-x-edge", "3",
                               "--height-cutoff", "250"])
        cli.cmd_batch(a)
        kw = calls[-1]
        check("batch forwards the occurrence knob", kw.get("occurrence_min") == 0.55, kw)
        check("batch forwards the edge multiplier", kw.get("height_cutoff_x_edge") == 3.0, kw)
        check("batch forwards the absolute cutoff as height_cutoff_cps",
              kw.get("height_cutoff_cps") == 250.0, kw)

        a = PARSER.parse_args(["batch", "--batch", "B", "--out-dir", d, "--no-report"])
        cli.cmd_batch(a)
        kw = calls[-1]
        # the edge multiple is NOT resolved at the CLI: the flag defaults to None so
        # the reagent profile's own multiple can win, and the package default applies
        # only when neither is given. The pipeline resolves it where the profile is.
        check("batch defaults: 'auto' knob, multiple left to the profile, no absolute cutoff",
              kw.get("occurrence_min") == "auto" and kw.get("height_cutoff_x_edge") is None
              and kw.get("height_cutoff_cps") is None, kw)

        a = PARSER.parse_args(["pool", "--batches", "B1,B2", "--out-dir", d, "--no-report",
                               "--occurrence-min", "auto",
                               "--height-cutoff-x-edge", "5",
                               "--height-cutoff", "12.5"])
        cli.cmd_pool(a)
        kw = calls[-1]
        check("pool forwards all three gate knobs",
              kw.get("occurrence_min") == "auto" and kw.get("height_cutoff_x_edge") == 5.0
              and kw.get("height_cutoff_cps") == 12.5, kw)
finally:
    cli._require_creds, PL.run_batch, PL.run_pooled_batches = _creds, _rb, _rp

# an unparseable knob is a PARSER error (exit 2), not a float() crash halfway
# through a batch run
for bad in ("maybe", "0.4.4", ""):
    try:
        PARSER.parse_args(["batch", "--batch", "B", "--occurrence-min", bad])
        check(f"--occurrence-min {bad!r} is rejected by the parser", False, "parsed")
    except SystemExit as e:
        check(f"--occurrence-min {bad!r} is rejected by the parser", e.code == 2, e.code)
for bad in ("maybe", "1.2.3"):
    try:
        PARSER.parse_args(["pool", "--batches", "B", "--height-cutoff-x-edge", bad])
        check(f"--height-cutoff-x-edge {bad!r} is rejected by the parser", False, "parsed")
    except SystemExit as e:
        check(f"--height-cutoff-x-edge {bad!r} is rejected by the parser", e.code == 2, e.code)
# 'auto' is a VALUE for the multiple too: derive the floor from the batch's own
# peaks (the package policy when nothing is set; typed explicitly it outranks a
# profile's number)
_a = PARSER.parse_args(["pool", "--batches", "B", "--height-cutoff-x-edge", "AUTO"])
check("--height-cutoff-x-edge auto is accepted (any case) as the policy token",
      _a.height_cutoff_x_edge == "auto", _a.height_cutoff_x_edge)
check("gate_config carries an explicit 'auto' multiple onto the cfg",
      PL.gate_config(height_cutoff_x_edge="auto").height_cutoff_x_edge == "auto"
      and PL.gate_config(height_cutoff_x_edge=2).height_cutoff_x_edge == 2.0)
try:
    PL.gate_config(height_cutoff_x_edge="never")
    check("gate_config rejects an unknown string multiple", False)
except ValueError:
    check("gate_config rejects an unknown string multiple", True)


# ---------------------------------------------------------------------------
# ONE batch tolerance, end to end. The occurrence table and the merge bin at the
# run's `tol_ppm`; the SELECTOR must bin at the same number, or a bin the cover
# was chosen to reach is not the bin the table and the merge see. (It used to
# fall through to timeseries' 5 ppm while both of those used 6.)
# ---------------------------------------------------------------------------
from peaky.batch import sampling as SS   # noqa: E402

check("the merge's default tolerance IS the one batch constant",
      AB.DEFAULT_TOL_PPM == SS.BATCH_TOL_PPM, AB.DEFAULT_TOL_PPM)
_sel_kw: list = []


def fake_select(peaks, **kw):
    _sel_kw.append(kw)
    out = pd.DataFrame({"sample_item_id": list(REPS), "pick": [1, 2],
                        "role": ["cover", "cover"], "bins_new": [10, 2],
                        "coverage": [0.8, 1.0]})
    out.attrs["selection"] = {"method": "presence-cover", "k": 2}
    return out


_real_sel, SS.select_cover_samples = SS.select_cover_samples, fake_select
A.run, _real_connect = fake_assign_run, IO.connect
IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
RUN_DIR2 = tempfile.mkdtemp(prefix="peaky-admission-sel-")
try:
    AB.run(peaks=TS, reagent="Br", out_dir=RUN_DIR2, ts_peaks=TS, n_jobs=1,
           log=lambda *a: None)
finally:
    SS.select_cover_samples = _real_sel
    A.run, IO.connect = _real["assign_run"], _real_connect
    shutil.rmtree(RUN_DIR2, ignore_errors=True)

check("assign_batch hands the selector the run's own binning tolerance",
      len(_sel_kw) == 1 and _sel_kw[0].get("tol_ppm") == AB.DEFAULT_TOL_PPM
      == SS.BATCH_TOL_PPM, _sel_kw)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
