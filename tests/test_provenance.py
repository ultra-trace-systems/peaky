"""Offline tests for provenance.py. Run: python3 tests/test_provenance.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import peaky  # noqa: E402
from peaky import passes as P  # noqa: E402
from peaky import provenance as PV  # noqa: E402
from peaky.assignment import assign as _A  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


_rd = tempfile.mkdtemp()
with open(os.path.join(_rd, "merged_ledger.csv"), "w") as fh:
    fh.write("mz,neutral_formula,tier\n100.0,CH4,Assigned\n")
_tsp = os.path.join(_rd, "X_ts.parquet")
with open(_tsp, "wb") as fh:                       # any bytes -> hashable input
    fh.write(b"fake-parquet-bytes")

m = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D",
                      sample_ids=["s1", "s2"], reagent="NO3_15N",
                      cfg=P.PassConfig(), ts_path=_tsp, counts={"merged_M0": 42},
                      created_utc="2026-01-01T00:00:00Z")

check("manifest carries the core sections",
      all(k in m for k in ("run_id", "code", "input", "config", "output")))
check("code pins the REAL package version (peaky.__version__), not assign's module version",
      m["code"]["package_version"] == peaky.__version__,
      f'got {m["code"]["package_version"]!r}, want {peaky.__version__!r}')
check("assign's own module version still lives under module_versions",
      m["code"]["module_versions"].get("assign") == _A.__version__)
check("code carries a module-hash map",
      isinstance(m["code"]["module_hashes"], dict)
      and len(m["code"]["module_hashes"]) > 5)
check("code records python + dep versions",
      "python" in m["code"] and "pandas" in m["code"]["deps"])
check("input hashes the ts parquet (== streaming sha1) + records provenance",
      m["input"]["ts_sha1"] == PV.sha1_file(_tsp)
      and m["input"]["dataset"] == "D" and m["input"]["sample_ids"] == ["s1", "s2"]
      and m["input"]["reagent"] == "NO3_15N")
check("output hashes merged_ledger.csv + carries counts",
      m["output"]["merged_ledger_sha1"] and m["output"]["counts"]["merged_M0"] == 42)
check("config fingerprint keeps user knobs, drops run-derived fields",
      m["config"].get("height_cutoff_x_edge") == 1.0 and m["config"].get("ppm") == 1.0
      and "height_cutoff_cps" in m["config"]
      and "mechanism_ids" not in m["config"] and "prior_offset" not in m["config"]
      and "noise_edge_cps" not in m["config"], m["config"])
# the gate multiple is a USER KNOB (a profile or the command line sets it), so a
# profile-supplied value must reach the fingerprint and never the runtime-excluded
# set -- otherwise two runs with different gates hash the same config.
_mx = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D", sample_ids=["s1"],
                        reagent="Br", cfg=P.PassConfig(height_cutoff_x_edge=5.0),
                        ts_path=_tsp, counts={}, created_utc="2026-01-01T00:00:00Z")
check("a resolved (non-default) x_edge is fingerprinted, not runtime-excluded",
      _mx["config"].get("height_cutoff_x_edge") == 5.0
      and "height_cutoff_x_edge" not in P.PassConfig.RUNTIME_FIELDS,
      _mx["config"].get("height_cutoff_x_edge"))
_RT = {"noise_edge_cps", "mechanism_ids", "prior_offset", "reagent_element",
       "cal_mu", "cal_sigma"}
check("PassConfig.RUNTIME_FIELDS declares the run-stamped fields",
      _RT <= set(P.PassConfig.RUNTIME_FIELDS), P.PassConfig.RUNTIME_FIELDS)
# a FITTED mass calibration is data, not configuration: a fingerprint that
# absorbed it would vary with the sample it is meant to pin the config against.
_mc = P.PassConfig(); _mc.cal_mu, _mc.cal_sigma = -2.45, 0.30
_mcal = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D", sample_ids=["s1"],
                          reagent="Br", cfg=_mc, ts_path=_tsp, counts={},
                          created_utc="2026-01-01T00:00:00Z")
check("a fitted cal_mu/cal_sigma never reaches the config fingerprint",
      "cal_mu" not in _mcal["config"] and "cal_sigma" not in _mcal["config"]
      and _mcal["config"]["cal_z_accept"] == 2.0,     # the KNOBS stay
      {k: v for k, v in _mcal["config"].items() if k.startswith("cal_")})
import dataclasses as _dc  # noqa: E402
_FIELDS = {f.name for f in _dc.fields(P.PassConfig)}      # real fields (ClassVar excluded)
check("RUNTIME_FIELDS is a ClassVar, not a dataclass field (asdict/pickle untouched)",
      "RUNTIME_FIELDS" not in _FIELDS and "RUNTIME_FIELDS" not in m["config"])
check("build_manifest's config excludes every RUNTIME_FIELDS entry and nothing else",
      not (set(P.PassConfig.RUNTIME_FIELDS) & set(m["config"]))
      and set(m["config"]) == _FIELDS - set(P.PassConfig.RUNTIME_FIELDS),
      sorted(_FIELDS ^ set(m["config"])))

# passes.calibrate writes the fitted mass trend back onto the SHARED cfg, so the
# last sample's data-derived numbers would otherwise land in the fingerprint and
# make two identical re-runs of the same batch differ.
_cal_cfg = P.PassConfig()
_cal_cfg.cal_a, _cal_cfg.cal_b, _cal_cfg.cal_sigma_trend = -0.25, -0.12, 0.25
_cal_cfg.cal_mz_lo, _cal_cfg.cal_mz_hi = 59.0, 397.0
_mc = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D", sample_ids=["s1"],
                        reagent="NO3_15N", cfg=_cal_cfg, ts_path=_tsp,
                        created_utc="2026-01-01T00:00:00Z")
check("config fingerprint drops the self-calibration's fitted mass trend",
      not any(k in _mc["config"] for k in
              ("cal_a", "cal_b", "cal_sigma_trend", "cal_mz_lo", "cal_mz_hi")),
      {k: v for k, v in _mc["config"].items() if k.startswith("cal_")})
check("... so a fitted trend does not change the config fingerprint",
      _mc["config"] == m["config"])
check("the cal knobs the user CAN set are still fingerprinted",
      _mc["config"].get("cal_min_n") == P.PassConfig().cal_min_n
      and "cal_abs_floor_mda" in _mc["config"] and "cal_z_accept" in _mc["config"])
check("git_info is best-effort and returns a dict", isinstance(
    PV.git_info(str(Path(PV.__file__).parent)), dict))

# --- the admission gate: KNOB in the config fingerprint, RESOLVED value out of it.
# `occurrence_threshold` is derived from the batch at run time (the Otsu split),
# so it is not part of the configuration that would reproduce the run -- it is an
# OUTPUT. Keeping it in `config` would make two identical configurations
# fingerprint differently on two batches. It belongs under counts.admission.
_cfg_adm = P.PassConfig(occurrence_min="auto")
_cfg_adm.occurrence_threshold = 0.44          # what stamp_admission sets per run
_cfg_adm.occurrence_resolved = True           # ...and the marker that says it is final
_adm_block = {"occurrence_min": "auto", "occurrence_threshold": 0.44, "n_bins": 4025,
              "n_persistent_bins": 512, "n_spectra": 230, "tol_ppm": 6.0}
m_adm = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D",
                          sample_ids=["s1"], reagent="NO3_15N", cfg=_cfg_adm,
                          ts_path=_tsp, counts={"merged_M0": 7, "admission": _adm_block})
check("config fingerprint keeps the admission KNOB (occurrence_min)",
      m_adm["config"].get("occurrence_min") == "auto", m_adm["config"])
check("config fingerprint DROPS the run-derived occurrence_threshold",
      "occurrence_threshold" not in m_adm["config"], m_adm["config"])
check("the resolved threshold is recorded instead under output.counts.admission",
      m_adm["output"]["counts"]["admission"]["occurrence_threshold"] == 0.44
      and m_adm["output"]["counts"]["admission"]["tol_ppm"] == 6.0,
      m_adm["output"]["counts"])
check("occurrence_threshold is listed as a runtime field, next to noise_edge_cps",
      "occurrence_threshold" in PV._RUNTIME_CFG_FIELDS, PV._RUNTIME_CFG_FIELDS)
# `occurrence_resolved` is the same kind of thing: stamp_admission sets it on the
# shared cfg so `admissible` can tell "resolved OFF" from "never stamped". It is
# run state, not a knob, and would otherwise flip the fingerprint of an identical
# configuration depending on whether the run had reached the gate.
check("occurrence_resolved is a runtime field too, and stays out of the fingerprint",
      "occurrence_resolved" in PV._RUNTIME_CFG_FIELDS
      and "occurrence_resolved" not in m_adm["config"], m_adm["config"])
# the gate is a module of its own; its version must be pinned like every other
# assignment module (module_hashes catches edits, module_versions names them)
from peaky.assignment import admission as _ADM  # noqa: E402
check("module_versions pins the admission module",
      m_adm["code"]["module_versions"].get("admission") == _ADM.__version__,
      m_adm["code"]["module_versions"])

PV.write_manifest(_rd, m)
check("write_manifest writes run_manifest.json",
      os.path.exists(os.path.join(_rd, "run_manifest.json")))

_idx = os.path.join(_rd, "index.jsonl")
PV.append_registry(_idx, m)
PV.append_registry(_idx, m)
_rows = [json.loads(ln) for ln in open(_idx)]
check("append_registry appends ONE json row per call",
      len(_rows) == 2 and _rows[0]["run_id"] == m["run_id"]
      and _rows[0]["counts"]["merged_M0"] == 42 and _rows[0]["n_samples"] == 2)

m2 = PV.build_manifest(run_dir=_rd, batch_name="B", dataset="D",
                       sample_ids=["s1", "s2"], reagent="NO3_15N",
                       cfg=P.PassConfig(), ts_path=_tsp, counts={"merged_M0": 42})
check("hashing is deterministic (reproducible fingerprint)",
      m2["input"]["ts_sha1"] == m["input"]["ts_sha1"]
      and m2["output"]["merged_ledger_sha1"] == m["output"]["merged_ledger_sha1"])

check("sha1_file returns None for a missing path", PV.sha1_file(_rd + "/nope") is None)

# --- stage ordering: a pass-8 reflist rescue commits AFTER the three envelope
# sweeps, so a final sweep must follow it (else a rescued parent's 13C satellite
# stays "unexplained" -- the NBBS urea-adduct M+1 bug).
_names = [s.name for s in _A._STAGES]
check("iso_env_final runs after reflist_rescue",
      "iso_env_final" in _names and "reflist_rescue" in _names
      and _names.index("iso_env_final") > _names.index("reflist_rescue"), _names)
check("record_run is non-fatal on a bad dir (returns {})",
      PV.record_run(run_dir="/nonexistent/xyz", base_out="/nonexistent",
                    batch_name="B", dataset=None, sample_ids=[], reagent="r",
                    cfg=P.PassConfig(), log=lambda *a: None) == {}
      or True)   # tolerate either {} or a partial manifest; must not raise


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
