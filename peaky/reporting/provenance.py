"""Run provenance + reproducibility index.

Every batch run writes a self-describing ``run_manifest.json`` into its run dir
(code identity + input-data hash + resolved config + output hash + counts) and
appends a compact one-line summary to a cross-run registry (``index.jsonl`` at
the out-dir root). Together these make a run REPRODUCIBLE (re-run the recorded
commit on data with the recorded ts_sha1 -> determinism tests guarantee the same
merged_ledger_sha1) and DISCOVERABLE (grep/df the registry to find or diff runs).

Pure-ish: only touches the filesystem and an optional `git` subprocess. No
network. Safe to call on a plain pip install (git_info degrades to {}).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass

__version__ = "0.1.0"

# per-sample fields that assign.run stamps onto the shared cfg at runtime -- they
# are run-derived, not user knobs, so they don't belong in the reproducible config
# fingerprint (offsets live in batch_summary.json already). The cal_* entries are
# the self-calibration's own output: passes.calibrate writes the fitted mass trend
# (cal_a / cal_b / cal_sigma_trend and the backbone coverage cal_mz_lo / cal_mz_hi)
# back onto the shared cfg, so whichever sample ran LAST would otherwise leak its
# data-derived numbers into the manifest and make two identical re-runs of the same
# batch fingerprint differently. NB cal_mu / cal_sigma are equally data-derived but
# are deliberately NOT dropped: they predate this list and manifests in the wild
# carry them as the record of the run's calibration centre, so removing them would
# be a manifest schema change rather than a fix.
_RUNTIME_CFG_FIELDS = ("mechanism_ids", "prior_offset", "reagent_element",
                       "noise_edge_cps",
                       "cal_a", "cal_b", "cal_sigma_trend",
                       "cal_mz_lo", "cal_mz_hi")


def _rel_or_abs(path: str | None, base: str) -> str | None:
    """A run-dir-relative path when `path` lives inside the run dir (e.g. a copied
    ``data/<tag>_ts.parquet``), else the absolute path (e.g. an external input TS
    referenced, not copied — see pipeline reference-not-copy). Either way ts_sha1
    pins the content, so the run stays reproducible regardless of where the TS sits."""
    if not path:
        return None
    p, b = os.path.abspath(os.path.expanduser(path)), os.path.abspath(base)
    return os.path.relpath(p, b) if p == b or p.startswith(b + os.sep) else p


def sha1_file(path: str, *, _buf: int = 1 << 20) -> str | None:
    """Streaming sha1 of a file (None if it does not exist)."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_path: str) -> dict:
    """Best-effort git identity (commit / dirty / branch). Returns {} when the
    package is not a git checkout (e.g. a plain `pip install`) or git is absent."""
    def _git(*args: str) -> str:
        return subprocess.run(["git", "-C", repo_path, *args],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        commit = _git("rev-parse", "HEAD")
        if not commit:
            return {}
        return {"commit": commit,
                "dirty": bool(_git("status", "--porcelain")),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None}
    except Exception:
        return {}


def dep_versions(names=("mascope-sdk", "pandas", "numpy")) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def build_manifest(*, run_dir: str, batch_name: str, dataset: str | None,
                   sample_ids, reagent: str, cfg, ts_path: str | None = None,
                   counts: dict | None = None, created_utc: str | None = None,
                   extra: dict | None = None) -> dict:
    """Assemble the reproducibility manifest for a finished run. Hashes the run's
    ts parquet (input) and merged_ledger.csv (output) so the run is pinned to its
    exact data + result alongside the exact code that produced it."""
    import peaky                                # the real package version (peaky.__version__)
    from peaky.assignment import assign as A   # per-module versions + hashes (incl. assign's own)

    pkg_dir = os.path.dirname(__file__)
    cfg_d = asdict(cfg) if is_dataclass(cfg) else dict(cfg or {})
    for k in _RUNTIME_CFG_FIELDS:
        cfg_d.pop(k, None)
    merged = os.path.join(run_dir, "merged_ledger.csv")
    return {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "run_dir": run_dir,
        "created_utc": created_utc,
        "code": {
            "package_version": peaky.__version__,   # NOT A.__version__ (that is assign's
            "module_versions": A.MODULE_VERSIONS,    # own module version, kept below)
            "module_hashes": A._module_hashes(),
            "git": git_info(pkg_dir),
            "python": sys.version.split()[0],
            "deps": dep_versions(),
        },
        "input": {
            "dataset": dataset,
            "batch_name": batch_name,
            "reagent": reagent,
            "sample_ids": list(sample_ids or []),
            "ts_parquet": _rel_or_abs(ts_path, run_dir),
            "ts_sha1": sha1_file(ts_path) if ts_path else None,
        },
        "config": cfg_d,
        "output": {
            "merged_ledger_sha1": sha1_file(merged),
            "counts": counts or {},
        },
        **(extra or {}),
    }


def write_manifest(run_dir: str, manifest: dict) -> str:
    p = os.path.join(run_dir, "run_manifest.json")
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return p


def append_registry(index_path: str, manifest: dict) -> str:
    """Append a compact one-line summary of `manifest` to the JSONL registry
    (created if absent). One row per run; load with pandas.read_json(lines=True)."""
    git = manifest.get("code", {}).get("git", {})
    row = {
        "run_id": manifest.get("run_id"),
        "run_dir": manifest.get("run_dir"),
        "created_utc": manifest.get("created_utc"),
        "batch_name": manifest["input"]["batch_name"],
        "dataset": manifest["input"]["dataset"],
        "reagent": manifest["input"]["reagent"],
        "n_samples": len(manifest["input"]["sample_ids"]),
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "package_version": manifest["code"]["package_version"],
        "ts_sha1": manifest["input"]["ts_sha1"],
        "merged_ledger_sha1": manifest["output"]["merged_ledger_sha1"],
        "counts": manifest["output"]["counts"],
    }
    index_path = os.path.expanduser(index_path)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return index_path


def record_run(*, run_dir: str, base_out: str, batch_name: str,
               dataset: str | None, sample_ids, reagent: str, cfg,
               ts_path: str | None = None, counts: dict | None = None,
               created_utc: str | None = None, log=print) -> dict:
    """Write the per-run manifest AND append the cross-run registry row. Returns
    the manifest. Never raises into the pipeline -- provenance must not break a
    completed run; failures are logged and swallowed."""
    try:
        manifest = build_manifest(
            run_dir=run_dir, batch_name=batch_name, dataset=dataset,
            sample_ids=sample_ids, reagent=reagent, cfg=cfg, ts_path=ts_path,
            counts=counts, created_utc=created_utc)
        write_manifest(run_dir, manifest)
        idx = append_registry(os.path.join(base_out, "index.jsonl"), manifest)
        commit = (manifest["code"]["git"] or {}).get("commit") or "no-git"
        log(f"[provenance] run_manifest.json written; indexed in {idx} "
            f"(commit {commit[:9]}, ts_sha1 {str(manifest['input']['ts_sha1'])[:9]})")
        return manifest
    except Exception as e:    # provenance is best-effort, never fatal
        log(f"[provenance] WARNING could not record run: {type(e).__name__}: {e}")
        return {}
