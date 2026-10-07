"""C46 end to end on one offline TOF-like sample: the pieces are wired.

The unit tests of tests/test_snr_policy.py and tests/test_tiers_tof_floor.py
prove each piece; this file proves `assign.run` connects them -- the class is
stamped on the cfg before the tier pass, the batch edge sizes the floor, the
stats say what happened, and the local scorer receives the counting-statistics
column, not the server's. Offline: a registered table, no network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascope_tools.composition.heuristic_filter import (  # noqa: E402
    anchor_on_monoisotopic,
    predict_isotopes,
)

from peaky.assignment import assign as A  # noqa: E402
from peaky.assignment import tiers as T  # noqa: E402
from peaky.assignment.passes.config import PassConfig  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402
from peaky.io import local_scoring as LS  # noqa: E402

SID = "offline-c46-tof"


def _tof_table(n_filler: int = 60, seed: int = 3) -> pd.DataFrame:
    """A bromide-CIMS-like TOF peak list: the [M+Br]- cluster of C10H16O6 (all
    lines), a few bright reagent-like peaks, and dim fillers; the server's
    `signal_to_noise` has nothing to do with height (as on the real TOF)."""
    rng = np.random.default_rng(seed)
    rows = []
    mzs, ints, labels = anchor_on_monoisotopic(*predict_isotopes("C10H16O6Br", -1))
    rel = ints / ints[0]
    for i, (m, r) in enumerate(zip(mzs, rel)):
        if r >= 0.01:
            rows.append({"peak_id": f"L{i}", "mz": float(m), "height": 500.0 * float(r)})
    for j in range(n_filler):
        rows.append({"peak_id": f"f{j}", "mz": 60.0 + j * 5.37, "height": float(rng.uniform(0.6, 6.0))})
    t = pd.DataFrame(rows)
    t["signal_to_noise"] = rng.uniform(0.5, 2.0, len(t))     # flat: not a signal-to-noise
    return t


@pytest.fixture(autouse=True)
def _clean():
    IO.unregister_offline_sample(SID)
    yield
    IO.unregister_offline_sample(SID)


def test_assign_run_wires_the_class_the_batch_edge_the_stats_and_the_scorers_column(monkeypatch):
    table = _tof_table()
    seen = []
    real = LS.score_candidates_local

    def spy(peaks, *a, **k):
        seen.append(peaks)
        return real(peaks, *a, **k)

    monkeypatch.setattr(LS, "score_candidates_local", spy)
    cfg = PassConfig(noise_edge_batch_cps=0.74)
    res = A.run(SID, context="ambient-air", cfg=cfg, peaks=table, use_cache=False, scoring="tof",
                adducts=["[M-H]-", "[M+Br]-", "[M+NO3]-"], reagent_n_relabel=False, log=lambda *a: None)
    st = res["stats"]
    # the column was judged and replaced, the class stamped, the floor sized from the batch edge
    assert st["snr_source"] == LS.SNR_SOURCE_POISSON
    assert st["instrument_type"] == "tof"
    assert st["noise_edge_batch_cps"] == pytest.approx(0.74)
    assert st["tof_assign_floor_cps"] == pytest.approx(3 * 0.74) == pytest.approx(T.tof_assign_floor(cfg))
    snap = res["pattern_scoring"]
    assert snap["snr_source"] == LS.SNR_SOURCE_POISSON and snap["snr_n"] == len(table)
    # the scorer saw the counting-statistics column on every call, never the server's
    assert seen, "the local scorer was never called"
    for peaks in seen:
        got = pd.to_numeric(peaks["signal_to_noise"], errors="coerce")
        want = LS.poisson_snr(pd.to_numeric(peaks["height"], errors="coerce").to_numpy(dtype=float), snap["snr_edge"])
        np.testing.assert_allclose(got.to_numpy(dtype=float), want, equal_nan=True)
    # the ledger: the bright bromide cluster is read, nothing under the floor is tier Assigned
    led = res["ledger"]
    m0 = led[led["role"] == "M0"]
    assert ((m0["tier"] == "Assigned") & (pd.to_numeric(m0["height"]) < 3 * 0.74)).sum() == 0


def test_an_orbitrap_class_run_leaves_the_column_and_has_no_floor():
    table = _tof_table()
    table["signal_to_noise"] = table["height"] / 20.0                      # a real signal-to-noise
    res = A.run(SID, context="ambient-air", cfg=PassConfig(noise_edge_batch_cps=0.74), peaks=table,
                use_cache=False, scoring="orbi", adducts=["[M-H]-", "[M+Br]-", "[M+NO3]-"],
                reagent_n_relabel=False, log=lambda *a: None)
    st = res["stats"]
    assert st["snr_source"] == LS.SNR_SOURCE_SERVER and st["instrument_type"] == "orbi"
    assert st["tof_assign_floor_cps"] is None


def test_without_a_batch_edge_the_files_own_edge_sizes_the_floor():
    table = _tof_table()
    res = A.run(SID, context="ambient-air", cfg=PassConfig(), peaks=table, use_cache=False, scoring="tof",
                adducts=["[M-H]-", "[M+Br]-", "[M+NO3]-"], reagent_n_relabel=False, log=lambda *a: None)
    st = res["stats"]
    assert st["noise_edge_batch_cps"] is None
    assert st["tof_assign_floor_cps"] == pytest.approx(3 * st["noise_edge_cps"])
