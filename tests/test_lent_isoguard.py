"""The isotope-contradiction guard on a file lent a calibration file's window.

A file no calibration reached is lent a window (the batch's pool, or a calibration json as `campaign` /
`lent`) through `calpool.apply_pool`; that function re-levels the file in the lent window
(`_level_inputs_with(..., cal, pool.source)`) and runs the per-file guard on the pair facts of that
re-level (`pairs_out`). The batch copies the guard record into the file's stats beside the record of the
lent re-tier. These tests pin the three together: the guard binds on a lent file, its record names the
lent source, and the batch keeps it on every file the json was lent to.
"""
from __future__ import annotations

import pandas as pd

from peaky.assignment import calpool as CP
from peaky.assignment import isoguard as IG
from peaky.assignment import tiers as T
from tests.test_calpool import _batch, _inputs_for
from tests.test_isoguard import _ledger as _guard_ledger
from tests.test_tier_calibration_input import _json, _run


def test_apply_pool_guards_a_file_lent_a_window(monkeypatch):
    """A reading whose 13C count contradicts it (6 C read for 10) is Candidate after a lent re-level, with
    the guard's note, and the record says `lent`; the level facts come from the lent window's re-level,
    whose run inputs name the window's own source (the level stage is told `lent`, not `pooled`)."""
    from peaky.assignment import evidence as EV
    seen = []
    real = EV.apply_levels

    def spy(ledger, *a, run_inputs=None, **kw):
        seen.append(((run_inputs.summary.get("per_file") or [{}])[0]).get("tier_calibration"))
        return real(ledger, *a, run_inputs=run_inputs, **kw)

    monkeypatch.setattr(EV, "apply_levels", spy)
    pool = CP.pool_calibration(_batch())
    pool.source = T.CAL_SOURCE_LENT
    led = _guard_ledger(c13_count=6.0, snr13=40.0)
    at = led.copy(deep=True)
    at["tier"] = pd.NA
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1"), audit=[], log=lambda *a: None)
    row = led[led["role"] == "M0"].iloc[0]
    assert row["tier"] == T.TIER_CANDIDATE and f"{IG.NOTE_MARK}: 13C reads " in str(row["tier_reason"])
    assert out["isotope_guard"]["source"] == T.CAL_SOURCE_LENT and out["isotope_guard"]["demoted"] == 1
    # the re-tier stamps the lent source, and the re-level's inputs carry it too
    assert row[T.TIER_CALIBRATION] == T.CAL_SOURCE_LENT
    assert seen == [T.CAL_SOURCE_LENT]


def test_a_batch_lent_a_calibration_json_records_the_guard_on_every_lent_file(tmp_path, monkeypatch):
    """A batch whose pool falls short, given a matching json (`campaign`) or another reagent's json with
    --tier-calibration-lent (`lent`): every file carries the guard record of its lent re-level, named by
    the same source as its `tier_calibration`, and the lent re-tier is recorded under the calibration
    file's state, not the batch pool's."""
    rec = _json(tmp_path / "..", "c1.json")
    summ = _run(tmp_path / "a", monkeypatch, record=rec)
    assert summ["per_file"]
    for p in summ["per_file"]:
        assert p["tier_calibration"] == T.CAL_SOURCE_CAMPAIGN
        assert p["isotope_guard"]["source"] == T.CAL_SOURCE_CAMPAIGN and p["isotope_guard"]["tested"] >= 1
    assert not (summ["tier_calibration"]["pool"] or {}).get("retiered_files")
    rec2 = _json(tmp_path / "..", "c3.json", reagent="NO3")
    summ = _run(tmp_path / "b", monkeypatch, record=rec2, lent=True, iso_rows=14)
    assert summ["per_file"]
    for p in summ["per_file"]:
        assert p["tier_calibration"] == T.CAL_SOURCE_LENT and p["isotope_guard"]["source"] == T.CAL_SOURCE_LENT
    assert not (summ["tier_calibration"]["pool"] or {}).get("retiered_files")


def test_calpool_version_is_newer_than_both_cards():
    """calpool carries the guard on the lent re-level (0.1.1's guard on the pooled re-level and 0.2.0's
    lent source, both): its module version, which the run manifest records, is above both."""
    v = tuple(int(x) for x in CP.__version__.split("."))
    assert v > (0, 2, 0) and v > (0, 1, 1)
