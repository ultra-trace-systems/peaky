"""Live contract tripwire for `peaky publish` -- the only test that exercises the
real import protocol against a real Mascope server.

Everything in tests/test_publish.py is offline: it pins what peaky *sends*, over
stubs. That cannot catch the failure this file exists for -- **the server's
contract moving underneath us**. A new required field, a tightened validator, a
renamed enum, a changed reserved-key list, a different owner-resolution rule:
each of those breaks publishing and none of them is visible from inside this
repository. peaky owns a private implementation of somebody else's wire
protocol, so drift is the standing risk, and until this file existed the only
thing that had ever run the protocol end to end was a person doing it by hand.

Opt-in, because it needs a server and it writes:

    MASCOPE_LIVE=1 MASCOPE_SID=<sample id> python3 tests/test_publish_contract.py

Same gate as the live smoke in test_io_mascope.py. Credentials resolve the usual
way (`MASCOPE_URL` / `MASCOPE_ACCESS_TOKEN`, or ~/.mascope/.env).

**What it costs the server.** One small completed run (a handful of rows) on the
named sample, plus one `importing` run that this file abandons itself. Point it
at a test or demo deployment, not a production one: an import is a write, and
Mascope's retention keeps the newest few completed runs per (sample, engine), so
repeated runs evict older ones on that sample.
"""
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.io import publish as P  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def _get(client, path, **params):
    """A read through the SDK's own http layer, so auth and TLS behave exactly
    as they do for every other call peaky makes."""
    from mascope_sdk._http import http_get

    return http_get(client.url, path, client.access_token, params=params or None).json()


def _delete(client, path):
    import requests

    return requests.delete(
        f"{client.url}/api/{path}",
        headers={
            "Authorization": f"Bearer {client.access_token}",
            "X-Service-Name": "mascope_sdk",
        },
        verify=False,
        timeout=60,
    )


def _ledger(sample_id, peaks):
    """A small ledger over REAL peak ids of the sample.

    Deliberately shaped to hit the translations that have broken before rather
    than to look like a plausible run: an M0 carrying alternatives and peaky's
    pre-rename tier spelling, a second M0 peaky tiered *below* what its fit
    earns (the demotion the whole `engine_tier` column exists for), an
    isotopologue child that must inherit its owner's formula and mechanism, and
    an unexplained residual row.
    """
    ids = [str(p) for p in peaks["peak_id"].head(4)]
    mz = [float(v) for v in peaks["mz"].head(4)]
    height = [float(v) for v in peaks["height"].head(4)]
    base = {
        "sample_item_id": sample_id,
        "area": 1.0,
        "sparsity": 0.0,
        "neutral_formula": np.nan,
        "ion_formula": np.nan,
        "adduct": np.nan,
        "ion_score": np.nan,
        "iso_match_score": np.nan,
        "ppm_error": np.nan,
        "parent_peak_id": np.nan,
        "iso_label": np.nan,
        "tier": np.nan,
        "tier_reason": np.nan,
        "confidence": np.nan,
        "synthetic": False,
        "alternatives": np.nan,
        "target_compound_id": np.nan,
    }
    rows = [
        {**base, "peak_id": ids[0], "mz": mz[0], "height": height[0], "role": "M0",
         "neutral_formula": "C6H12O6", "ion_formula": "C6H13O6+", "adduct": "[M+H]+",
         "ion_score": 0.95, "ppm_error": 0.4, "tier": "Identified",
         "confidence": "High", "tier_reason": "unique in the calibrated window",
         "alternatives": '[{"formula": "C5H8O4", "adduct": "[M+H]+", '
                         '"ion_score": 0.31, "ppm": 1.2}]'},
        {**base, "peak_id": ids[1], "mz": mz[1], "height": height[1], "role": "M0",
         "neutral_formula": "C7H14O6", "ion_formula": "C7H15O6+", "adduct": "[M+H]+",
         "ion_score": 0.93, "tier": "Candidate",
         "tier_reason": "crowded mass; no corroboration"},
        {**base, "peak_id": ids[2], "mz": mz[2], "height": height[2],
         "role": "iso_child", "parent_peak_id": ids[0], "iso_match_score": 0.9,
         "iso_label": "13C"},
        {**base, "peak_id": ids[3], "mz": mz[3], "height": height[3],
         "role": "unexplained"},
    ]
    return pd.DataFrame(rows), ids


if os.environ.get("MASCOPE_LIVE") != "1" or not os.environ.get("MASCOPE_SID"):
    print("(publish contract tripwire skipped; set MASCOPE_LIVE=1 and "
          "MASCOPE_SID=<sample id> to run)")
else:
    import urllib3

    urllib3.disable_warnings()
    from peaky.io import io_mascope as IO

    SID = os.environ["MASCOPE_SID"]
    client = IO.connect()
    peaks = IO.fetch_peaks(client, SID, use_cache=False)
    check("the sample has peaks to publish against", peaks is not None and len(peaks) >= 4,
          None if peaks is None else len(peaks))

    led, ids = _ledger(SID, peaks)
    intensity, _kind = P.intensity_column_for(client.samples.get(SID))
    mech = IO.resolve_mechanism_ids(client, [IO.ADDUCT_TO_MECH["[M+H]+"]])
    mech_ids = {"[M+H]+": mech.get(IO.ADDUCT_TO_MECH["[M+H]+"])} if mech else {}
    rows, summary = P.build_rows(led, intensity_column=intensity,
                                 bands=P.DEFAULT_TIER_BANDS, mechanism_ids=mech_ids)
    check("the ledger translated to four rows", len(rows) == 4, len(rows))
    check("no tier is sent -- the server derives it",
          all("tier" not in r for r in rows))

    # ---- the assembly protocol, on a run this file cleans up -----------------
    # Staged, never finalized: it proves the offset bookkeeping without leaving a
    # completed run behind, and the abandon endpoint is what releases the sample.
    key = f"contract-{uuid.uuid4().hex[:12]}"
    envelope = {"engine": P.ENGINE, "engine_version": "contract-test",
                "tier_bands": P.DEFAULT_TIER_BANDS,
                "calibration": P.build_calibration(None), "config": {"probe": True}}
    state = P._post(client, SID, {**envelope, "rows": rows[:2],
                                  "chunk": {"import_id": key, "index": 0,
                                            "complete": False}}, (15, 120))
    run_id = state["peak_assignment_run_id"]
    check("a create returns the run and its row count",
          state["rows"] == 2 and state["run_status"] == "importing", state)
    check("the deployment states its own row cap",
          isinstance(state.get("max_rows_per_request"), int))

    replay = P._post(client, SID, {"engine": P.ENGINE, "engine_version": "contract-test",
                                   "rows": rows[:2],
                                   "chunk": {"import_id": key, "run_id": run_id,
                                             "index": 0, "complete": False}}, (15, 120))
    # The SDK retries POSTs with no way to opt out, so a chunk the server applied
    # but never acknowledged WILL be re-sent. If this stops being a no-op, every
    # dense publish starts duplicating rows onto the unique constraint.
    check("re-sending an applied chunk is an idempotent no-op",
          replay["rows"] == 2, replay)

    try:
        P._post(client, SID, {"engine": P.ENGINE, "engine_version": "contract-test",
                              "rows": rows[2:], "chunk": {"import_id": key,
                                                          "run_id": run_id,
                                                          "index": 99,
                                                          "complete": False}}, (15, 120))
        check("a chunk at the wrong offset is refused", False, "accepted")
    except Exception as exc:  # noqa: BLE001 - any refusal is the contract holding
        check("a chunk at the wrong offset is refused", "409" in str(exc) or
              "conflict" in str(exc).lower(), str(exc)[:120])

    abandoned = _delete(client, f"peak-assignments/sample/{SID}/runs/{run_id}")
    check("an abandoned assembly releases the sample", abandoned.status_code == 200,
          abandoned.status_code)

    # ---- the round trip ------------------------------------------------------
    state = P.publish(client, SID, rows, tier_bands=P.DEFAULT_TIER_BANDS,
                      calibration=P.build_calibration(None),
                      config={"contract_test": True}, version="contract-test",
                      log=lambda _m: None)
    check("the run completes", state["run_status"] == "completed", state)
    published = state["peak_assignment_run_id"]

    runs = _get(client, f"peak-assignments/sample/{SID}/runs")["data"]
    run = next(r for r in runs if r["peak_assignment_run_id"] == published)
    check("the run is attributed to peaky", run["engine"] == "peaky", run.get("engine"))
    check("the declared bands are stored on the run",
          run.get("tier_bands") == P.DEFAULT_TIER_BANDS, run.get("tier_bands"))
    check("the calibration disclosure is stored", bool(run.get("calibration")))

    served = _get(client, f"peak-assignments/sample/{SID}",
                  peak_assignment_run_id=published, limit=50)["data"]
    by_peak = {r["sample_peak_id"]: r for r in served}
    check("every row round-trips", len(served) == 4, len(served))

    m0, demoted, iso, residual = (by_peak[i] for i in ids)

    # The server derived these from the fit and the formula; peaky sent no tier.
    check("the server tiered the strong row 'assigned'", m0["tier"] == "assigned",
          m0["tier"])
    check("a row with no formula is 'unassigned'", residual["tier"] == "unassigned",
          residual["tier"])
    check("the server's evidence is served beside the tier",
          m0.get("evidence") is not None)

    # peaky's own verdict, and the disagreement the column exists to show.
    check("peaky's pre-rename spelling normalises to 'assigned'",
          m0["engine_tier"] == "assigned", m0["engine_tier"])
    check("a demotion peaky made survives as a disagreement",
          demoted["engine_tier"] == "candidate" and demoted["tier"] != "candidate",
          (demoted["tier"], demoted["engine_tier"]))
    check("a row peaky did not tier carries no engine tier",
          residual["engine_tier"] is None)

    # The two fields the app needs beyond display: the mechanism is part of a
    # verification's identity and the fit view refuses to open without it.
    if mech_ids.get("[M+H]+"):
        check("the M0 carries a resolved ionization mechanism",
              m0["ionization_mechanism_id"] == mech_ids["[M+H]+"])
        check("the isotopologue inherited its owner's mechanism",
              iso["ionization_mechanism_id"] == mech_ids["[M+H]+"])
    check("an M0 is labelled M0, not left blank", m0["isotope_label"] == "M0",
          m0["isotope_label"])
    check("the isotopologue inherited its owner's formula",
          iso["assigned_formula"] == "C6H12O6", iso["assigned_formula"])
    check("the owner link resolved to a minted assignment id",
          iso.get("owner_peak_assignment_id") == m0["peak_assignment_id"],
          iso.get("owner_peak_assignment_id"))

    # Provenance and alternatives live on the detail record, not the slim row.
    detail = _get(client,
                  f"peak-assignments/sample/{SID}/assignment/{m0['peak_assignment_id']}"
                  )["data"][0]
    prov = detail.get("provenance") or {}
    check("the server wrote its own evidence", prov.get("evidence") is not None)
    check("the server wrote its own plausibility", prov.get("plausibility") is not None)
    check("no reserved key is stored",
          not {"p_correct", "calibration", "corroboration"} & set(prov))
    check("peaky's own reasoning survives under its own name",
          (prov.get("engine_provenance") or {}).get("engine") == "peaky")
    alts = detail.get("alternatives") or []
    check("an alternative names its formula where the app reads it",
          bool(alts) and alts[0].get("assigned_formula") == "C5H8O4", alts[:1])

    print(f"\n(left one completed run {published} on sample {SID}; "
          "retention prunes it)")


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
