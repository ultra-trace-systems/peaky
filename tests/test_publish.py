"""Offline tests for `peaky publish` (io/publish.py): the ledger -> Mascope
import translation, the local pre-validation, request chunking, and the CLI
wiring. Nothing here touches the network.

The rules these pin come from Mascope's run-import contract
(docs/dev/sdk_peak_assignment.md 8.2). Where a rule mirrors a server function,
the test states the server's behaviour it has to agree with -- that agreement is
the whole point of the mirror, and it is invisible from inside this repo.

Run: python3 tests/test_publish.py  (or pytest tests/test_publish.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import cli  # noqa: E402
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


BANDS = {"assigned": 0.75, "candidate": 0.45}


def _ledger(rows):
    """A ledger frame with peaky's columns, defaulted the way the real one is."""
    base = {
        "sample_item_id": "SAMPLE0000000001",
        "peak_id": None, "mz": 100.0, "area": 10.0, "height": 1000.0,
        "role": "unexplained", "neutral_formula": np.nan, "ion_formula": np.nan,
        "adduct": np.nan, "ion_score": np.nan, "iso_match_score": np.nan,
        "ppm_error": np.nan, "parent_peak_id": np.nan, "iso_label": np.nan,
        "tier": np.nan, "tier_reason": np.nan, "confidence": np.nan,
        "synthetic": False, "alternatives": np.nan, "target_compound_id": np.nan,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# ---- evidence mirrors the server's own evidence_for ---------------------------
# Server: no formula -> the bare fit, unrounded; a formula -> round(fit*plaus, 4).
check("no fit score has no evidence", P.evidence_for(None, "C6H6") is None)
check("no formula leaves the fit unrounded",
      P.evidence_for(0.7499873329897717, None) == 0.7499873329897717)
check("a formula-bearing row is rounded to 4dp",
      P.evidence_for(0.7499873329897717, "C8H15NO") == 0.75)
check("NaN fit has no evidence", P.evidence_for(float("nan"), "C6H6") is None)

# The regression this file exists for. A fit of 0.7499873 is 0.75 to the server,
# which is 'assigned' under a 0.75 band; skipping the rounding made it
# 'candidate' here and the server refused the row -- and with it the whole
# import, since one incoherent row is a 422.
check("a band-edge fit tiers as the server tiers it",
      P.derive_tier(0.7499873329897717, "C8H15NO", "M0", BANDS) == "assigned")

# ---- tier derivation ---------------------------------------------------------
check("evidence at the band is 'assigned'", P.derive_tier(0.80, "C6H6", "M0", BANDS)
      == "assigned")
check("evidence between bands is 'candidate'",
      P.derive_tier(0.50, "C6H6", "M0", BANDS) == "candidate")
check("evidence below both bands is 'below_assignability'",
      P.derive_tier(0.10, "C6H6", "M0", BANDS) == "below_assignability")
# The server admits exactly two tiers for a null fit and this picks between them
# the way the in-app engine writes them.
check("no fit and no formula is 'unassigned'",
      P.derive_tier(None, None, "unassigned", BANDS) == "unassigned")
check("no fit but a committed formula is 'below_assignability'",
      P.derive_tier(None, "C6H6", "M0", BANDS) == "below_assignability")

# ---- bands -------------------------------------------------------------------
check("default bands are the in-app engine's", P.normalize_bands(None) == BANDS)
check("the legacy 'identified' band key is accepted",
      P.normalize_bands({"identified": 0.8, "candidate": 0.5})["assigned"] == 0.8)
try:
    P.normalize_bands({"assigned": 0.3, "candidate": 0.6})
    check("inverted bands are refused", False, "no error raised")
except P.PublishError:
    check("inverted bands are refused", True)
try:
    P.normalize_bands({"candidate": 0.4})
    check("a missing band is refused", False, "no error raised")
except P.PublishError:
    check("a missing band is refused", True)

# ---- intensity: heights for Orbitrap, areas for TOF --------------------------
check("orbitrap publishes heights",
      P.intensity_column_for({"filename": "orbi-1_2026.raw"})[0] == "height")
check("tof publishes areas",
      P.intensity_column_for({"filename": "tof-3_2026.h5"})[0] == "area")
check("the instrument field is read too",
      P.intensity_column_for({"instrument": "APi-TOF"})[0] == "area")
try:
    P.intensity_column_for({"filename": "mystery_2026.h5"})
    check("an unclassifiable sample refuses to guess", False, "no error raised")
except P.PublishError as e:
    # Guessing here is the one failure nothing downstream can detect.
    check("an unclassifiable sample refuses to guess", "--intensity" in str(e))

# ---- translation -------------------------------------------------------------
led = _ledger([
    {"peak_id": "P0000000000000000M0", "role": "M0", "neutral_formula": "C10H16O2",
     "ion_formula": "C10H17O2+", "adduct": "[M+H]+", "ion_score": 0.99,
     "ppm_error": 0.12, "tier": "Identified", "confidence": "High", "height": 5.0},
    {"peak_id": "P000000000000000iso", "role": "iso_child",
     "parent_peak_id": "P0000000000000000M0", "iso_match_score": 0.98,
     "iso_label": "13C", "height": 6.0},
    {"peak_id": "P0000000000000unexp", "role": "unexplained", "height": 7.0},
    {"peak_id": "P00000000000000arti", "role": "artifact", "height": 8.0},
    {"peak_id": "P0000000000000synth", "role": "M0", "synthetic": True,
     "neutral_formula": "C2H6", "ion_score": 0.9, "height": 9.0},
])
rows, summary = P.build_rows(led, intensity_column="height", bands=BANDS)
by_id = {r["sample_peak_id"]: r for r in rows}

check("synthetic rows are excluded", "P0000000000000synth" not in by_id
      and summary["dropped_synthetic"] == 1)
check("four publishable rows", len(rows) == 4, len(rows))
check("peaky's 'unexplained' becomes Mascope's 'unassigned'",
      by_id["P0000000000000unexp"]["role"] == "unassigned")
check("the artifact role survives", by_id["P00000000000000arti"]["role"] == "artifact")
check("the intensity comes from the named column",
      by_id["P0000000000000unexp"]["sample_peak_intensity"] == 7.0)

m0 = by_id["P0000000000000000M0"]
check("an M0 is scored by its ion score", m0["fit_score"] == 0.99)
# The row carries NO `tier`: it is a pure function of the fit, the formula and
# the declared bands, all of which the server holds, so publishing one would be
# a second copy of the server's own rule - and a copy that disagrees at a band
# edge refuses the whole import.
check("no tier is published", "tier" not in m0)
check("peaky's own verdict is published instead", m0["engine_tier"] == "assigned")
check("an M0 with no target is untargeted", m0["source"] == "untargeted")

iso = by_id["P000000000000000iso"]
check("an iso_child is scored by its isotope match", iso["fit_score"] == 0.98)
# Mascope models the family by having the child carry the M0's formula; peaky's
# child row holds none of its own.
check("an iso_child inherits its owner's formula",
      iso["assigned_formula"] == "C10H16O2" and summary["inherited_formulas"] == 1)
check("an iso_child references its owner by peak id",
      iso["owner_sample_peak_id"] == "P0000000000000000M0")
check("only an iso_child carries an owner",
      m0["owner_sample_peak_id"] is None)

check("a peak with no formula has no fit score",
      by_id["P0000000000000unexp"]["fit_score"] is None)
# peaky tiers committed M0 assignments and nothing else, so silence is the
# honest answer on the rest - and the server reads absence as "no verdict",
# never as agreement.
check("a row peaky did not tier carries no engine tier",
      by_id["P0000000000000unexp"]["engine_tier"] is None
      and by_id["P00000000000000arti"]["engine_tier"] is None
      and by_id["P000000000000000iso"]["engine_tier"] is None)

# peaky's own verdict is kept, under a name of its own, beside the published one.
prov = m0["provenance"]["engine_provenance"]
check("peaky's own spelling is preserved in the blob", prov["peaky_tier"] == "Identified")
check("the mapped verdict is recorded beside it", prov["engine_tier"] == "assigned")
check("peaky's confidence is preserved", prov["confidence"] == "High")
check("the provenance names its engine", prov["engine"] == "peaky")

# ---- the engine's own tier ---------------------------------------------------
# The point of the feature: peaky tiers mechanically (window uniqueness,
# corroboration, degeneracy) and Mascope bands on evidence, so the two disagree
# on real rows. Publishing peaky's verdict in `engine_tier` keeps that visible;
# publishing it as `tier` would be refused outright.
_ET = [
    ("Identified", "assigned"),   # peaky's pre-rename spelling
    ("Assigned", "assigned"),     # peaky's current spelling
    ("Candidate", "candidate"),
    ("candidate", "candidate"),   # already lower-case
    ("Below assignability", "below_assignability"),
]
for _peaky_tier, _expected in _ET:
    _rows, _ = P.build_rows(
        _ledger([{"peak_id": "P0000000000000000M0", "role": "M0",
                  "neutral_formula": "C6H6", "ion_score": 0.9,
                  "tier": _peaky_tier}]),
        intensity_column="height", bands=BANDS)
    check(f"peaky tier {_peaky_tier!r} maps to {_expected!r}",
          _rows[0]["engine_tier"] == _expected, _rows[0]["engine_tier"])

# A spelling neither vocabulary knows is dropped rather than guessed at: the
# server's vocabulary is closed and would refuse the whole import over it.
_rows, _ = P.build_rows(
    _ledger([{"peak_id": "P0000000000000000M0", "role": "M0",
              "neutral_formula": "C6H6", "ion_score": 0.9, "tier": "Splendid"}]),
    intensity_column="height", bands=BANDS)
check("an unknown peaky tier publishes as no verdict",
      _rows[0]["engine_tier"] is None)

# The disagreement is counted so `--dry-run` can say how much there is to look
# at. A high fit that peaky nonetheless called Candidate is the shape that
# matters: Mascope's banding puts it at 'assigned', peaky demoted it for
# reasons the bands cannot express, and BOTH are now on the row.
_rows, _summary = P.build_rows(
    _ledger([
        {"peak_id": "P0000000000000000M0", "role": "M0", "neutral_formula": "C6H6",
         "ion_score": 0.99, "tier": "Candidate"},
        {"peak_id": "P0000000000000000M1", "role": "M0", "neutral_formula": "C6H6",
         "ion_score": 0.99, "tier": "Assigned"},
    ]),
    intensity_column="height", bands=BANDS)
check("a demotion peaky made is published as such",
      _rows[0]["engine_tier"] == "candidate")
check("the disagreement is counted", _summary["engine_tier_disagreements"] == 1)
check("an agreeing row is not counted as a disagreement",
      _rows[1]["engine_tier"] == "assigned")
check("the predicted tiers are reported separately from peaky's",
      _summary["by_predicted_tier"] == {"assigned": 2}
      and _summary["by_engine_tier"] == {"candidate": 1, "assigned": 1})


# ---- rendering in Mascope's own shapes ---------------------------------------
# Three fields where publishing peaky's own spelling produced a blank rather
# than an error, because nothing validates them: the ledger's isotope column,
# the isotope formula, and the inspector's close-alternatives list.
check("an M0 row is labelled M0, not left blank", m0["isotope_label"] == "M0")
check("an M0 row's isotope formula is its ion formula",
      m0["isotope_formula"] == "C10H17O2+")
check("an iso_child keeps peaky's own isotope label",
      iso["isotope_label"] == "13C")
check("an iso_child gets no isotope formula peaky did not compute",
      iso["isotope_formula"] is None)
check("a row with no role-derived label has none",
      by_id["P0000000000000unexp"]["isotope_label"] is None)

# The inspector reads `assigned_formula` off each alternative; peaky writes
# `formula`, so published verbatim every runner-up renders as "?".
_alts = P._alternatives(json.dumps([
    {"formula": "C8H18O2", "adduct": "[M+Na]+", "ion_score": 0.4, "ppm": 1.2},
    {"formula": "C5H13NO4", "adduct": "[M+NH4]+", "ion_score": 0.0, "ppm": None},
]))
check("an alternative carries the formula where the app looks for it",
      [a["assigned_formula"] for a in _alts] == ["C8H18O2", "C5H13NO4"])
check("an alternative says which search produced it",
      all(a["source"] == "untargeted" for a in _alts))
check("peaky's own numbers ride along under their own name",
      _alts[0]["engine_provenance"] == {"adduct": "[M+Na]+", "ion_score": 0.4, "ppm": 1.2})
check("a null field is dropped rather than published as null",
      "ppm" not in _alts[1]["engine_provenance"])
# Plausibility is the server's reading of the chemistry; peaky never weighed
# these, and the inspector shows a dash for an entry that has none.
check("no plausibility is invented", all("plausibility" not in a for a in _alts))
check("an already-Mascope-shaped entry passes through",
      P._alternatives([{"assigned_formula": "C6H6"}])[0]["assigned_formula"] == "C6H6")
check("an entry with no formula is dropped", P._alternatives([{"ppm": 1.0}]) is None)
check("an empty list publishes as nothing", P._alternatives([]) is None)
check("a non-list publishes as nothing", P._alternatives("not json") is None)


# ---- reserved provenance keys ------------------------------------------------
# The server strips these because the app renders what it derives from them as
# its own calibrated judgement; sending them is not an error, just not honoured.
led_res = _ledger([{"peak_id": "P0000000000000000M0", "role": "M0",
                    "neutral_formula": "C6H6", "ion_score": 0.9,
                    "provenance": json.dumps({"p_correct": 0.99, "corroboration": {},
                                              "calibration": {}, "keep_me": 1})}])
rows_res, summary_res = P.build_rows(led_res, intensity_column="height", bands=BANDS)
sent = rows_res[0]["provenance"]
check("reserved provenance keys are dropped",
      not {"p_correct", "corroboration", "calibration"} & set(sent))
check("the drop is reported", summary_res["reserved_provenance_dropped"]
      == ["calibration", "corroboration", "p_correct"])
check("unreserved provenance rides along", sent.get("keep_me") == 1)

# ---- non-finite floats -------------------------------------------------------
# The server sets allow_inf_nan=False on every float a row carries, and a NaN
# inside an opaque blob breaks the READ of the whole run once stored.
check("NaN becomes null", P._sanitize({"a": float("nan")}) == {"a": None})
check("inf becomes null", P._sanitize([float("inf")]) == [None])
check("numpy floats are unwrapped", P._sanitize(np.float64(1.5)) == 1.5)
check("nested NaN is reached", P._sanitize({"a": [{"b": float("nan")}]})
      == {"a": [{"b": None}]})
check("a NaN ppm_error is published as null",
      by_id["P0000000000000unexp"]["mz_error_ppm"] is None)

# ---- local pre-validation ----------------------------------------------------
check("a clean payload has no problems", P.validate_rows(rows) == [])
check("an empty payload is refused", len(P.validate_rows([])) == 1)

dupes = [dict(rows[0]), dict(rows[0])]
check("a repeated peak is caught", "more than one row" in " ".join(P.validate_rows(dupes)))

owner_on_m0 = [{**rows[0], "owner_sample_peak_id": "P0000000000000unexp"}]
check("an owner on a non-iso_child row is caught",
      "only an iso_child" in " ".join(P.validate_rows(owner_on_m0)))

missing_owner = [{**iso, "owner_sample_peak_id": "NOPE"}]
check("an owner outside the import is caught",
      "not in this import" in " ".join(P.validate_rows(missing_owner)))

# The server checks this at finalize, because an owner may arrive in another
# chunk; A-owns-B with B-owns-A resolves into a shape no in-app run can produce.
chain = [
    {**iso, "sample_peak_id": "A", "owner_sample_peak_id": "B"},
    {**iso, "sample_peak_id": "B", "owner_sample_peak_id": "C"},
    {**iso, "sample_peak_id": "C", "owner_sample_peak_id": None, "role": "M0"},
]
check("an owner that is itself an iso_child is caught",
      "one level deep" in " ".join(P.validate_rows(chain)))

# ---- chunking ----------------------------------------------------------------
many = [dict(rows[0], sample_peak_id=f"P{i:019d}") for i in range(2500)]
chunks = P.chunk_rows(many, max_rows=1000)
check("chunks respect the row cap", [len(c) for c in chunks] == [1000, 1000, 500])
check("chunking loses no rows", sum(len(c) for c in chunks) == 2500)
tight = P.chunk_rows(many, max_rows=1000, max_bytes=20_000)
check("chunks respect the byte budget too", all(len(c) < 1000 for c in tight)
      and sum(len(c) for c in tight) == 2500)
check("one oversized row still goes out alone",
      len(P.chunk_rows([many[0]], max_bytes=1)) == 1)

# ---- run-level fields --------------------------------------------------------
cal = P.build_calibration({"prescan": {"ppm_offset": -1.9}})
check("the calibration discloses it is not Mascope-verified",
      cal["provisional"] is True and cal["verified_against_mascope"] is False)
check("a known prescan offset is disclosed", cal["ppm_offset"] == -1.9)

big = {"huge": ["x" * 1000 for _ in range(200)], "small": 1}
capped = P.build_config(big, log=lambda _m: None)
check("an oversized config is trimmed under the cap",
      len(json.dumps(capped).encode()) <= P.MAX_JSON_BYTES)
check("trimming is recorded", capped.get("_truncated") == ["huge"])
check("the small keys survive trimming", capped.get("small") == 1)

# ---- CLI wiring --------------------------------------------------------------
PARSER = cli.build_parser()
a = PARSER.parse_args(["publish", "led.csv"])
check("parse `publish`", a.func is cli.cmd_publish and a.ledger_csv == "led.csv")
check("publish defaults to auto intensity", a.intensity == "auto")
check("publish defaults to the in-app bands",
      a.assigned_band == 0.75 and a.candidate_band == 0.45)
check("publish does not send by default", a.dry_run is False)
a = PARSER.parse_args(["publish", "led.csv", "--intensity", "area", "--dry-run",
                       "--sample-id", "S1", "--import-id", "abc"])
check("parse `publish` overrides", a.intensity == "area" and a.dry_run
      and a.sample_id == "S1" and a.import_id == "abc")

# ---- a merged ledger is refused rather than half-published -------------------
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    merged = _ledger([
        {"peak_id": "P0000000000000000A0", "role": "M0", "neutral_formula": "C6H6",
         "ion_score": 0.9, "sample_item_id": "SAMPLE0000000001"},
        {"peak_id": "P0000000000000000B0", "role": "M0", "neutral_formula": "C7H8",
         "ion_score": 0.9, "sample_item_id": "SAMPLE0000000002"},
    ])
    path = Path(d) / "merged_ledger.csv"
    merged.to_csv(path, index=False)
    args = PARSER.parse_args(["publish", str(path), "--intensity", "height",
                              "--dry-run"])
    try:
        cli.cmd_publish(args)
        check("a multi-sample ledger is refused", False, "no SystemExit")
    except SystemExit as e:
        # An import targets exactly one sample; publishing the first one silently
        # would withdraw the others' peaks from the batch overview.
        check("a multi-sample ledger is refused", "sample_item_id" in str(e))


# ---- the upload protocol -----------------------------------------------------
class _StubServer:
    """Records requests and answers the way the import endpoint does.

    Only the four state fields a client is allowed to steer by: the run id, the
    rows staged so far (which is the next chunk's offset), the effective row
    cap, and the run status.
    """

    def __init__(self, cap=1000, run_id="RUN1"):
        self.cap, self.run_id = cap, run_id
        self.staged = 0
        self.requests = []

    def __call__(self, _client, _sample_id, body, _timeout):
        self.requests.append(body)
        chunk = body["chunk"]
        if chunk["index"] != self.staged:            # gap or rewind
            raise AssertionError(f"offset {chunk['index']} != staged {self.staged}")
        self.staged += len(body["rows"])
        return {
            "peak_assignment_run_id": self.run_id,
            "rows": self.staged,
            "max_rows_per_request": self.cap,
            "run_status": "completed" if chunk["complete"] else "importing",
        }


def _publish_with(server, rows, **kw):
    original = P._post
    P._post = server
    try:
        return P.publish(object(), "SAMPLE0000000001", rows, tier_bands=BANDS,
                         calibration={"provisional": True}, config={"engine": "peaky"},
                         version="9.9.9", log=lambda _m: None, **kw)
    finally:
        P._post = original


srv = _StubServer()
state = _publish_with(srv, rows)
check("a slim ledger publishes in one request", len(srv.requests) == 1)
check("the single request finalizes", srv.requests[0]["chunk"]["complete"] is True)
check("the run completes", state["run_status"] == "completed")
check("the create carries the run-level fields",
      {"tier_bands", "calibration", "config"} <= set(srv.requests[0]))
check("the create carries no run id", "run_id" not in srv.requests[0]["chunk"])
check("an import_id is always sent", srv.requests[0]["chunk"]["import_id"])
check("the engine is named", srv.requests[0]["engine"] == "peaky")
check("the engine version is stamped", srv.requests[0]["engine_version"] == "9.9.9")

srv = _StubServer()
_publish_with(srv, many, max_rows=1000)
check("a dense ledger assembles across requests", len(srv.requests) == 3)
# The index is a row OFFSET, not a sequence counter -- that is what lets the
# server recognise a retried chunk as a replay instead of duplicating rows.
check("each chunk's index is a row offset",
      [r["chunk"]["index"] for r in srv.requests] == [0, 1000, 2000])
check("follow-ups carry the run id",
      all(r["chunk"]["run_id"] == "RUN1" for r in srv.requests[1:]))
check("only the last chunk finalizes",
      [r["chunk"]["complete"] for r in srv.requests] == [False, False, True])
check("run-level fields are not resent",
      all("config" not in r for r in srv.requests[1:]))
check("one import_id spans the whole assembly",
      len({r["chunk"]["import_id"] for r in srv.requests}) == 1)

# The deployment's cap is the server's to lower; only the first request has to
# size itself blind.
srv = _StubServer(cap=400)
_publish_with(srv, many[:1200], max_rows=1000)
check("a lowered row cap is adopted for the remainder",
      all(len(r["rows"]) <= 400 for r in srv.requests[1:]),
      [len(r["rows"]) for r in srv.requests])
check("re-chunking still covers every row",
      sum(len(r["rows"]) for r in srv.requests) == 1200)

srv = _StubServer(run_id="RUN2")
resumed = _publish_with(srv, rows, import_id="fixed-key")
check("a supplied import_id is used verbatim",
      srv.requests[0]["chunk"]["import_id"] == "fixed-key")
check("the run id comes back to the caller",
      resumed["peak_assignment_run_id"] == "RUN2")


class _NeverFinalizes(_StubServer):
    def __call__(self, *a):
        out = super().__call__(*a)
        out["run_status"] = "importing"
        return out


try:
    _publish_with(_NeverFinalizes(), rows)
    check("an unfinished import is reported", False, "no error raised")
except P.PublishError as e:
    # A non-terminal run refuses every later import AND in-app assign for the
    # sample, so the way out has to be in the message.
    check("an unfinished import is reported", "--import-id" in str(e))


# ---- refusals say what to do next --------------------------------------------
# Both are 409s and they need opposite advice, which is why they are told apart
# on the server's wording rather than on the status alone. Confirmed against a
# live server: re-sending a finished import_id returns the run it already made.
replayed = P._upload_failure_message(
    RuntimeError("[HTTP 409] Peak assignment run 'R1' is already completed and "
                 "cannot take more rows."), None, 0)
check("a replayed import_id says nothing was duplicated",
      "Nothing was duplicated" in replayed and "without --import-id" in replayed)
check("a replayed import_id does not tell you to delete a run",
      "delete that run" not in replayed)

in_flight = P._upload_failure_message(
    RuntimeError("[HTTP 409] a run is already in progress for this sample"), None, 0)
check("an in-flight run says how to release the sample",
      "delete that run" in in_flight)

body_too_big = P._upload_failure_message(RuntimeError("[HTTP 413] too large"), "R1", 500)
check("a 413 names the proxy limit and the run it stalled",
      "client_max_body_size" in body_too_big and "run R1, 500 row(s) staged" in body_too_big)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
