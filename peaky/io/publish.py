"""Publish a peaky ledger into Mascope's peak-assignment run ledger.

peaky reads peaks out of Mascope and writes its ledgers as local files. This
module closes that loop: it translates a finished ledger into Mascope's run
import contract and uploads it, so a peaky run lands in the same store the
in-app engine writes to -- run selector, peak inspector, batch overview,
verification loop -- instead of one researcher's filesystem.

    Mascope peaks --(SDK read)--> peaky ledger --(publish)--> Mascope run ledger

The endpoint is ``POST /api/peak-assignments/sample/{id}/runs/import``. It
accepts a complete, externally computed run for one sample, assembled over one
or more requests. This module owns the whole client side of that protocol: the
column translation, the local pre-validation that turns a server 422 into a
message naming the offending rows before anything is uploaded, and the chunked
upload with its offset bookkeeping.

Three things about the contract drive most of the code here, and each is a place
a naive field-for-field mapping is silently wrong:

1. **The tier is not peaky's tier.** peaky tiers mechanically -- window
   uniqueness, corroboration, degeneracy, oxygen count (``assignment/tiers.py``)
   -- while Mascope requires a row's tier to be the one its *evidence*
   (``fit x chemical plausibility``) falls in under the run's declared
   ``tier_bands``, and recomputes that product server-side to check. The two
   agree only by accident, and a single disagreeing row is a 422 that rejects
   the whole import. So publish DERIVES the published tier from the declared
   bands and carries peaky's own verdict through as inspector detail under
   ``provenance.engine_provenance``. Nothing is lost; the ledger column just
   means what Mascope's column means.

2. **Vocabularies differ where the docs say they do not.** peaky's
   ``unexplained`` role is Mascope's ``unassigned``; peaky's ``Assigned`` /
   ``Candidate`` tiers are capitalized where Mascope's are not (only the
   lowercase legacy ``identified`` is aliased server-side).

3. **Not every ledger row is publishable.** Composite de-blending invents
   synthetic sub-peaks that exist in no Mascope peak file, and every
   ``sample_peak_id`` must exist in the sample or the import is refused.

Reserved provenance keys (``p_correct``, ``calibration``, ``corroboration``) are
stripped by the server at the top level of an imported blob, because the app
renders them as its own calibrated judgement. peaky's numbers therefore go under
``provenance.engine_provenance``, which the server stores verbatim and never
interprets -- that is what makes an in-app run and a peaky run comparable.

Only :func:`publish` touches the network; everything above it is pure and is
tested offline in ``tests/test_publish.py``.
"""
from __future__ import annotations

import json
import math
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# the server's contract, mirrored
# --------------------------------------------------------------------------- #
#: Engine name stamped on every run this module publishes. Mascope reserves its
#: own identities ('mascope', 'mascope-copy') and refuses them from a client;
#: retention budgets runs per (sample, engine), so this names the ENGINE and
#: never the build -- the version goes in `engine_version`.
ENGINE = "peaky"

#: Rows one request may carry (the server's MAX_IMPORT_ROWS_PER_REQUEST). The
#: create response echoes the deployment's effective cap, which is authoritative
#: and may be lower; this is only what the first request has to guess with.
MAX_ROWS_PER_REQUEST = 1000

#: Byte ceiling on one request body (the server's MAX_IMPORT_BODY_BYTES). Rows
#: differ in size by an order of magnitude depending on whether they carry
#: `alternatives`/`provenance`, so chunks are sized by serialized bytes as well
#: as by row count -- the row cap is the ceiling, not the target.
MAX_BODY_BYTES = 5 * 1024 * 1024

#: Byte ceiling on `config` and on `calibration` (the server's
#: MAX_IMPORT_JSON_BYTES). Both are re-served on every run listing, which is a
#: hot path, so both are capped.
MAX_JSON_BYTES = 64 * 1024

#: Mascope's tier vocabulary, most confident first.
TIER_ASSIGNED = "assigned"
TIER_CANDIDATE = "candidate"
TIER_BELOW_ASSIGNABILITY = "below_assignability"
TIER_UNASSIGNED = "unassigned"

#: Mascope's role vocabulary for an imported row.
IMPORT_ROLES = ("M0", "iso_child", "reagent", "artifact", "unassigned")

#: peaky role -> Mascope role. Only one name actually moves: peaky calls the
#: residual 'unexplained', Mascope calls it 'unassigned'. The rest are shared.
ROLE_MAP = {
    "unexplained": "unassigned",
    "M0": "M0",
    "iso_child": "iso_child",
    "reagent": "reagent",
    "artifact": "artifact",
}

#: peaky's own report tier -> Mascope's vocabulary, for the `engine_tier`
#: column. peaky capitalizes and has used two spellings for the top tier
#: (`Identified` before it was `Assigned`); Mascope's vocabulary is lower-case
#: and aliases only the lower-case legacy word, so both are mapped here rather
#: than left to the server. A row peaky did not tier maps to nothing and
#: publishes as null: peaky tiers committed M0 assignments only, and silence is
#: not a verdict.
ENGINE_TIER_MAP = {
    "assigned": TIER_ASSIGNED,
    "identified": TIER_ASSIGNED,
    "candidate": TIER_CANDIDATE,
    "below assignability": TIER_BELOW_ASSIGNABILITY,
    "below_assignability": TIER_BELOW_ASSIGNABILITY,
    "unassigned": TIER_UNASSIGNED,
}

#: The in-app engine's own bands, on the evidence scale. Publishing under these
#: is what makes a peaky tier mean the same thing as an in-app tier on the same
#: sample, which is the point of putting both runs in one store. Overridable,
#: because they are run config rather than an engine constant.
DEFAULT_TIER_BANDS = {"assigned": 0.75, "candidate": 0.45}

#: Provenance keys the server strips from an imported blob because the app
#: presents the values it derives from them as its own calibrated judgement.
#: Mirrored here so a dry run reports the stripping instead of the user
#: discovering it as a silently empty column after the fact.
RESERVED_PROVENANCE_KEYS = ("p_correct", "calibration", "corroboration")

#: Widths of the `peak_assignment` columns an imported row lands in. Enforced
#: locally so an over-long string is named here rather than 422-ing an upload
#: that may already be several chunks in.
_MAX_LENGTHS = {
    "sample_peak_id": 20,
    "assigned_formula": 256,
    "ion_formula": 4096,
    "ionization_mechanism_id": 16,
    "isotope_label": 64,
    "isotope_formula": 256,
    "target_compound_id": 16,
    "target_ion_id": 16,
    "owner_sample_peak_id": 20,
}


class PublishError(RuntimeError):
    """A ledger that cannot be published as it stands, or a refused upload."""


# --------------------------------------------------------------------------- #
# evidence and tiering -- mirrors of the server's own functions
# --------------------------------------------------------------------------- #
def _load_plausibility() -> Callable[[str], float] | None:
    """The server's chemical-plausibility function, when this install has it.

    Evidence is ``fit x formula_plausibility(assigned_formula)`` and the server
    recomputes it from the formula on every row, so publishing a coherent tier
    means computing the same product with the same function. It lives in
    `mascope_tools` -- which peaky already depends on -- but was added after the
    2026.6.25 release that dependency currently resolves to, so an install
    pinned to a published wheel does not have it yet.

    :return: The plausibility function, or None when unavailable.
    """
    try:
        from mascope_tools.composition.heuristic_filter import formula_plausibility
    except ImportError:
        return None
    return formula_plausibility


#: Resolved once: it is a pure function and the import is the expensive part.
PLAUSIBILITY = _load_plausibility()

#: True when tiers can be derived exactly as the server will check them.
EXACT_PLAUSIBILITY = PLAUSIBILITY is not None

PLAUSIBILITY_UNAVAILABLE_NOTE = (
    "mascope_tools.composition.heuristic_filter.formula_plausibility is not "
    "available in this environment, so the tier PREVIEW below assumes a "
    "plausibility of 1.0 and will read high for a row whose formula the server "
    "weighs down. Nothing is at risk: the tier is derived server-side from the "
    "evidence and this preview is not published. Upgrade mascope-tools to a "
    "release exporting formula_plausibility for an exact preview."
)


def _finite(value: Any) -> float | None:
    """A float the server will accept, or None.

    The server sets ``allow_inf_nan=False`` on every float an imported row
    carries, and pandas hands out NaN for every empty cell in a CSV, so this is
    the funnel every numeric column goes through. Non-finite is None rather than
    an error: a missing score is ordinary ledger content, and the tier rules
    below have a documented answer for it.

    :param value: A cell from the ledger frame.
    :return: The value as a finite float, or None.
    """
    if value is None or value is pd.NA:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def evidence_for(fit_score: float | None, formula: str | None) -> float | None:
    """Evidence for a committed formula: ``fit x plausibility``.

    A mirror of the server's ``engine.evidence_for``, including its rounding and
    its fail-open behaviour, so a locally derived tier matches the one the
    server's coherence check computes. Plausibility is a pure function of the
    formula, so nothing here has to be declared or trusted -- it is recomputed
    on both sides from the same string.

    Fails open at plausibility 1.0 when the formula is absent, unparseable, or
    when this install cannot compute plausibility at all
    (:data:`EXACT_PLAUSIBILITY`); in the last case the caller is warned, because
    failing open is safe for the server's own check but can tier a row too high
    here.

    :param fit_score: The row's fit score, or None.
    :param formula: The committed neutral formula, or None.
    :return: The evidence, or None when there is no fit score to weigh.
    """
    fit = _finite(fit_score)
    if fit is None:
        return None
    if not formula:
        return fit
    if PLAUSIBILITY is None:
        # Still ROUNDED, because the server rounds every formula-bearing row and
        # the rounding is not cosmetic at a band edge: a fit of 0.7499873 is
        # 0.75 to the server, which is 'assigned' under a 0.75 band, while the
        # bare value is 'candidate'. Dropping the round here disagreed with the
        # server on exactly such a row and would have 422-ed the whole import.
        return round(fit, 4)
    try:
        return round(fit * float(PLAUSIBILITY(formula)), 4)
    except Exception:  # plausibility must never decide whether a row publishes
        return fit


def tier_for_evidence(
    evidence: float | None,
    *,
    candidate_threshold: float,
    assigned_threshold: float,
) -> str:
    """Map evidence onto a tier under the declared bands.

    A mirror of the server's ``engine.tier_for_evidence``, down to the
    ``evidence <= 0`` guard that lands a zero at 'below_assignability' even
    under a zero candidate band. Keyword-only for the same reason it is there:
    the two thresholds are the same type in the opposite order to their names,
    and a positional call written in band order silently inverts them.

    :param evidence: The row's evidence, or None.
    :param candidate_threshold: Evidence at or above which a row is 'candidate'.
    :param assigned_threshold: Evidence at or above which a row is 'assigned'.
    :return: The tier this evidence earns.
    """
    if evidence is None or not np.isfinite(evidence) or evidence <= 0:
        return TIER_BELOW_ASSIGNABILITY
    if evidence >= assigned_threshold:
        return TIER_ASSIGNED
    if evidence >= candidate_threshold:
        return TIER_CANDIDATE
    return TIER_BELOW_ASSIGNABILITY


def derive_tier(
    fit_score: float | None,
    formula: str | None,
    role: str,
    bands: dict[str, float],
) -> str:
    """The tier to publish for one row.

    Banded whenever there is a fit score to band. Without one the server admits
    exactly two tiers and this picks between them the way the in-app engine
    writes them: a peak nothing was proposed for is 'unassigned', while a row
    that committed a formula but has no usable score is 'below_assignability'.
    Neither is a claim about confidence, which is why the server accepts both.

    :param fit_score: The row's fit score, or None.
    :param formula: The row's committed neutral formula, or None.
    :param role: The row's Mascope role.
    :param bands: The declared evidence-scale bands.
    :return: A tier coherent with the evidence under these bands.
    """
    evidence = evidence_for(fit_score, formula)
    if evidence is None:
        return TIER_BELOW_ASSIGNABILITY if formula else TIER_UNASSIGNED
    return tier_for_evidence(
        evidence,
        candidate_threshold=bands["candidate"],
        assigned_threshold=bands["assigned"],
    )


def normalize_bands(bands: dict[str, float] | None) -> dict[str, float]:
    """Validate and normalise declared tier bands.

    Accepts the legacy ``identified`` spelling of the top band, which the server
    also accepts, so a config written against the older vocabulary keeps working
    without a version negotiation.

    :param bands: Declared bands, or None for the in-app defaults.
    :return: Bands under the current spellings.
    :raises PublishError: If a band is missing, non-numeric or out of order.
    """
    raw = dict(DEFAULT_TIER_BANDS if bands is None else bands)
    if "identified" in raw and "assigned" not in raw:
        raw["assigned"] = raw.pop("identified")
    raw.pop("identified", None)
    missing = [k for k in ("assigned", "candidate") if k not in raw]
    if missing:
        raise PublishError(f"tier_bands is missing {', '.join(missing)}")
    out = {}
    for key in ("assigned", "candidate"):
        value = _finite(raw[key])
        if value is None:
            raise PublishError(f"tier_bands['{key}'] is not a finite number")
        out[key] = value
    if out["assigned"] < out["candidate"]:
        raise PublishError(
            f"tier_bands assigned ({out['assigned']}) is below candidate "
            f"({out['candidate']}); the assigned band is the stricter one"
        )
    return out


# --------------------------------------------------------------------------- #
# intensity: which quantity Mascope expects for this sample
# --------------------------------------------------------------------------- #
def resolve_instrument_type(instrument_name: str) -> str | None:
    """Classify an instrument name as 'orbi' or 'tof'.

    A port of ``mascope_file.name.resolve_instrument_type`` (peaky does not
    depend on mascope-file). Returns None rather than raising, because an
    unclassifiable name is answered by asking the user, not by guessing.

    :param instrument_name: Instrument name.
    :return: 'orbi', 'tof', or None when unresolved.
    """
    name = (instrument_name or "").lower()
    if "orbi" in name:
        return "orbi"
    if "tof" in name or "api" in name:
        return "tof"
    return None


def instrument_type_from_filename(filename: str) -> str | None:
    """Classify a sample file the way Mascope does: the name up to the first
    underscore is the instrument.

    :param filename: Sample file name.
    :return: 'orbi', 'tof', or None when unresolved.
    """
    return resolve_instrument_type((filename or "").split("_")[0])


def intensity_column_for(sample: dict | None) -> tuple[str, str]:
    """The ledger column holding the intensity Mascope expects for this sample.

    Not the importer's choice: the in-app engine supplies peak HEIGHTS for
    Orbitrap files and peak AREAS for TOF files, the value lands on the batch
    occurrence that scales this sample's consensus vote, and the batch peak's
    declared unit is derived independently from the file's instrument type.
    Supplying the other quantity leaves the stored number disagreeing with the
    label above it and shifts the sample's weight against its peers -- silently,
    since nothing on either side can detect the swap.

    Which is exactly why an unclassifiable sample raises instead of defaulting:
    the failure this guards against is invisible once it happens.

    :param sample: The sample record from the API, or None.
    :return: The ledger column to read, and the instrument type it came from.
    :raises PublishError: If the instrument type cannot be resolved.
    """
    record = sample or {}
    kind = None
    if record.get("instrument"):
        kind = resolve_instrument_type(str(record["instrument"]))
    if kind is None and record.get("filename"):
        kind = instrument_type_from_filename(str(record["filename"]))
    if kind is None:
        raise PublishError(
            "cannot tell whether this sample is Orbitrap or TOF, and the two "
            "take different intensities (heights for Orbitrap, areas for TOF). "
            "Publishing the wrong one silently mis-weights this sample in the "
            "batch overview, so pass --intensity height|area explicitly."
        )
    return ("area" if kind == "tof" else "height"), kind


# --------------------------------------------------------------------------- #
# ledger -> import rows
# --------------------------------------------------------------------------- #
def _text(value: Any) -> str | None:
    """A trimmed string, or None for anything empty or missing."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def _json_cell(value: Any) -> Any:
    """Decode a ledger cell that holds JSON text, or pass a live object through.

    A ledger round-trips through CSV, so `alternatives` arrives as a string on a
    frame read from disk and as a list on one held in memory. Undecodable text
    is dropped rather than published: it is inspector detail, and no part of it
    is worth failing an import over.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = _text(value)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


#: peaky's own keys on an alternative entry, kept but moved aside so the entry
#: reads as Mascope's shape at the top level.
_ALTERNATIVE_ENGINE_KEYS = ("adduct", "ion_score", "raw_score", "eff_score", "ppm")


def _alternatives(value: Any) -> list | None:
    """peaky's runner-up list, in the shape the peak inspector reads.

    The two engines name this field's contents differently and only one of them
    is the one the app renders. Mascope writes ``assigned_formula`` per entry;
    peaky writes ``formula``. Published verbatim, every close alternative shows
    as "?" in the inspector - the formula is there, under a name nothing looks
    for. This is the one place the "compatibility is by construction" claim does
    not hold, because `alternatives` is opaque JSON that no schema validates, so
    nothing refuses the mismatch and it surfaces as a blank instead of an error.

    ``source`` is set to 'untargeted' to match the row itself: it says which kind
    of search produced the candidate, and peaky's grid search is the untargeted
    one. It also makes the entry eligible for the inspector's score-on-request
    path, which needs a formula-only alternative carrying no ion or target id.

    Plausibility is deliberately absent rather than guessed: it is this server's
    reading of the chemistry, and the inspector renders a dash for an entry that
    has none, which is the honest answer for a candidate peaky never weighed.

    :param value: The ledger cell, JSON text or a live list.
    :return: Entries in Mascope's shape, or None.
    """
    entries = _json_cell(value)
    if not isinstance(entries, list):
        return None
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        formula = _text(entry.get("formula") or entry.get("assigned_formula"))
        if formula is None:
            continue
        engine_detail = {
            key: _sanitize(entry[key])
            for key in _ALTERNATIVE_ENGINE_KEYS
            if entry.get(key) is not None
        }
        mapped: dict[str, Any] = {"assigned_formula": formula, "source": "untargeted"}
        if engine_detail:
            mapped["engine_provenance"] = engine_detail
        out.append(mapped)
    return out or None


def _sanitize(value: Any) -> Any:
    """Recursively replace non-finite floats with None.

    The whole payload goes through this before upload. NaN reaches JSON from two
    directions -- every empty cell pandas parses, and peaky's own manifest,
    which is written with the stdlib's NaN-permitting encoder -- and the server
    rejects both on floats it validates while a NaN inside an opaque blob is
    stored and then breaks the *read* of the whole run.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return _sanitize(value.item())
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


#: Ledger columns carried through to `provenance.engine_provenance` -- peaky's
#: own judgement, kept under a name of its own so it survives beside Mascope's
#: rather than being mistaken for it (or stripped as a reserved key).
_ENGINE_PROVENANCE_COLUMNS = (
    "tier",
    "tier_reason",
    "confidence",
    "adduct",
    "eff_score",
    "eff_margin",
    "tied",
    "candidate_density",
    "composite_note",
    "degeneracy_density",
    "degeneracy_note",
    "assigned_fraction",
    "pass_no",
    "method",
    "locked",
    "anchor_peak_id",
    "series_unit",
    "commentary",
    "isotopologues",
    "below_assignability",
    "iso_match_score",
    "compound_score",
    "dbe",
    "sparsity",
)


def _engine_provenance(row: pd.Series, engine_tier: str | None) -> dict:
    """peaky's reasoning for this row, for the peak inspector.

    The VERDICT now has a column of its own (`engine_tier`), so this blob is the
    explanation rather than the answer: the tier reason, the confidence label,
    the arbitration numbers and the commentary that say *why* peaky landed
    where it did. The mapped verdict is repeated here under peaky's own
    spelling, because the blob is also what survives an export and a reader
    should not have to join it back to a column to make sense of it.
    """
    out: dict[str, Any] = {"engine": ENGINE}
    for column in _ENGINE_PROVENANCE_COLUMNS:
        if column not in row.index:
            continue
        value = _sanitize(row[column])
        if value is None or value == "":
            continue
        out[column] = value
    if "tier" in out:
        out["peaky_tier"] = out.pop("tier")
    if engine_tier is not None:
        out["engine_tier"] = engine_tier
    return out


def build_rows(
    ledger: pd.DataFrame,
    *,
    intensity_column: str,
    bands: dict[str, float],
    mechanism_ids: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Translate a peaky ledger into import rows.

    Pure: no network, no server state. Everything the server would refuse and
    that is knowable from the frame alone is either fixed here (vocabularies,
    non-finite floats, derived tiers) or reported in the summary so the caller
    can stop before uploading.

    :param ledger: A peaky ledger frame.
    :param intensity_column: 'height' or 'area' -- see :func:`intensity_column_for`.
    :param bands: Declared evidence-scale bands, already normalised.
    :param mechanism_ids: Optional adduct-notation -> mechanism id mapping. A
        supplied id must exist on the deployment and carry the sample's
        polarity, so an unmapped adduct sends null rather than a guess.
    :return: The rows, and a summary of what was translated and what was dropped.
    :raises PublishError: If the frame is missing columns an import needs.
    """
    required = {"peak_id", "mz", "role", intensity_column}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise PublishError(
            f"ledger is missing required column(s): {', '.join(missing)}"
        )

    frame = ledger
    dropped_synthetic = 0
    if "synthetic" in frame.columns:
        synthetic = frame["synthetic"].fillna(False).astype(bool)
        dropped_synthetic = int(synthetic.sum())
        frame = frame[~synthetic]

    # An iso_child publishes its owner's formula: Mascope models the family by
    # having the child carry the M0's committed formula, and the child's own
    # ledger row holds no formula of its own.
    formula_by_peak: dict[str, str] = {}
    if "neutral_formula" in frame.columns:
        for peak_id, formula in zip(frame["peak_id"], frame["neutral_formula"]):
            text = _text(formula)
            if text:
                formula_by_peak[str(peak_id)] = text

    rows: list[dict] = []
    dropped_no_peak_id = 0
    unknown_roles: dict[str, int] = {}
    oversized: list[str] = []
    tier_counts: dict[str, int] = {}
    engine_tiers: dict[str, int] = {}
    disagreements = 0
    role_counts: dict[str, int] = {}
    reserved_seen: set[str] = set()
    inherited_formulas = 0

    for _, row in frame.iterrows():
        peak_id = _text(row.get("peak_id"))
        if peak_id is None:
            dropped_no_peak_id += 1
            continue

        peaky_role = _text(row.get("role")) or "unexplained"
        role = ROLE_MAP.get(peaky_role)
        if role is None:
            unknown_roles[peaky_role] = unknown_roles.get(peaky_role, 0) + 1
            continue

        owner = _text(row.get("parent_peak_id")) if role == "iso_child" else None
        formula = _text(row.get("neutral_formula"))
        if formula is None and owner is not None:
            formula = formula_by_peak.get(owner)
            if formula is not None:
                inherited_formulas += 1

        # An iso_child is scored by how well it matches the predicted
        # isotopologue; an M0 by how well the ion pattern fits.
        fit = _finite(row.get("iso_match_score") if role == "iso_child" else None)
        if fit is None:
            fit = _finite(row.get("ion_score"))
        # fit_score carries a `ge=0, le=1` bound server-side; peaky's arbitration
        # scores can sit outside it, and a clamp would publish a number that is
        # not the one that was scored.
        if fit is not None and not (0.0 <= fit <= 1.0):
            fit = None

        # peaky's OWN verdict, published as `engine_tier`. Only M0 rows carry
        # one: peaky tiers committed assignments and says nothing about
        # isotopologue children, artifacts or the unexplained residual, and
        # null is the honest way to say so - the server treats absence as "no
        # verdict", not as agreement.
        engine_tier = ENGINE_TIER_MAP.get((_text(row.get("tier")) or "").lower())
        if engine_tier is not None:
            engine_tiers[engine_tier] = engine_tiers.get(engine_tier, 0) + 1

        # What the SERVER will tier this row, computed only to show the user
        # before anything is sent. It is deliberately NOT published: `tier` is a
        # pure function of the fit, the formula and the declared bands, all of
        # which the server holds, so sending it would be a second copy of the
        # server's own rule - and a copy that disagrees at a band edge refuses
        # the whole import. Omitted from the payload, the tier is derived
        # server-side and cannot disagree.
        predicted = derive_tier(fit, formula, role, bands)
        target_compound_id = _text(row.get("target_compound_id"))
        adduct = _text(row.get("adduct"))
        ion_formula = _text(row.get("ion_formula"))

        provenance = {"engine_provenance": _engine_provenance(row, engine_tier)}
        supplied = _json_cell(row.get("provenance"))
        if isinstance(supplied, dict):
            for key, value in supplied.items():
                if key in RESERVED_PROVENANCE_KEYS:
                    reserved_seen.add(key)
                    continue
                provenance.setdefault(key, _sanitize(value))

        record = {
            "sample_peak_id": peak_id,
            "sample_peak_mz": _finite(row.get("mz")),
            "sample_peak_intensity": _finite(row.get(intensity_column)),
            "sample_peak_tof": None,
            "role": role,
            "assigned_formula": formula,
            "ion_formula": ion_formula,
            "ionization_mechanism_id": (
                (mechanism_ids or {}).get(adduct) if adduct else None
            ),
            # An M0 row's isotope label is the string "M0", not nothing: that is
            # the in-app engine's convention and what the ledger's isotope
            # column renders. Left null, peaky's main peaks read as "-" beside
            # in-app rows that say M0 for the same thing. Children keep peaky's
            # own label (13C, 81Br, ...), which is the same vocabulary.
            "isotope_label": _text(row.get("iso_label"))
            or ("M0" if role == "M0" else None),
            # And the M0's own isotopologue formula is its ion formula - the
            # isotopologue a main peak stands for is the unlabelled one.
            "isotope_formula": ion_formula if role == "M0" else None,
            "source": "database" if target_compound_id else "untargeted",
            "fit_score": fit,
            "mz_error_ppm": _finite(row.get("ppm_error")),
            "abundance_error": None,
            # No `tier`: the server derives it. See `predicted` above.
            "engine_tier": engine_tier,
            "target_compound_id": target_compound_id,
            "target_ion_id": _text(row.get("target_ion_id")),
            "owner_sample_peak_id": owner,
            "alternatives": _alternatives(row.get("alternatives")),
            "provenance": provenance,
        }

        if record["sample_peak_mz"] is None or record["sample_peak_intensity"] is None:
            # Both are non-nullable server-side; a row without them is not a
            # measurement, so it is dropped rather than sent to be refused.
            dropped_no_peak_id += 1
            continue

        for field, limit in _MAX_LENGTHS.items():
            value = record.get(field)
            if isinstance(value, str) and len(value) > limit:
                oversized.append(f"{peak_id}.{field} ({len(value)} > {limit})")

        rows.append(record)
        tier_counts[predicted] = tier_counts.get(predicted, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        if engine_tier is not None and engine_tier != predicted:
            disagreements += 1

    summary = {
        "rows": len(rows),
        "by_predicted_tier": tier_counts,
        "by_engine_tier": engine_tiers,
        "engine_tier_disagreements": disagreements,
        "by_role": role_counts,
        "intensity_column": intensity_column,
        "dropped_synthetic": dropped_synthetic,
        "dropped_incomplete": dropped_no_peak_id,
        "unknown_roles": unknown_roles,
        "oversized": oversized,
        "inherited_formulas": inherited_formulas,
        "reserved_provenance_dropped": sorted(reserved_seen),
        "exact_plausibility": EXACT_PLAUSIBILITY,
    }
    return rows, summary


def validate_rows(rows: list[dict]) -> list[str]:
    """Check locally everything the server checks that does not need the sample.

    Peak existence is the one payload-wide rule that genuinely needs the server,
    so it is not here. The rest are caught before the first byte is uploaded --
    which matters because a refusal mid-assembly leaves an `importing` run
    holding the sample until it is abandoned or aged out.

    :param rows: Translated import rows.
    :return: Human-readable problems; empty when the payload is publishable.
    """
    problems: list[str] = []
    if not rows:
        return ["the ledger produced no publishable rows; a run with no rows is refused"]

    seen: dict[str, int] = {}
    for row in rows:
        peak_id = row["sample_peak_id"]
        seen[peak_id] = seen.get(peak_id, 0) + 1
    duplicates = sorted(p for p, n in seen.items() if n > 1)
    if duplicates:
        shown = ", ".join(duplicates[:5])
        more = f" (+{len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        problems.append(
            f"{len(duplicates)} peak(s) carry more than one row: {shown}{more}. "
            "A run holds at most one assignment per peak."
        )

    owned = {r["sample_peak_id"] for r in rows if r.get("owner_sample_peak_id")}
    for row in rows:
        owner = row.get("owner_sample_peak_id")
        if owner and row["role"] != "iso_child":
            problems.append(
                f"{row['sample_peak_id']}: role '{row['role']}' carries an owner; "
                "only an iso_child may."
            )
        if owner and owner not in seen:
            problems.append(
                f"{row['sample_peak_id']}: owner {owner} is not in this import."
            )
        if owner and owner in owned:
            problems.append(
                f"{row['sample_peak_id']}: owner {owner} is itself an iso_child; "
                "the owner link is one level deep."
            )
        if row["role"] not in IMPORT_ROLES:
            problems.append(f"{row['sample_peak_id']}: unknown role '{row['role']}'")
    return problems


# --------------------------------------------------------------------------- #
# run-level payload
# --------------------------------------------------------------------------- #
def _capped(value: Any, field: str, log: Callable[[str], None]) -> dict | None:
    """Drop the biggest keys of an opaque blob until it fits the server's cap."""
    if value is None:
        return None
    blob = _sanitize(value)
    if not isinstance(blob, dict):
        blob = {"value": blob}
    size = len(json.dumps(blob, default=str).encode("utf-8"))
    if size <= MAX_JSON_BYTES:
        return blob
    kept = dict(blob)
    dropped = []
    by_size = sorted(
        kept,
        key=lambda k: len(json.dumps(kept[k], default=str).encode("utf-8")),
        reverse=True,
    )
    for key in by_size:
        if len(json.dumps(kept, default=str).encode("utf-8")) <= MAX_JSON_BYTES:
            break
        kept.pop(key)
        dropped.append(key)
    kept["_truncated"] = dropped
    log(
        f"[publish] {field} was {size} bytes, above the {MAX_JSON_BYTES}-byte "
        f"cap; dropped {', '.join(dropped)}"
    )
    return kept


def build_calibration(
    manifest: dict | None,
    *,
    note: str | None = None,
) -> dict:
    """The calibration disclosure that replaces the m/z verification gate.

    An import bypasses the server-side gate an in-app run passes, so the run
    records what it calibrated against instead -- and 'nothing' is a valid,
    and honest, answer. peaky calibrates its own mass axis per run rather than
    against Mascope's verified assignments, so that is what this says.

    :param manifest: The run manifest peaky wrote beside the ledger.
    :param note: Extra disclosure from the caller.
    :return: The calibration blob.
    """
    prescan = (manifest or {}).get("prescan") or {}
    disclosure: dict[str, Any] = {
        "provisional": True,
        "source": "engine",
        "verified_against_mascope": False,
        "description": (
            "peaky estimates the run's m/z offset from its own high-confidence "
            "commits; it does not use Mascope's verified-assignment calibration."
        ),
    }
    if isinstance(prescan, dict):
        for key in ("ppm_offset", "offset_ppm", "mz_offset_ppm", "ppm", "sigma_ppm"):
            value = _finite(prescan.get(key))
            if value is not None:
                disclosure[key] = value
    if note:
        disclosure["note"] = note
    return disclosure


def build_config(manifest: dict | None, log: Callable[[str], None] = print) -> dict:
    """The engine's run configuration, stored verbatim and never read.

    peaky's manifest is the natural carrier -- pass summaries, module versions
    and hashes make the run reproducible -- but it is unbounded and the server
    re-serves it on every run listing, so it is capped.
    """
    config = dict(manifest or {})
    config.setdefault("engine", ENGINE)
    return _capped(config, "config", log) or {"engine": ENGINE}


def engine_version(manifest: dict | None) -> str:
    """peaky's version string for the run record."""
    versions = (manifest or {}).get("module_versions") or {}
    try:
        from peaky import __version__ as pkg_version
    except Exception:  # noqa: BLE001 - version reporting must not fail a publish
        pkg_version = "unknown"
    assign_version = versions.get("assign") if isinstance(versions, dict) else None
    return f"{pkg_version}+assign{assign_version}" if assign_version else str(pkg_version)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def _row_bytes(row: dict) -> int:
    return len(json.dumps(row, default=str).encode("utf-8"))


def chunk_rows(
    rows: list[dict],
    *,
    max_rows: int = MAX_ROWS_PER_REQUEST,
    max_bytes: int = MAX_BODY_BYTES,
    envelope_bytes: int = 0,
) -> list[list[dict]]:
    """Split rows into request-sized chunks.

    Sized by serialized bytes as well as by row count: a row carrying
    `alternatives` and `provenance` is an order of magnitude larger than a slim
    one, so the row cap is the ceiling rather than the target. A single row over
    the byte budget still goes out alone -- the server's own error is a better
    answer than a client-side refusal to try.

    :param rows: Translated import rows, in payload order.
    :param max_rows: Rows one request may carry.
    :param max_bytes: Byte ceiling on one request body.
    :param envelope_bytes: Bytes the run-level fields take in the same body.
    :return: Chunks, in order.
    """
    budget = max(1024, max_bytes - envelope_bytes - 4096)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for row in rows:
        row_size = _row_bytes(row) + 1
        if current and (len(current) >= max_rows or size + row_size > budget):
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += row_size
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #
def _post(client, sample_id: str, body: dict, timeout) -> dict:
    """POST one import request and return the state record from its envelope."""
    from mascope_sdk._http import http_post

    response = http_post(
        client.url,
        f"peak-assignments/sample/{sample_id}/runs/import",
        client.access_token,
        _sanitize(body),
        timeout=timeout,
    )
    payload = response.json() or {}
    data = payload.get("data") or []
    if not data:
        raise PublishError(
            f"import response carried no state record: {payload!r}"
        )
    return data[0]


def publish(
    client,
    sample_id: str,
    rows: list[dict],
    *,
    tier_bands: dict[str, float],
    calibration: dict,
    config: dict,
    version: str,
    import_id: str | None = None,
    max_rows: int = MAX_ROWS_PER_REQUEST,
    timeout=(15, 300),
    log: Callable[[str], None] = print,
) -> dict:
    """Upload a translated ledger as one imported run.

    Assembles across as many requests as the payload needs. The first creates
    the run and returns its id plus the deployment's effective row cap; each
    later request carries the run id and a row OFFSET, which is what makes a
    retry safe -- the SDK retries POSTs on timeouts with no way to opt out, so a
    chunk the server applied but never acknowledged is re-sent, and the offset
    is what lets the server recognise it as a replay instead of duplicating
    rows. ``import_id`` covers the same hazard for the create, which has no
    offset to be idempotent about.

    :param client: A connected MascopeClient.
    :param sample_id: The sample this run belongs to.
    :param rows: Translated import rows.
    :param tier_bands: Declared evidence-scale bands.
    :param calibration: The calibration disclosure.
    :param config: The engine's opaque run configuration.
    :param version: The engine version string to stamp.
    :param import_id: Idempotency key; a fresh one starts a new run.
    :param max_rows: Rows per request to start with; the create response's
        ``max_rows_per_request`` lowers it when the deployment caps it tighter.
    :param timeout: Connect/read timeout for each request.
    :param log: Progress sink.
    :return: The final state record (run id, status, row count).
    :raises PublishError: If the server refuses the import.
    """
    from mascope_sdk._http import MascopeAPIError

    key = import_id or uuid.uuid4().hex
    envelope = {
        "engine": ENGINE,
        "engine_version": version,
        "tier_bands": tier_bands,
        "calibration": calibration,
        "config": config,
    }
    envelope_bytes = len(json.dumps(_sanitize(envelope), default=str).encode("utf-8"))
    chunks = chunk_rows(rows, max_rows=max_rows, envelope_bytes=envelope_bytes)
    log(
        f"[publish] {len(rows)} row(s) in {len(chunks)} request(s) "
        f"(import_id {key})"
    )

    run_id: str | None = None
    offset = 0
    state: dict = {}
    for index, chunk in enumerate(chunks):
        last = index == len(chunks) - 1
        body: dict[str, Any] = {
            "engine": ENGINE,
            "engine_version": version,
            "rows": chunk,
            "chunk": {
                "import_id": key,
                "index": offset,
                "complete": last,
            },
        }
        if run_id is None:
            # Run-level fields are required on the request that creates the run
            # and are pure weight on the rest.
            body["tier_bands"] = tier_bands
            body["calibration"] = calibration
            body["config"] = config
        else:
            body["chunk"]["run_id"] = run_id

        try:
            state = _post(client, sample_id, body, timeout)
        except MascopeAPIError as exc:
            raise PublishError(_upload_failure_message(exc, run_id, offset)) from exc

        run_id = state.get("peak_assignment_run_id") or run_id
        served_offset = state.get("rows")
        # The server's count is authoritative: it is what a replayed chunk
        # resynchronises from, and a client that trusts its own arithmetic
        # instead sends the next chunk at an offset the server calls a gap.
        offset = int(served_offset) if served_offset is not None else offset + len(chunk)
        cap = state.get("max_rows_per_request")
        if cap and index == 0 and cap < MAX_ROWS_PER_REQUEST and len(chunks) > 1:
            log(
                f"[publish] deployment caps requests at {cap} rows; "
                "re-chunking the remainder"
            )
            remaining = [r for c in chunks[index + 1:] for r in c]
            chunks = chunks[: index + 1] + chunk_rows(remaining, max_rows=int(cap))
        log(
            f"[publish]   request {index + 1}/{len(chunks)}: "
            f"{offset} row(s) staged, status {state.get('run_status')}"
        )

    if state.get("run_status") != "completed":
        raise PublishError(
            f"import finished with status {state.get('run_status')!r} rather than "
            f"'completed'; the run ({run_id}) is still assembling. Re-run with "
            f"--import-id {key} to resume, or delete it to release the sample."
        )
    return state


def _upload_failure_message(exc: Exception, run_id: str | None, offset: int) -> str:
    """Turn an SDK error into something a user can act on.

    A 409's body is deliberately not machine-readable, so recovery is by state
    rather than by parsing prose: an import that dies mid-assembly leaves a
    non-terminal run that refuses every later import AND in-app assign for the
    sample until it is deleted or aged out, and the way out is the run id.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    hint = ""
    if "already completed" in lowered:
        # Not a conflict with someone else's work: this import_id was already
        # published, and the server resolved it to that run rather than making a
        # second one. Re-running with the same key is how a resume is spelled,
        # so the fix is to stop asking for one.
        hint = (
            "\n  This import_id has already been published in full, so the server "
            "returned the run it made for it. Nothing was duplicated. To publish "
            "the ledger AGAIN as a new run, re-run without --import-id; to resume "
            "an interrupted upload, keep the id but do not re-run a finished one."
        )
    elif "409" in lowered or "conflict" in lowered:
        hint = (
            "\n  A 409 means the sample already has a run in flight -- either an "
            "in-app assignment, or an earlier import that never finished. Find it "
            "in the run selector (an assembling run is visible there) or via "
            "list_runs, then delete that run to release the sample."
        )
    elif "422" in lowered or "validation" in lowered:
        hint = (
            "\n  A 422 is a payload rule: the message names the offending rows. "
            "Peak existence is the usual one -- every sample_peak_id must belong "
            "to this sample, so check the ledger was computed for it."
        )
    elif "413" in lowered:
        hint = (
            "\n  A 413 came from the proxy, not the API: the request body exceeded "
            "its client_max_body_size. Publish with a smaller --max-rows."
        )
    where = (
        f" (run {run_id}, {offset} row(s) staged)" if run_id else " (no run created)"
    )
    return f"import refused{where}: {text}{hint}"
