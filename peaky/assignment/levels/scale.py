"""The evidence scale's vocabulary: levels, buckets, claims and the output columns.

``evidence_level`` takes one of LEVELS (3c 4a 4b 5a 5b) or a BUCKET (reagent,
NA). Levels 1 and 2 (an authentic standard, a library spectrum) are defined and
never assigned. ``NA`` is written literally: pandas reads it as NaN by default,
so a reader that must tell "not assessed" from "no level" reads the ``claim``
column or passes ``keep_default_na=False``.

The claim is what a reader may say about the committed formula: identified
(level <= 3), neutral (4a: the neutral established, no named identity), ion
(4b: the ion composition only) or tentative (5a, 5b, or a committed row with no
level). The two buckets are reported beside the claims, never folded into
them: reagent ("reagent") and NA ("not assessed").
"""
from __future__ import annotations

import pandas as pd

SCALE_RELEASE = "0.10.0"
LEVEL_ORDER = ["1", "2", "3c", "4a", "4b", "5a", "5b"]
LEVELS = ["3c", "4a", "4b", "5a", "5b"]
BUCKETS = ["reagent", "NA"]
LEVEL_MEANING = {
    "1": "confirmed by an authentic standard in the same source and chemistry (never assigned)",
    "2": "matched to a library spectrum or MS2 standard (never assigned)",
    "3c": "ion established, neutral / adduct split pinned, and a named context-list entry names the neutral",
    "4a": "ion established, split pinned, and a positive fact (an own isotope line of the neutral's elements, "
          "the 15N label, or an NH4 adduct tracking its parent)",
    "4b": "ion established; the split is open, pinned without a positive fact, or the channel reads the ion only",
    "5a": "a competitor ion is left in the calibrated window",
    "5b": "rejected by a check, or nothing could be enumerated or tested",
    "reagent": "a reagent ion or reagent cluster (not levelled)",
    "NA": "not assessed on this instrument class",
}
CLAIMS = ("identified", "neutral", "ion", "tentative")
CLAIM_REAGENT = "reagent"
CLAIM_NA = "not assessed"
CLAIM_KEYS = CLAIMS + (CLAIM_REAGENT, CLAIM_NA)
CLAIM_IDENTIFIED = frozenset({"1", "2", "3c"})
CLAIM_NEUTRAL = frozenset({"4a"})
CLAIM_ION = frozenset({"4b"})
CLAIM_MEANING = {
    "identified": "the compound is named: ion established, split pinned and a named context-list entry (level <= 3)",
    "neutral": "the neutral is established (4a): ion and split pinned with a positive fact; no named identity",
    "ion": "the ion composition is established (4b); the neutral / adduct split or the process stays open",
    "tentative": "a competitor is left, the reading is rejected, or the row has no level (5a, 5b, none)",
    "reagent": "a reagent ion or reagent cluster (not levelled)",
    "not assessed": "not assessed on this instrument class (the scale needs an Orbitrap-class width model)",
}
#: the columns every committed M0 row carries (empty off M0)
COLUMNS = ("evidence_level", "evidence", "would_lift", "competitors_left", "tags", "context", "context_source",
           "claim")


def claim_class(level) -> str:
    """The claim a level supports; a missing level reads tentative."""
    if level is None or (not isinstance(level, str) and pd.isna(level)):
        return "tentative"
    lv = str(level).strip()
    if lv in CLAIM_IDENTIFIED:
        return "identified"
    if lv in CLAIM_NEUTRAL:
        return "neutral"
    if lv in CLAIM_ION:
        return "ion"
    if lv == "reagent":
        return CLAIM_REAGENT
    if lv == "NA":
        return CLAIM_NA
    return "tentative"


def summarize(levels) -> dict:
    """{level: n} over LEVELS + BUCKETS, in that order, zeros dropped (``NA``
    counted from the literal token; a NaN level is no level)."""
    counts = pd.Series(levels, dtype=object).dropna().astype(str).value_counts()
    return {k: int(counts[k]) for k in LEVELS + BUCKETS if k in counts.index and counts[k]}


def summarize_claims(claims) -> dict:
    """{claim: n} over the four claims and the two buckets, in that order, zeros kept."""
    counts = pd.Series(claims, dtype=object).dropna().astype(str).value_counts()
    return {k: int(counts.get(k, 0)) for k in CLAIM_KEYS}
