"""The ONLY module that talks to Mascope.

It wraps the mascope-sdk MascopeClient and exposes exactly the operations the
pipeline needs:

  * connect()                  -- build a client from ~/.mascope/.env
  * fetch_peaks()              -- pull + cache the raw peak table
  * resolve_mechanism_ids()    -- ionization name -> id
  * query_candidates()         -- cheminfo formula enumerator for one m/z
  * score_candidates()         -- match_compounds -> flat per-isotopologue table

The scoring oracle is Mascope: match_compounds returns a compound -> ion ->
isotopologue tree, every node carrying its own match_score and (for isotopes)
the attributed sample_peak_id. flatten_match_tree() turns that tree into a flat
table and is a PURE function, unit-tested offline against a captured fixture.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

__version__ = "0.6.0"  # modern (datasets-based) servers only; a batch is settled
#                          ONCE (exact id > exact name > unique substring) and the
#                          time series is loaded by that exact name (resolve_batch);
#                          the peaks read carries signal_to_noise and the sample
#                          carries a fitted PatternScoring

# Credential .env search order. The long-running MCP server holds a STALE in-memory
# token and 401s; the SDK reads the live file, so always load from disk.
# Precedence: --env / $MASCOPE_ENV > a PROJECT-LOCAL .env (the repo root, next to
# pyproject.toml/the package — clone-and-go — or the current dir) > the home
# locations (~/.mascope/.env is the canonical shared one).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))           # .../peaky
_REPO_ENV = os.path.join(os.path.dirname(_PKG_DIR), ".env")     # repo-root/.env (editable install)
CANONICAL_ENV = "~/.mascope/.env"
ENV_SEARCH = [_REPO_ENV, ".env", CANONICAL_ENV, "~/mascope-mcp/.env",
              "~/.claude/skills/mascope-sdk/.env"]
CACHE_ROOT = Path(os.path.expanduser("~/.mascope-assign-cache"))

#: The fit score every candidate is scored with. Stamped on a published run so
#: a v1 reference and a v2 one are told apart in the store rather than by date.
SCORE_VERSION = 2

#: Anchors below which no offset is claimed either. A median and a robust spread
#: are not the same measurement: the spread needs a distribution
#: (`MASS_ACCURACY_MIN_ANCHORS`), the median needs fewer points. Discarding the
#: offset because the width could not be fitted is not the cautious choice it
#: looks like - it asserts the instrument sits on calibration, which on a source
#: that sits 1.2 ppm low charges that error to every candidate and to none of its
#: rivals equally, since the rival with the compensating error then scores best.
#:
#: Five, not three, and the reason is what an anchor can be. An anchor is a peak
#: the server matched to a known species within the instrument's MATCHING window,
#: which on a TOF is 15 ppm - five times its accuracy - so a mis-match sits in
#: the set looking like a measurement. Three anchors, two of them mis-matched to
#: the same wrong species, put the median on the mis-match: on a bromide TOF that
#: produced an offset of -10.4 ppm for a source the engine measures within 0.3
#: ppm of calibration. Two bad anchors cannot carry a median of five.
MIN_OFFSET_ANCHORS = 5

#: Per-sample `(PatternScoring, snapshot)`, keyed by sample id (see
#: `scoring_for_sample` and `scoring_snapshot`).
_SCORING_CACHE: dict = {}

#: Per-sample mass-dependent scoring centre (C42): a `masscal.MassTrend` the
#: sample's own calibration accepted (`set_scoring_trend`, from assign.run) or
#: its stand-in inherited with a measured sample's snapshot. Candidates of a
#: sample listed here are scored against the trend's centre at each line's m/z
#: instead of the snapshot's one offset. `_SCORING_TREND_INHERITED` holds the
#: ids whose trend came with the snapshot, which a new run keeps.
_SCORING_TREND: dict = {}
_SCORING_TREND_INHERITED: set = set()


def _trend_record(trend) -> dict:
    return {"a": round(float(trend.a), 6), "b": round(float(trend.b), 6),
            "sigma": round(float(trend.sigma), 6), "n": int(trend.n),
            "mz_lo": None if trend.mz_lo is None else round(float(trend.mz_lo), 4),
            "mz_hi": None if trend.mz_hi is None else round(float(trend.mz_hi), 4)}


def _trend_from_record(rec: dict):
    from peaky.assignment.masscal import MassTrend
    return MassTrend(float(rec["a"]), float(rec["b"]), float(rec.get("sigma") or 0.0),
                     int(rec.get("n") or 0),
                     None if rec.get("mz_lo") is None else float(rec["mz_lo"]),
                     None if rec.get("mz_hi") is None else float(rec["mz_hi"]))


def set_scoring_trend(sample_id: str, trend) -> None:
    """Score this sample's candidates against `trend` (a `masscal.MassTrend`)
    from now on, and record it in the sample's scoring snapshot (`mu_source`
    "trend", the fit under `trend`) so a run's record and every stand-in that
    inherits the snapshot (a decoy arm) are judged at the same centre."""
    _SCORING_TREND[sample_id] = trend
    if sample_id in _SCORING_CACHE:
        scoring, snap = _SCORING_CACHE[sample_id]
        snap = dict(snap)
        snap["mu_source"] = "trend"
        snap["trend"] = _trend_record(trend)
        _SCORING_CACHE[sample_id] = (scoring, snap)


def scoring_trend(sample_id: str):
    """The sample's mass-dependent scoring centre, or None (constant offset)."""
    return _SCORING_TREND.get(sample_id)


def reset_scoring_trend(sample_id: str) -> None:
    """Forget a trend the sample's OWN calibration set (a new run of it fits its
    own); a trend inherited with a stand-in's snapshot stays."""
    if sample_id in _SCORING_TREND_INHERITED:
        return
    if _SCORING_TREND.pop(sample_id, None) is not None:
        # the cached snapshot names the trend; the next scoring_for_sample
        # recomputes it from the (cached) peaks
        _SCORING_CACHE.pop(sample_id, None)


def _find_env(explicit: str | None = None) -> str:
    # precedence: explicit arg (e.g. CLI --env) > $MASCOPE_ENV > the search list.
    head = [explicit] if explicit else ([os.environ["MASCOPE_ENV"]]
                                        if os.environ.get("MASCOPE_ENV") else [])
    for cand in head + ENV_SEARCH:
        p = os.path.expanduser(cand)
        if os.path.exists(p):
            return p
    # last resort: walk up from the cwd for a project-local .env (clone-and-go)
    try:
        from dotenv import find_dotenv
        found = find_dotenv(usecwd=True)
        if found:
            return found
    except Exception:
        pass
    return os.path.expanduser(CANONICAL_ENV)
MATCH_BATCH = 200   # match_compounds times out above ~500

# Default server-side match parameters. mz_tolerance is INTEGER ppm.
DEFAULT_MATCH_PARAMS = {
    "mz_tolerance": 5,
    "isotope_ratio_tolerance": 0.2,
    "peak_min_intensity": 0.0,
    "min_isotope_abundance": 0.15,
    "min_isotope_correlation": 0.7,
    "probable_match_threshold": 0.8,
    "possible_match_threshold": 0.4,
}

# bracketed heavy-isotope tokens, e.g. [13C], [13C]2, [81Br], [18O], [37Cl]
_ISO_TOKEN = re.compile(r"\[(\d+[A-Z][a-z]?)\](\d*)")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def connect(env_path: str | None = None, workspace: str | None = None):
    """Build a MascopeClient from the .env (MASCOPE_URL, MASCOPE_ACCESS_TOKEN).
    Searches a project-local .env (repo root or cwd) then the home locations, or
    $MASCOPE_ENV, unless env_path is given. Process env vars of the same name also
    work directly. The Mascope WORKSPACE is selected by `workspace` (name /
    substring / id) or $MASCOPE_WORKSPACE; with neither, the SDK auto-selects only
    when the token sees exactly one workspace (else use `peaky list workspaces`)."""
    from dotenv import load_dotenv
    path = _find_env(env_path)
    load_dotenv(path)
    url = os.environ.get("MASCOPE_URL")
    tok = os.environ.get("MASCOPE_ACCESS_TOKEN")
    if not url or not tok:
        raise RuntimeError(
            f"MASCOPE_URL / MASCOPE_ACCESS_TOKEN not found (looked in {path}). "
            "Copy .env.example to .env in the repo root (or ~/.mascope/.env) and fill "
            "it in, set $MASCOPE_ENV to your .env path, or export the two variables.")
    ws = workspace or os.environ.get("MASCOPE_WORKSPACE") or None
    from mascope_sdk import MascopeClient
    try:
        return MascopeClient(url=url, access_token=tok, workspace=ws)
    except Exception as e:                       # noqa: BLE001 — friendlier guidance
        if not ws and "workspace" in str(e).lower():
            raise RuntimeError(
                "This Mascope server exposes multiple workspaces; pick one with "
                "`--workspace NAME` (or set MASCOPE_WORKSPACE). "
                "Run `peaky list workspaces` to see them.") from e
        raise


# Batch-name matching follows the SDK's unified contract (mascope-sdk >= 2026.8.12):
# a plain string is a case-insensitive LITERAL substring on every name filter
# (metacharacters like the ^ in '^Nitrate' or the parens in '(Ur+ CIMS)' carry no
# regex meaning), and only a compiled re.Pattern is a regex. Pass batch names RAW —
# pre-escaping them re-escapes the backslashes into literals that match nothing
# (it only still "works" via the SDK's deprecation shim, which re-reads a
# nothing-matching string as a regex under a DeprecationWarning).
#
# The SDK applies that substring rule DIFFERENTLY on the two calls a batch run
# makes. load_peaks (the time series) keeps EVERY batch whose name contains the
# string; samples.list (the roster) demands a UNIQUE match. A batch whose name is
# a prefix of its siblings' ("Site A Ur 122-600" beside "Site A Ur 122-600 11")
# therefore pooled three batches' peaks into one time series and then died at
# the roster with "Multiple batchs matching"; addressed by id, the roster
# resolved but the time series found nothing. resolve_batch() settles the batch
# ONCE -- exact id, else exact (casefolded) name, else a unique literal substring;
# anything ambiguous RAISES with the candidates -- and both fetchers go through
# it: the roster is asked by id, the time series by the resolved name with
# exact=True. The pool path (fetch_pooled_peaks) is the deliberate exception: a
# compiled regex there is MEANT to match many batches.


def list_workspaces() -> pd.DataFrame:
    """All workspaces the token can see, WITHOUT binding one — so it works as the
    first discovery step on a multi-workspace server (where building a
    workspace-scoped client would fail). Hits /api/workspaces via the SDK's raw
    http_get (verify_ssl defaults off, so a self-signed internal cert is fine).
    Powers `list workspaces`; columns include workspace_id / workspace_name."""
    from dotenv import load_dotenv
    from mascope_sdk._http import http_get
    load_dotenv(_find_env())
    url = os.environ.get("MASCOPE_URL")
    tok = os.environ.get("MASCOPE_ACCESS_TOKEN")
    if not url or not tok:
        raise RuntimeError("MASCOPE_URL / MASCOPE_ACCESS_TOKEN not found "
                           "(see `peaky setup`).")
    r = http_get(url, "workspaces", tok, timeout=(15, 60))
    data = (r.json() or {}).get("data", []) or []
    if not data:
        raise RuntimeError("no workspaces returned (check MASCOPE_URL / token)")
    return pd.DataFrame(data)


def list_datasets(client) -> pd.DataFrame:
    """All datasets visible to the token. Powers `list datasets` discovery."""
    ds = client.datasets.list()
    if ds is None or not len(ds):
        raise RuntimeError("no datasets returned (check MASCOPE_URL / token)")
    return ds


def list_batches(client, dataset: str | None = None) -> pd.DataFrame:
    """Sample batches (optionally in one dataset). Columns include
    `sample_batch_name` / `polarity` / `sample_batch_id` / `status`."""
    bs = client.batches.list(dataset=dataset)
    if bs is None or not len(bs):
        raise RuntimeError(
            f"no batches returned for dataset {dataset!r} (see `peaky list datasets`)")
    return bs


class ResolvedBatch(NamedTuple):
    """One batch, settled by `resolve_batch`: its server id, its display name, and
    which rule found it (`how`: 'id' / 'name' / 'substring'). `n_same_name` counts
    the batches in the listing that share its casefolded name -- 1 unless the
    dataset holds duplicates, which a loader that addresses batches by NAME
    (the SDK's load_peaks) cannot tell apart."""
    id: str
    name: str
    how: str
    n_same_name: int = 1


def _all_batches(client, dataset: str | None) -> pd.DataFrame:
    """The listing `resolve_batch` matches against: one dataset's batches when
    `dataset` is given (a name, substring or id -- the SDK settles it), else every
    dataset's, as the SDK's own samples.list does without a dataset. Only the
    listing calls are WAF-retried; the matching is not, because an error that
    quotes a batch name like 'HR-CIMS 100-500' would read as a transient 500."""
    if dataset is not None:
        bs = _with_waf_retry(lambda: client.batches.list(dataset=dataset))
        frames = [bs] if bs is not None and len(bs) else []
    else:
        ds = _with_waf_retry(lambda: client.datasets.list())
        if ds is None or not len(ds):
            raise RuntimeError("no datasets returned (check MASCOPE_URL / token)")
        frames = []
        for did in ds["dataset_id"].tolist():
            bs = _with_waf_retry(lambda did=did: client.batches.list(dataset=did))
            if bs is not None and len(bs):
                frames.append(bs)
    if not frames:
        where = f"dataset {dataset!r}" if dataset is not None else "any dataset"
        raise RuntimeError(f"no batches returned for {where} (see `peaky list batches`)")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def resolve_batch(client, batch: str, *, dataset: str | None = None) -> ResolvedBatch:
    """Settle `batch` -- a batch id or a batch name -- to ONE batch.

    Precedence: an exact `sample_batch_id`; else an exact case-insensitive name;
    else a unique case-insensitive LITERAL substring (the SDK's own rule, kept as
    the last resort). Zero matches, or several, RAISE with the candidates: the
    resolver never picks one of several and never hands back a set. Same
    precedence as curate's resolve_batch_id.

    The exact-name step is what the SDK lacks: 'Site A Ur 122-600' is a substring
    of its sibling 'Site A Ur 122-600 11', so on the SDK alone it either pools the
    two (load_peaks) or is refused (samples.list); here the exact name wins
    outright, and the caller passes the id / the exact name on to the SDK."""
    if not isinstance(batch, str):
        raise TypeError(f"batch must be a batch id or name (str), got "
                        f"{type(batch).__name__}; a compiled pattern is the pool "
                        "path (fetch_pooled_peaks)")
    bs = _all_batches(client, dataset)
    names = bs["sample_batch_name"].astype(str)
    folded = names.str.casefold()
    where = f"in dataset {dataset!r}" if dataset is not None else "in any dataset"

    def _one(row, how):
        name = str(row["sample_batch_name"])
        return ResolvedBatch(id=str(row["sample_batch_id"]), name=name, how=how,
                             n_same_name=int((folded == name.casefold()).sum()))

    hit = bs[bs["sample_batch_id"].astype(str) == batch]
    if len(hit) == 1:
        return _one(hit.iloc[0], "id")
    hit, how = bs[folded == batch.casefold()], "name"
    if not len(hit):
        # the SDK's substring rule, verbatim (_name_mask with exact=False)
        hit, how = bs[names.str.contains(re.escape(batch), case=False, na=False)], "substring"
    if len(hit) == 1:
        return _one(hit.iloc[0], how)
    if not len(hit):
        avail = sorted(names.tolist())
        shown = ", ".join(repr(n) for n in avail[:30]) + (" ..." if len(avail) > 30 else "")
        raise ValueError(f"No batch matching {batch!r} {where}. "
                         f"Available batches: {shown}")
    opts = "; ".join(f"{n!r} (id {i})" for n, i in
                     zip(hit["sample_batch_name"], hit["sample_batch_id"]))
    if how == "name":
        raise ValueError(f"{len(hit)} batches {where} are named {batch!r}: {opts}. "
                         "Pass the batch id (`peaky list batches` shows it).")
    raise ValueError(f"Multiple batches matching {batch!r} {where}: {opts}. Pass the "
                     "exact name or the batch id (`peaky list batches` shows both).")


def fetch_batch_samples(client, batch: str, *, dataset: str | None = None,
                        drop_columns=None) -> pd.DataFrame:
    """Per-sample table for a batch (one row per sample). Carries `sample_item_id`,
    `sample_item_name`, `datetime_utc`, `tic`, `polarity`, ... — the sample
    roster, WITHOUT loading every peak. (Cover selection needs the per-PEAK table;
    this one cannot be binned -- see sampling.is_per_peak.) `batch` is a batch id
    or name, settled by `resolve_batch`; the SDK is then asked by ID, so a name
    that is a prefix of a sibling's no longer trips its unique-substring rule."""
    rb = resolve_batch(client, batch, dataset=dataset)
    sl = client.samples.list(batch=rb.id, dataset=dataset,
                             drop_columns=[] if drop_columns is None else drop_columns)
    if sl is None or not len(sl):
        raise RuntimeError(f"no samples for batch {rb.name!r} (id {rb.id}) "
                           f"in dataset {dataset!r}")
    return sl


# Cloudflare-WAF / origin-5xx / read-timeout signatures that CLEAR on retry. A
# Mascope origin behind Cloudflare throws these under burst load (521/522 origin
# down, 403 "Attention Required" WAF challenge, 429 rate-limit, gateway 5xx, read
# timeouts). Anything else (404, 422, auth, client bugs) must NOT be retried --
# it re-raises immediately so the real failure surfaces.
_TRANSIENT_SIGNS = ("403", "429", "500", "502", "503", "504", "521", "522",
                    "attention required", "timed out", "timeout",
                    "temporarily unavailable", "bad gateway")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(sig in s for sig in _TRANSIENT_SIGNS)


def _with_waf_retry(fn, *, tries: int = 4, base_delay: float = 2.0):
    """Call ``fn()``; on a transient WAF/5xx/timeout error back off (base_delay·2ⁿ:
    2s, 4s, 8s) and retry up to ``tries`` times. A NON-transient error re-raises
    immediately so the real failure surfaces unmasked. Tests pass
    ``base_delay=0`` to avoid real sleeps."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc) or attempt == tries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            try:
                from loguru import logger
                logger.warning(
                    f"transient server error ({type(exc).__name__}); "
                    f"retry {attempt + 1}/{tries - 1} in {delay:.0f}s")
            except Exception:
                pass
            time.sleep(delay)


def fetch_batch_peaks(client, dataset: str, batch: str, *, save_path: str | None = None
                      ) -> pd.DataFrame:
    """Load the per-sample peak time-series for a whole batch (the TS / cluster /
    correlation layer). Distinct from fetch_peaks (one assignment sample). Uses the
    SDK batch loader (dataset=, not the deprecated workspace=). `batch` is a batch
    id or name, settled ONCE by `resolve_batch`; the loader is then asked for that
    exact name (exact=True), so a name that is a prefix of a sibling's selects one
    batch and never a silent pool of them. The loader addresses batches by NAME
    only, so a name several batches share is refused rather than pooled -- rename
    one of them, or hand the run a per-batch parquet."""
    rb = resolve_batch(client, batch, dataset=dataset)
    if rb.n_same_name > 1:
        raise RuntimeError(
            f"{rb.n_same_name} batches in dataset {dataset!r} are named {rb.name!r} "
            f"(id {rb.id} is one of them); the SDK time-series loader selects batches "
            "by name, so this batch's peaks cannot be isolated -- rename one of them, "
            "or pass the batch time series as a parquet (--ts)")
    # confirm_above=None: never prompt (non-interactive; batches can exceed 100
    # samples). exact=True: the name must EQUAL the batch's, not merely occur in
    # it. WAF-retry so a burst 521/403 doesn't kill a long batch load;
    # non-transient errors propagate unmasked.
    peaks = _with_waf_retry(
        lambda: client.load_peaks(dataset=dataset, batches=rb.name, exact=True,
                                  confirm_above=None))
    if peaks is None or len(peaks) == 0:
        raise RuntimeError(f"no peaks for batch {rb.name!r} (id {rb.id}) "
                           f"in dataset {dataset!r}")
    if save_path:
        peaks.to_parquet(os.path.expanduser(save_path))
    return peaks


def fetch_pooled_peaks(client, dataset: str, batches_regex: str, *,
                       save_path: str | None = None) -> pd.DataFrame:
    """Load the peak time-series of EVERY batch whose name matches `batches_regex`
    in ONE bulk call, pooling them into a single frame that keeps `sample_batch_name`
    so the caller can regroup. Unlike `fetch_batch_peaks`, the regex is compiled
    UN-escaped — the whole point is a multi-batch match (e.g.
    `'HR-CIMS 100-500.*zone'` pools the per-zone batches of one mode x range);
    the SDK treats a plain string as a literal substring, so only a compiled
    pattern keeps its regex meaning. WAF-retried; confirm_above=None so a large
    pool never prompts."""
    peaks = _with_waf_retry(
        lambda: client.load_peaks(dataset=dataset,
                                  batches=re.compile(batches_regex, re.IGNORECASE),
                                  confirm_above=None))
    if peaks is None or len(peaks) == 0:
        raise RuntimeError(
            f"no peaks matched /{batches_regex}/ in dataset {dataset!r}")
    if "sample_batch_name" not in peaks.columns:
        raise RuntimeError(
            "pooled load returned no 'sample_batch_name' column -- cannot regroup "
            f"(got {list(peaks.columns)[:8]})")
    if save_path:
        peaks.to_parquet(os.path.expanduser(save_path))
    return peaks


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------
# --- offline samples ----------------------------------------------------------
# A sample that exists only in this process: a synthetic table (one peak per
# batch trace, batch.tracefirst) or a fixture. Registered here it is served by
# fetch_peaks without a server, and with `client=None` the mechanism lookups
# resolve to the names themselves for the channels the sample declares -- so
# assign.run(..., peaks=frame) runs the whole engine on the local scorer with no
# round trip at all.
_OFFLINE: dict[str, tuple[pd.DataFrame, frozenset]] = {}
# What an offline sample is scored at, when it stands in for a measured one
# (`register_offline_sample(scoring=)`): see `scoring_for_sample`.
_OFFLINE_SCORING: dict = {}


def register_offline_sample(sample_id: str, peaks: pd.DataFrame, mechanisms=(), *,
                            scoring=None) -> None:
    """Serve `peaks` as `sample_id` from now on (this process only); `mechanisms`
    are the mechanism names its channels use ('[M-H]-', '[M+NO3]-', ...; the
    legacy '-H+' spelling names the same channel).

    `scoring` is what the sample is judged at. An offline sample has no server
    record to name its instrument and usually no server matches to fit a width
    from, so without it `scoring_for_sample` falls back to the more forgiving
    class (a TOF's) at zero offset -- wrong for a table that stands in for a
    measured sample, such as a decoy arm of a scored run. Pass the measured
    sample's `pattern_scoring` snapshot (a run's batch summary records one per
    sample) or a `PatternScoring` to be judged exactly as it was, or an
    instrument class ('orbi' / 'tof') to be judged at that class's width.
    Registering again replaces the table and the scoring and forgets any
    scoring computed for the old table."""
    _check_offline_scoring(scoring)
    if isinstance(scoring, str):
        scoring = scoring.strip().lower()
    _OFFLINE[sample_id] = (peaks, frozenset(mechanisms))
    _OFFLINE_SCORING[sample_id] = scoring
    _SCORING_CACHE.pop(sample_id, None)
    _SCORING_TREND.pop(sample_id, None)
    _SCORING_TREND_INHERITED.discard(sample_id)


def _check_offline_scoring(scoring) -> None:
    """Refuse a `register_offline_sample(scoring=)` that could not be scored at:
    a class other than 'orbi' / 'tof'; a snapshot or `PatternScoring` whose
    width, offset or window is not a finite real number (an unset width or
    offset of a `PatternScoring` is allowed: the library default / zero), whose
    width or window is not > 0, or whose abundance floor is not in [0, 1) (a
    snapshot may omit its floor: the library default); or any other type."""
    from numbers import Real

    from mascope_tools.composition import PatternScoring

    def num(v, name, *, optional=False):
        if v is None and optional:
            return None
        if isinstance(v, bool) or not isinstance(v, Real):        # text, None, a container: not a number
            raise ValueError(f"{name} is not a number: {v!r}")
        try:
            x = float(v)
        except (ValueError, OverflowError):
            raise ValueError(f"{name} is not a number: {v!r}") from None
        if not np.isfinite(x):
            raise ValueError(f"{name} is not finite: {v!r}")
        return x

    if scoring is None:
        return
    if isinstance(scoring, str):
        if scoring.strip().lower() not in ("orbi", "tof"):
            raise ValueError(f"an offline sample's instrument class is 'orbi' or 'tof', not {scoring!r}")
        return
    if isinstance(scoring, PatternScoring):
        sigma = num(scoring.sigma_ppm, "sigma_ppm", optional=True)
        num(scoring.mu_ppm, "mu_ppm", optional=True)
        window = num(scoring.mz_tolerance_ppm, "mz_tolerance_ppm")
        floor = num(scoring.abundance_floor, "abundance_floor")
    elif isinstance(scoring, dict):
        try:
            sigma = num(scoring.get("sigma_ppm"), "sigma_ppm")
            num(scoring.get("mu_ppm"), "mu_ppm")
            window = num(scoring.get("mz_tolerance_ppm"), "mz_tolerance_ppm")
            floor = num(scoring.get("abundance_floor"), "abundance_floor", optional=True)
        except ValueError as e:
            raise ValueError(f"a pattern_scoring snapshot to inherit needs finite numbers for sigma_ppm, "
                             f"mu_ppm and mz_tolerance_ppm, and abundance_floor if given: {e}") from None
    else:
        raise TypeError(f"an offline sample's scoring is a pattern_scoring snapshot, a PatternScoring or "
                        f"'orbi' / 'tof', not {type(scoring).__name__}")
    if (sigma is not None and sigma <= 0) or window <= 0:
        raise ValueError(f"an inherited width and window must be > 0: sigma {sigma}, window {window}")
    if floor is not None and not 0 <= floor < 1:
        raise ValueError(f"an inherited abundance floor is in [0, 1): {floor}")


def unregister_offline_sample(sample_id: str) -> None:
    _OFFLINE.pop(sample_id, None)
    _OFFLINE_SCORING.pop(sample_id, None)
    _SCORING_CACHE.pop(sample_id, None)
    _SCORING_TREND.pop(sample_id, None)
    _SCORING_TREND_INHERITED.discard(sample_id)


def is_offline_sample(sample_id: str) -> bool:
    return sample_id in _OFFLINE


def _offline_mechanisms() -> set:
    return set().union(*(m for _, m in _OFFLINE.values())) if _OFFLINE else set()


def fetch_peaks(client, sample_id: str, *, use_cache: bool = True,
                cache_root: Path = CACHE_ROOT) -> pd.DataFrame:
    """Pull the raw peak table (with Mascope's own matches flattened in) and
    cache it. Returns the full multi-row-per-peak frame; dedup is the ledger's
    job. An offline sample (`register_offline_sample`) is served from memory.

    The cache file is versioned because the peaks payload gained the per-peak
    `signal_to_noise` the scorer judges a faint line by: a frame cached before
    the server sent it has no such column, and silently scoring without it is
    the difference between charging an absent isotopologue and excusing it."""
    if sample_id in _OFFLINE:
        return _OFFLINE[sample_id][0].copy()
    cdir = Path(cache_root) / sample_id
    cfile = cdir / "peaks.v2.parquet"
    if use_cache and cfile.exists():
        return pd.read_parquet(cfile)
    peaks = client.samples.get_peaks(sample_id=sample_id, matches=True)
    if peaks is None or len(peaks) == 0:
        raise RuntimeError(f"no peaks returned for sample {sample_id!r}")
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        peaks.to_parquet(cfile)
    except Exception:
        peaks.to_csv(cdir / "peaks.csv", index=False)
    return peaks


def resolve_mechanism_ids(client, names: list[str]) -> dict[str, str]:
    """Map ionization-mechanism names ('[M-H]-', '[M+Br]-') to their ids.

    A row is matched by the mechanism it names, not by its spelling: a server
    before Mascope 1.10 stores the legacy '-H+' and one from 1.10 on the
    standard '[M-H]-', and both answer for '[M-H]-'. The keys of the result
    are the names as they were asked.

    With no client (an offline sample) a name is its own id, and only the
    channels the registered offline samples declare resolve -- matched by the
    same key, so a sample declared in either spelling opens its channels -- and
    the opportunistic extra channels stay closed, as on a server that does not
    list them."""
    if client is None:
        allowed = {_mechanism_key(m) for m in _offline_mechanisms()}
        return {n: n for n in names if _mechanism_key(n) in allowed}
    table = client.ionization.list()
    by_key = {_mechanism_key(r.ionization_mechanism): r.ionization_mechanism_id
              for r in table.itertuples()}
    out: dict[str, str] = {}
    for n in names:
        key = _mechanism_key(n)
        if key in by_key:
            out[n] = by_key[key]
    return out


# The adduct labels that have a server mechanism, each mapped to that
# mechanism's one spelling: the standard adduct notation, which Mascope stores
# and shows from 1.10 on and which the labels themselves are written in, so a
# value is its key canonicalised ('[M+(CH4N2O)H]+' is stored as
# '[M+CH4N2O+H]+'; test_io_mascope pins every value to `standard_notation`).
# A server before 1.10 stores the legacy '<operation><moiety><moiety charge>'
# spelling ('-H+' for '[M-H]-', '+' for '[M]+.'), so every lookup against a
# server row goes through `_mechanism_key`, which reads both, and never
# through this dict's keys or values directly.
ADDUCT_TO_MECH = {
    "[M-H]-": "[M-H]-",
    "[M+Br]-": "[M+Br]-",
    "[M+Cl]-": "[M+Cl]-",
    "[M+I]-": "[M+I]-",
    "[M+NO3]-": "[M+NO3]-",
    "[M+^NO3]-": "[M+^NO3]-",          # ¹⁵N-labelled nitrate reagent cluster
    "[M+HSO4]-": "[M+HSO4]-",
    "[M+Br2]-": "[M+Br2]-",
    "[M+Br3]-": "[M+Br3]-",
    "[M+I2]-": "[M+I2]-",
    "[M+I3]-": "[M+I3]-",
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+^NH4]+": "[M+^NH4]+",          # ¹⁵N-labelled ammonium reagent cluster
    "[M+CO3]-": "[M+CO3]-",
    "[M+(CH4N2O)H]+": "[M+CH4N2O+H]+",   # protonated-urea (uronium) adduct channel
    # bare molecular cation (EasyIC⁺ fluoranthene charge transfer). The hydride-
    # abstraction twin [M-H]+ is a local-scoring channel (LOCAL_ONLY_ADDUCT_MECH)
    # and stays off this map (the [M-H+I2]- ruling).
    "[M]+.": "[M]+.",
}
MECH_TO_ADDUCT = {v: k for k, v in ADDUCT_TO_MECH.items()}


def _mechanism_key(name) -> str:
    """The spelling a server's mechanism is compared by: the standard adduct
    notation whichever notation the row is stored in ('-H+' and '[M-H]-' are
    one key), and the text itself where it reads as neither, so an unreadable
    row matches nothing rather than raising (`mascope_tools.composition.mechanism_key`)."""
    from mascope_tools.composition import mechanism_key

    return mechanism_key(str(name))

# POSITIVE-MODE ABSTRACTION CHANNELS -- local-scoring only.
#
# An abstraction ion is the neutral minus a RADICAL (H·, CH3·): hydride
# abstraction [M-H]+ and methyl loss [M-CH3]+. No server before Mascope 1.10
# holds an ionization mechanism for either (a deployment's positive set is
# [M]+., [M+H]+, [M+Na]+, [M+NH4]+, [M+^NH4]+ plus its cluster reagents), so
# they are not in ADDUCT_TO_MECH, which lists the channels a server can name;
# they reach the scorer as tagged tokens instead (LOCAL_MECH_PREFIX). The
# collision that used to keep them out for good is gone with the standard
# notation - the legacy spelling wrote deprotonation '-H+' too, so [M-H]+ and
# [M-H]- shared a name - and Mascope 1.10 ships both mechanisms with the rest,
# so joining the map, and with it the server's ids and the publish path, is
# the follow-up once main runs against that release alone.
#
# The local scorer reads the standard notation, so the two channels are
# spelled to it exactly as their labels are, and it already computes them,
# isotope envelope included: on the 2026-09-22 certified mixture C5H8 on
# [M-H]+ matches C5H7+ @67.0542 at 0.05 ppm with its ¹³C line, and
# C8H24O4Si4 on [M-CH3]+ matches C7H21O4Si4+ @281.0509 with BOTH the ²⁹Si and
# ³⁰Si satellites.
#
# Until that run, the channels were unreachable rather than unscorable: the
# adducts were in a profile and in ADDUCT_SHIFTS, but the mechanism plumbing is
# keyed on server ids, so ADDUCT_TO_MECH filtered them out before the scorer
# ever saw them. Every hydride analyte therefore lost its peak to the
# mass-identical [M+H]+ reading of the neutral two hydrogens lighter (acetone
# read as propenal, isoprene as cyclopentadiene, hexanal as C6H10O), and the
# siloxanes' quantifier ion was named as a neutral that does not exist.
LOCAL_ONLY_ADDUCT_MECH = {
    "[M-H]+": "[M-H]+",
    "[M-CH3]+": "[M-CH3]+",
}

#: An abstraction channel travels through the pipeline INSIDE
#: ``cfg.mechanism_ids``, tagged with this prefix, instead of as a parallel
#: argument. Fourteen call sites already thread `mechanism_ids=cfg.mechanism_ids`
#: from every pass; a second parameter would have to be added to each of them
#: (and to the two that call score_candidates directly), and the one that got
#: missed would silently lose the channel again. The tag is stripped in the two
#: places ids reach the SERVER (`_server_mech_ids`) and translated to a mechanism
#: name in the one place they reach the local scorer (`_mechanism_names`).
LOCAL_MECH_PREFIX = "local:"


def local_mechanism_tokens(adducts: list[str]) -> list[str]:
    """Tagged mechanism tokens for the abstraction channels among `adducts`.
    Empty unless a profile actually asks for one."""
    return [LOCAL_MECH_PREFIX + LOCAL_ONLY_ADDUCT_MECH[a]
            for a in adducts if a in LOCAL_ONLY_ADDUCT_MECH]


def _server_mech_ids(mechanism_ids: list[str] | None) -> list[str] | None:
    """The subset of `mechanism_ids` that are real deployment ids -- what may be
    sent to match_compounds / the cheminfo candidate query."""
    if not mechanism_ids:
        return mechanism_ids
    out = [m for m in mechanism_ids if not str(m).startswith(LOCAL_MECH_PREFIX)]
    return out or None


def _local_mech_names(mechanism_ids: list[str] | None) -> list[str]:
    """The abstraction-channel mechanism names carried in `mechanism_ids`."""
    return [str(m)[len(LOCAL_MECH_PREFIX):]
            for m in (mechanism_ids or []) if str(m).startswith(LOCAL_MECH_PREFIX)]


def detect_adducts(peaks: pd.DataFrame) -> list[str]:
    """Infer the reagent/adduct system from the sample's own peak matches
    (the `ionization_mechanism` column). This is what makes a Br-CIMS sample
    get [M+Br]- offered as an interpretation instead of forcing Br into the
    neutral. Falls back to [M-H]- if nothing is recognised."""
    if peaks is None or "ionization_mechanism" not in peaks.columns:
        return ["[M-H]-"]
    out: list[str] = []
    for name in peaks["ionization_mechanism"].dropna().unique():
        a = MECH_TO_ADDUCT.get(_mechanism_key(name))
        if a and a not in out:
            out.append(a)
    return out or ["[M-H]-"]


def sample_mass_errors(peaks: pd.DataFrame, *,
                       max_abs_ppm: float = 10.0) -> list[float]:
    """The ppm mass errors of the sample's OWN server matches (base ions only).

    The sample's anchors: peaks Mascope already attributed to a known species,
    whose error against the theoretical mass is a measurement of this run's mass
    accuracy rather than of any assignment peaky makes. One collection feeds both
    readings of them - the rough offset the pre-calibration gates need, and the
    (mu, sigma) the fit score is judged at - so the two cannot drift apart.

    :param peaks: The raw peaks frame, matches flattened in.
    :param max_abs_ppm: Gross-outlier guard. The default is a compromise for the
        offset; a caller that knows the instrument class should pass its
        matching window, since an anchor outside that window is not a match.
    :return: The errors in ppm, in the frame's order.
    """
    from peaky.chem import chemistry as C
    cols = {"target_compound_formula", "ionization_mechanism", "mz"}
    if peaks is None or not cols <= set(peaks.columns):
        return []
    iso_col = "target_isotope_formula" in peaks.columns
    ppms: list[float] = []
    for r in peaks.dropna(subset=["target_compound_formula", "mz",
                                  "ionization_mechanism"]).itertuples():
        if iso_col and "[" in str(getattr(r, "target_isotope_formula", "") or ""):
            continue                                 # heavy-isotope row, skip
        add = MECH_TO_ADDUCT.get(_mechanism_key(r.ionization_mechanism))
        if not add or add not in C.ADDUCT_SHIFTS:
            continue
        try:
            theo = C.ion_mz(str(r.target_compound_formula), add)
        except Exception:
            continue
        p = (float(r.mz) - theo) / theo * 1e6
        if abs(p) <= max_abs_ppm:
            ppms.append(p)
    return ppms


def estimate_offset(peaks: pd.DataFrame, *, min_n: int = 8) -> float | None:
    """Rough median ppm mass-offset from the sample's OWN server matches (base
    ions only). The pass-1 self-calibration is the authoritative fit, but it runs
    AFTER pass 0 / pass 1 -- so a source with a large systematic offset (the
    instrument sits at e.g. -1.9 ppm) is blind to it in pass 0's |ppm|<=2 known-
    species gate, which then drops real contaminants whose on-trend mass is just
    past 2 ppm and lets pass 1 grab the peak with an off-trend mass-coincidence.
    This seeds those pre-calibration gates. None when too few matches to trust."""
    ppms = sorted(sample_mass_errors(peaks))
    if len(ppms) < min_n:
        return None
    n = len(ppms)
    return (ppms[n // 2] if n % 2 else (ppms[n // 2 - 1] + ppms[n // 2]) / 2)


def instrument_type_for(sample: dict | None) -> str | None:
    """The sample's instrument class, 'orbi' or 'tof'.

    The record's own field first - the class the reader wrote when it converted
    the file - then the instrument name and the file name, which is the order
    Mascope itself resolves it in. None when nothing says, and the caller then
    takes the more forgiving of the two class widths.
    """
    from peaky.io.publish import instrument_type_from_filename, resolve_instrument_type

    record = sample or {}
    declared = str(record.get("instrument_type") or "").strip().lower()
    if declared in ("orbi", "tof"):
        return declared
    if record.get("instrument"):
        kind = resolve_instrument_type(str(record["instrument"]))
        if kind:
            return kind
    if record.get("filename"):
        return instrument_type_from_filename(str(record["filename"]))
    return None


def scoring_for_sample(client, sample_id: str, peaks: pd.DataFrame | None = None,
                       *, refresh: bool = False):
    """What this sample's candidates are scored at: a `PatternScoring`.

    Three statements about the sample, each read the way Mascope's own engine
    reads it (`engine.pattern_scoring_for`), so that a reference run and an
    in-app run judge the same spectrum by the same measurement:

      * the width - the sample's own anchors fitted by the library's
        `fit_mass_accuracy`, widened for prediction and centroiding, falling
        back to the instrument class where too few anchors matched to measure
        anything. This is what a TOF needed: at the Orbitrap-shaped default a
        TOF's every candidate is several sigma out and scores near zero.
      * the offset - the same anchors' median, subtracted before a mass error is
        scored, so an instrument sitting at -1.5 ppm does not charge it to every
        candidate.
      * the window a predicted line may be matched in, the instrument class's.

    Cached per sample: the passes score many batches against one sample, and the
    fit is a property of the sample rather than of the batch.

    The pre-calibration pass gates keep their own, more conservative rule
    (`estimate_offset`, eight matches before it states an offset at all). They
    decide which candidates are considered rather than how a considered one
    scores, and widening that is a change to the search.

    An offline sample has no record and usually no anchors: registered with a
    `scoring` (`register_offline_sample`) it is judged at that -- a measured
    sample's snapshot or `PatternScoring` as it is (`sigma_source` /
    `mu_source` "inherited"), an instrument class as that class's width.
    """
    from mascope_tools.composition import (
        PatternScoring,
        fit_mass_accuracy,
        resolve_fallback_sigma_ppm,
        resolve_match_tolerance_ppm,
        scoring_sigma_ppm,
    )

    if not refresh and sample_id in _SCORING_CACHE:
        return _SCORING_CACHE[sample_id][0]
    given = _OFFLINE_SCORING.get(sample_id) if sample_id in _OFFLINE else None
    if given is not None and not isinstance(given, str):
        raw = fetch_peaks(client, sample_id) if peaks is None else peaks
        scoring, snapshot = _inherited_scoring(given, raw)
        _SCORING_CACHE[sample_id] = (scoring, snapshot)
        if isinstance(given, dict) and given.get("trend"):
            # the measured sample was scored at its mass trend: so is its stand-in
            _SCORING_TREND[sample_id] = _trend_from_record(given["trend"])
            _SCORING_TREND_INHERITED.add(sample_id)
        return scoring
    try:
        record = client.samples.get(sample_id)
    except Exception:
        record = None
    # an offline sample registered with an instrument class is judged at it
    kind = (given if isinstance(given, str)
            else instrument_type_for(record if isinstance(record, dict) else None))
    window = resolve_match_tolerance_ppm(kind)
    raw = fetch_peaks(client, sample_id) if peaks is None else peaks
    # An anchor outside the matching window is not a match at this instrument,
    # so it is not evidence about its accuracy either.
    anchors = sample_mass_errors(raw, max_abs_ppm=window)
    mu, sigma = fit_mass_accuracy(anchors)
    if sigma is None and len(anchors) >= MIN_OFFSET_ANCHORS:
        # The fit reports no width and, with it, no offset. The width is
        # genuinely unmeasurable here; the offset is not.
        mu = float(np.median(anchors))
    if mu is None:
        # The library answers None for an offset it did not measure, and a
        # sample can have too few anchors even for the rule above - a TOF file
        # whose matches were just cleared has none at all. Scoring at zero is
        # the only honest reading of "no offset measured", and the snapshot
        # says `assumed_zero` so nothing downstream reads it as a measurement.
        mu = 0.0
    scoring = PatternScoring(
        sigma_ppm=scoring_sigma_ppm(sigma, resolve_fallback_sigma_ppm(kind)),
        mu_ppm=mu,
        mz_tolerance_ppm=window,
    )
    snapshot = {
        "score_version": SCORE_VERSION,
        "sigma_ppm": round(float(scoring.sigma_ppm), 4),
        "mu_ppm": round(float(scoring.mu_ppm), 4),
        # "fitted" means this sample measured its own width; "instrument_class"
        # means too few known ions matched to fit one and the class stood in.
        "sigma_source": "fitted" if sigma is not None else "instrument_class",
        # The offset stands on its own: a sample can measure one and not a width.
        "mu_source": (
            "fitted" if len(anchors) >= MIN_OFFSET_ANCHORS else "assumed_zero"
        ),
        "fitted_anchors": len(anchors),
        "mz_tolerance_ppm": float(scoring.mz_tolerance_ppm),
        "abundance_floor": float(scoring.abundance_floor),
        "instrument_type": kind,
        # Whether any peak of this sample actually carries an estimate, not
        # whether the column is there: the endpoint sends the column with nulls
        # for a file that stores none, and the two are the difference between a
        # score that charges an absent line by the noise and one that charges it
        # by abundance alone.
        "has_signal_to_noise": bool(
            pd.to_numeric(raw["signal_to_noise"], errors="coerce").notna().any()
        )
        if "signal_to_noise" in getattr(raw, "columns", [])
        else False,
        **_snr_policy_keys(sample_id, raw),
    }
    _SCORING_CACHE[sample_id] = (scoring, snapshot)
    return scoring


# re-exported for callers that read a snapshot's `snr_source`
from peaky.io.local_scoring import (  # noqa: E402
    SNR_SOURCE_NONE, SNR_SOURCE_POISSON, SNR_SOURCE_SERVER,
)


def _snr_policy_keys(sample_id, raw) -> dict:
    """The snapshot's record of WHAT signal-to-noise the sample's lines are
    judged at (C46): `snr_source` ("server" -- the table's own column; "none"
    -- no column, the score's no-SNR mode; "poisson_fallback" -- the column
    does not track height and the counting-statistics SNR h/sqrt(h+edge^2)
    stands in), `snr_spearman` (height vs the column, over the file's peaks),
    `snr_n` (peaks judged) and `snr_edge` (the detection edge the fallback
    uses). `peaks_for_scoring` reads them back."""
    from loguru import logger

    from peaky.io import local_scoring

    a = local_scoring.assess_snr(raw)
    if a["source"] == SNR_SOURCE_POISSON:
        logger.info("sample {}: the peak table's signal_to_noise does not track height "
                    "(Spearman {} over {} peaks) -- lines judged at the counting-statistics "
                    "SNR h/sqrt(h + edge^2), edge {}",
                    sample_id or "<offline>", a["spearman"], a["n"],
                    None if a["edge"] is None else round(a["edge"], 4))
    return {"snr_source": a["source"], "snr_spearman": a["spearman"],
            "snr_n": a["n"], "snr_edge": a["edge"]}


def peaks_for_scoring(sample_id: str, raw: pd.DataFrame) -> pd.DataFrame:
    """`raw` as the local scorer should read it: its own table, or a copy
    whose `signal_to_noise` is the counting-statistics SNR where the sample's
    snapshot says the column is not one (`snr_source` "poisson_fallback").
    Before `scoring_for_sample` has judged the sample, the table as it is."""
    from peaky.io import local_scoring

    snap = _SCORING_CACHE.get(sample_id, (None, None))[1] or {}
    if snap.get("snr_source") != SNR_SOURCE_POISSON:
        return raw
    return local_scoring.with_poisson_snr(raw, snap.get("snr_edge"))


def _has_snr(raw) -> bool:
    return (bool(pd.to_numeric(raw["signal_to_noise"], errors="coerce").notna().any())
            if "signal_to_noise" in getattr(raw, "columns", []) else False)


def _inherited_scoring(given, raw):
    """(PatternScoring, snapshot) for an offline sample judged at another
    sample's measurement: `given` is that sample's `pattern_scoring` snapshot
    or a `PatternScoring`. The snapshot keeps the measurement's numbers and
    says it was inherited (`sigma_source` / `mu_source`, the originals under
    `inherited`); whether peaks carry a signal-to-noise is this table's own."""
    from mascope_tools.composition import PatternScoring

    if isinstance(given, PatternScoring):
        scoring, src = given, {"sigma_source": None, "mu_source": None}
        if scoring.sigma_ppm is None or scoring.mu_ppm is None:
            # an unset width is the library's own default, as score_pattern_v2 reads it; an
            # unset offset is none measured, zero
            from mascope_tools.composition.heuristic_filter import FALLBACK_SIGMA_PPM
            scoring = PatternScoring(
                sigma_ppm=FALLBACK_SIGMA_PPM if scoring.sigma_ppm is None else scoring.sigma_ppm,
                mu_ppm=0.0 if scoring.mu_ppm is None else scoring.mu_ppm,
                mz_tolerance_ppm=scoring.mz_tolerance_ppm, abundance_floor=scoring.abundance_floor)
    else:
        src = dict(given)
        kw = {"sigma_ppm": float(src["sigma_ppm"]), "mu_ppm": float(src["mu_ppm"]),
              "mz_tolerance_ppm": float(src["mz_tolerance_ppm"])}
        if src.get("abundance_floor") is not None:
            kw["abundance_floor"] = float(src["abundance_floor"])
        scoring = PatternScoring(**kw)
    snapshot = {
        "score_version": SCORE_VERSION,
        "sigma_ppm": round(float(scoring.sigma_ppm), 4),
        "mu_ppm": round(float(scoring.mu_ppm), 4),
        "sigma_source": "inherited",
        "mu_source": "inherited",
        **({"trend": dict(src["trend"])} if src.get("trend") else {}),
        "inherited": {"sigma_source": src.get("sigma_source"), "mu_source": src.get("mu_source"),
                      "fitted_anchors": src.get("fitted_anchors"),
                      "has_signal_to_noise": src.get("has_signal_to_noise"),
                      "snr_source": src.get("snr_source")},
        "fitted_anchors": 0,
        "mz_tolerance_ppm": float(scoring.mz_tolerance_ppm),
        "abundance_floor": float(scoring.abundance_floor),
        "instrument_type": src.get("instrument_type"),
        "has_signal_to_noise": _has_snr(raw),
        **_snr_policy_keys(None, raw),
    }
    return scoring, snapshot


def scoring_snapshot(client, sample_id: str, peaks: pd.DataFrame | None = None) -> dict:
    """What a run records about the width it judged a mass error against.

    The same keys the in-app engine stamps on its own run config, so a reader
    holding both runs of one sample can see whether they were judged alike -
    which is the first question about any disagreement between them.
    """
    scoring_for_sample(client, sample_id, peaks)
    return dict(_SCORING_CACHE[sample_id][1])


def describe_scoring(scoring, *, instrument_type: str | None = None) -> str:
    """One line naming what a sample was judged at, for the run log."""
    return (f"sigma={scoring.sigma_ppm:.2f} ppm mu={scoring.mu_ppm:+.2f} ppm "
            f"window={scoring.mz_tolerance_ppm:.0f} ppm"
            + (f" instrument={instrument_type}" if instrument_type else ""))


# ---------------------------------------------------------------------------
# Candidate enumeration (cheminfo)
# ---------------------------------------------------------------------------
def query_candidates(client, mz: float, mechanism_ids: list[str], *,
                     formula_ranges: str, ppm: float = 5.0,
                     limit: int = 25) -> list[str]:
    """Return candidate NEUTRAL formulas for one m/z (deduped).

    cheminfo is a flaky endpoint (timeouts / 500s). A failure here must NOT kill
    the run: the local grid covers the same CHO/CHON formula space, so degrade
    to [] on any error and let the grid carry that m/z."""
    try:
        res = client.cheminfo.query_by_mz(
            mz=mz, ionization_mechanism_ids=_server_mech_ids(mechanism_ids),
            formula_ranges=formula_ranges, mz_tolerance=float(ppm), limit=limit) or []
    except Exception:
        return []
    out = []
    seen = set()
    for r in res:
        f = r.get("target_compound_formula")
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def query_candidates_bulk(client, mzs: list[float], mechanism_ids: list[str], *,
                          formula_ranges: str, ppm: float = 5.0, limit: int = 25,
                          workers: int = 12) -> dict[float, list[str]]:
    """Parallel cheminfo enumeration over many m/z."""
    def _one(mz):
        return mz, query_candidates(client, mz, mechanism_ids,
                                    formula_ranges=formula_ranges, ppm=ppm, limit=limit)
    out: dict[float, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for mz, cands in ex.map(_one, mzs):
            out[mz] = cands
    return out


# ---------------------------------------------------------------------------
# Scoring oracle (match_compounds) + PURE parser
# ---------------------------------------------------------------------------
def parse_isotope_label(isotope_formula: str) -> tuple[str, bool]:
    """('[13C]C3H5O2-') -> ('13C', False). Base (no heavy isotope) -> ('M0', True)."""
    toks = _ISO_TOKEN.findall(isotope_formula or "")
    if not toks:
        return "M0", True
    parts = []
    for sym, n in toks:
        parts.append(sym + (n if n and int(n) > 1 else ""))
    return "+".join(parts), False


def flatten_match_tree(tree: list[dict]) -> pd.DataFrame:
    """PURE: flatten match_compounds output into one row per
    (compound, ion, isotopologue). No network. Columns:

      compound_formula, compound_score, compound_category,
      ion_formula, ion_score, ion_category, mechanism_id,
      isotope_formula, iso_label, is_base, theo_mz, rel_abundance,
      iso_score, iso_category, sample_peak_id, sample_peak_mz,
      sample_peak_intensity, ppm_error, abundance_error
    """
    rows: list[dict] = []
    for comp in tree or []:
        cf = comp.get("target_compound_formula")
        cs = comp.get("match_score")
        cc = comp.get("match_category")
        for ion in comp.get("children", []) or []:
            ifl = ion.get("target_ion_formula")
            isc = ion.get("match_score")
            icat = ion.get("match_category")
            mech = ion.get("ionization_mechanism_id")
            ion_rows: list[dict] = []
            for iso in ion.get("children", []) or []:
                iso_f = iso.get("target_isotope_formula")
                label, is_base = parse_isotope_label(iso_f)
                theo = iso.get("mz")
                spk_mz = iso.get("sample_peak_mz")
                spk_int = iso.get("sample_peak_intensity") or 0.0
                spid = iso.get("sample_peak_id") or None
                # ppm is only meaningful for a GENUINELY matched isotope (a real
                # attributed peak). Unmatched/forced nodes carry sample_peak_mz ==
                # theoretical mz and zero intensity -> leave ppm undefined.
                matched = bool(spid) and float(spk_int) > 0 and spk_mz and float(spk_mz) > 0
                if matched and theo:
                    ppm = (float(spk_mz) - float(theo)) / float(theo) * 1e6
                else:
                    ppm = None
                ion_rows.append({
                    "compound_formula": cf, "compound_score": cs, "compound_category": cc,
                    "ion_formula": ifl, "ion_score": isc, "ion_category": icat,
                    "mechanism_id": mech,
                    "isotope_formula": iso_f, "iso_label": label, "is_base": is_base,
                    "theo_mz": theo, "rel_abundance": iso.get("relative_abundance"),
                    "iso_score": iso.get("match_score"), "iso_category": iso.get("match_category"),
                    "sample_peak_id": spid, "sample_peak_mz": spk_mz,
                    "sample_peak_intensity": iso.get("sample_peak_intensity"),
                    "ppm_error": ppm, "abundance_error": iso.get("match_abundance_error"),
                })
            # ISOTOPICALLY-LABELLED REAGENT (^N = ¹⁵N nitrate): the server models the
            # reagent N as natural-abundance, so it tags the all-light form as the M0
            # base (a PHANTOM at the ¹⁴N mass with NO signal) and the single-¹⁵N line
            # -- the ACTUAL monoisotopic ion, since the reagent is 100% ¹⁵N -- as a
            # '15N' isotopologue (is_base=False). Re-anchor the base onto that line so
            # the assignment passes (which commit only is_base peaks) see the real peak.
            if "^N" in str(ifl or ""):
                _reanchor_labelled_reagent(ion_rows, delta=0.997035, label="15N")
            rows.extend(ion_rows)
    return pd.DataFrame(rows)


def _reanchor_labelled_reagent(ion_rows: list[dict], *, delta: float, label: str) -> None:
    """Move is_base from the (phantom, signal-less) all-light M0 onto the single
    heavy-reagent-isotope line for a 100%-labelled reagent. In place. No-op unless
    the all-light base exists and a matching heavy line sits delta higher."""
    base = next((r for r in ion_rows if r.get("is_base")), None)
    if base is None or base.get("theo_mz") in (None, 0):
        return
    target = float(base["theo_mz"]) + delta
    cand = [r for r in ion_rows
            if r.get("iso_label") == label and r.get("theo_mz")]
    if not cand:
        return
    new_base = min(cand, key=lambda r: abs(float(r["theo_mz"]) - target))
    if abs(float(new_base["theo_mz"]) - target) > 0.01:
        return
    base["is_base"] = False
    new_base["is_base"] = True
    new_base["iso_label"] = "M0"


# concurrent match_compounds batches per process (I/O-bound; server-safe). Read
# from the env so the batch process pool can DIVIDE it across worker processes
# (N_procs * PEAKY_MATCH_WORKERS stays a bounded total load on the flaky server).
MATCH_WORKERS = max(1, int(os.environ.get("PEAKY_MATCH_WORKERS", "5")))


def _polarity_sign(pol) -> str | None:
    """Server polarity field -> '+' / '-' (tolerates '+'/'-', 'pos'/'neg', ±1)."""
    s = str(pol).strip().lower()
    if s in ("+", "pos", "positive", "1", "+1"):
        return "+"
    if s in ("-", "neg", "negative", "-1"):
        return "-"
    return None


def _mechanism_names(client, mechanism_ids: list[str] | None) -> list[str]:
    """Reverse-map resolved ionization-mechanism ids -> the mechanism strings
    the local scorer is handed: the standard adduct notation, whichever
    notation the server stores.

    A server before Mascope 1.10 stores the legacy spelling, whose trailing
    sign is the added or removed species' charge rather than the ion's: '-H+'
    (a proton removed) is the anion '[M-H]-', and '-H-' (a hydride removed) the
    cation '[M-H]+'. The library reads that spelling by its grammar since the
    release this branch pins, so the rewrite this function used to do - flip
    the trailing sign to the row's polarity - would now turn a deprotonation
    into a hydride abstraction; the row is converted with `standard_notation`
    instead, which is exact on every spelling the fleet stores. A row whose
    polarity column contradicts its spelling is read by the spelling, as the
    server reads it, and logged; a row that reads as neither notation is
    skipped, since the scorer could not parse it either."""
    if not mechanism_ids:
        return []
    # abstraction channels arrive already spelled for the local scorer (they have
    # no deployment id to reverse-map); they are the whole point of the tag.
    out_local = _local_mech_names(mechanism_ids)
    mechanism_ids = [m for m in mechanism_ids
                     if not str(m).startswith(LOCAL_MECH_PREFIX)]
    if not mechanism_ids:
        return out_local
    from loguru import logger
    from mascope_tools.composition import parse_mechanism
    from mascope_tools.composition.mechanism_notation import MechanismNotationError

    if client is None:
        # offline: the ids ARE the names, converted like a server row's (no
        # polarity column to cross-check): '-H+' is the anion '[M-H]-' under the
        # library's grammar, so the old sign flip to '-H-' would be a hydride loss
        out = []
        for m in mechanism_ids:
            try:
                std = parse_mechanism(str(m)).standard
            except MechanismNotationError as e:
                logger.warning("offline ionization mechanism {!r} is not scored "
                               "locally: {}", m, e)
                continue
            if std not in out:
                out.append(std)
        return out + [m for m in out_local if m not in out]
    table = client.ionization.list()
    id2 = {
        r.ionization_mechanism_id: (
            r.ionization_mechanism,
            r.ionization_mechanism_polarity,
        )
        for r in table.itertuples()
    }
    out = []
    for m in mechanism_ids:
        if m not in id2:
            continue
        name, pol = id2[m]
        try:
            parts = parse_mechanism(str(name))
        except MechanismNotationError as e:
            logger.warning("ionization mechanism {!r} (id {}) is not scored "
                           "locally: {}", name, m, e)
            continue
        sign = _polarity_sign(pol)
        if sign and sign != parts.polarity:
            logger.warning("ionization mechanism {!r} (id {}) is stored with "
                           "polarity {!r} but reads as {}; scored as it reads",
                           name, m, pol, parts.standard)
        out.append(parts.standard)
    return out + [m for m in out_local if m not in out]


def _local_scoring_enabled() -> bool:
    """Local scoring is the default; PEAKY_LOCAL_SCORING=0/false/no/off opts back
    to the network match_compounds path (the escape hatch)."""
    v = os.environ.get("PEAKY_LOCAL_SCORING")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _score_candidates_local(client, sample_id, formulas, mechanism_ids):
    """Local, in-process scoring (no match_compounds round-trip) producing the same
    flat per-isotopologue schema as the backend path. Peaks from the cached
    fetch_peaks; channels from the reverse-mapped mechanism names; the sample's
    own fitted width, offset and class window from `scoring_for_sample`."""
    from peaky.io import local_scoring

    raw = fetch_peaks(client, sample_id)          # cached; mz/height/peak_id/snr
    mechs = _mechanism_names(client, mechanism_ids)
    # The window is the instrument class's, so PEAKY_MATCH_PPM is gone: an
    # operator setting 15 for a TOF was saying what the class already knows,
    # and nothing said it for the width the mass is then scored against.
    scoring = scoring_for_sample(client, sample_id, raw)
    out = local_scoring.score_candidates_local(
        peaks_for_scoring(sample_id, raw), formulas, mechanisms=mechs,
        scoring=scoring,
        centre=scoring_trend(sample_id),
    )
    out.attrs["match_batches"] = 0
    out.attrs["match_batch_failures"] = []
    out.attrs["match_formulas"] = len(formulas)
    return out


def score_candidates(client, sample_id: str, formulas: list[str], *,
                     match_params: dict | None = None,
                     mechanism_ids: list[str] | None = None,
                     batch: int = MATCH_BATCH,
                     workers: int = MATCH_WORKERS,
                     allow_partial: bool = False) -> pd.DataFrame:
    """Score candidate NEUTRAL formulas against the sample. Returns the flat
    per-isotopologue table (see flatten_match_tree). Batches are scored
    CONCURRENTLY -- match_compounds is network-bound, so the wall-clock for a
    many-batch pass (e.g. a wide heteroatom family) scales with the worker
    count, not the batch count.

    By default, any failed batch raises. A partial candidate universe is worse
    than a failed pass: it can make the ledger look clean while alternatives
    were never scored. Set allow_partial=True only for exploratory tooling; the
    returned frame then carries failure details in ``frame.attrs``.
    """
    formulas = sorted({f for f in formulas if f})
    if not formulas:
        return pd.DataFrame()
    # Local in-process scoring (mascope_tools) is now the DEFAULT: same scoring
    # maths run locally, ~2x faster, no 30k-row match trees, no timeout/OOM. The
    # network match_compounds path below stays as an escape hatch -- disable local
    # with PEAKY_LOCAL_SCORING=0 (or false/no/off). Validated full-pipeline on
    # Bromide (0.932 agreement) + Uronium; see docs/MASCOPE_TOOLS_INTEGRATION.md.
    if _local_scoring_enabled():
        return _score_candidates_local(client, sample_id, formulas, mechanism_ids)
    if _local_mech_names(mechanism_ids):
        # The escape hatch cannot reach the abstraction channels: they exist
        # only as local-scorer mechanism names (LOCAL_ONLY_ADDUCT_MECH). Say so
        # rather than silently dropping a profile's channel -- that silence is
        # what hid the hydride gap through eleven EasyIC batches.
        from loguru import logger
        logger.warning(
            "PEAKY_LOCAL_SCORING is off: the abstraction channels {} have no "
            "server mechanism and will NOT be scored this run",
            ", ".join(sorted(_local_mech_names(mechanism_ids))),
        )
    mechanism_ids = _server_mech_ids(mechanism_ids)
    mp = dict(DEFAULT_MATCH_PARAMS)
    if match_params:
        mp.update(match_params)
    mp["mz_tolerance"] = int(round(mp.get("mz_tolerance", 5)))  # server needs int ppm
    chunks = [formulas[i:i + batch] for i in range(0, len(formulas), batch)]

    def _score(chunk):
        try:
            tree = client.matching.match_compounds(
                sample_id=sample_id, formulas=chunk, match_params=mp,
                ionization_mechanism_ids=mechanism_ids)
        except Exception as e:
            return pd.DataFrame(), {
                "n_formulas": len(chunk),
                "first_formula": chunk[0] if chunk else None,
                "last_formula": chunk[-1] if chunk else None,
                "error_type": type(e).__name__,
                "error": str(e),
            }
        return flatten_match_tree(tree or []), None

    if len(chunks) == 1:
        results = [_score(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
            results = list(ex.map(_score, chunks))
    frames = [r[0] for r in results]
    failures = [r[1] for r in results if r[1] is not None]
    if failures and not allow_partial:
        first = failures[0]
        raise RuntimeError(
            "match_compounds failed for "
            f"{len(failures)}/{len(chunks)} batches "
            f"({sum(f['n_formulas'] for f in failures)} formulas); "
            f"first failed chunk {first['first_formula']}..{first['last_formula']}: "
            f"{first['error_type']}: {first['error']}"
        )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["match_batches"] = len(chunks)
    out.attrs["match_batch_failures"] = failures
    out.attrs["match_formulas"] = len(formulas)
    return out
