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

import pandas as pd

__version__ = "0.3.0"  # + legacy workspace-server support (connect health-check
#                          bypass, raw sample/batches resolution, list fallbacks)

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


def _patch_datasets_list_for_legacy_servers() -> None:
    """Tolerate OLDER Mascope builds that predate the 'datasets' concept.

    On a legacy server the top-level object is a WORKSPACE and there is no
    ``/api/datasets`` endpoint, so ``DatasetsResource.list`` 404s. That is fatal
    because ``MascopeClient.__init__`` calls ``datasets.list()`` as an early
    health check -> the client can't even be constructed. Wrap the method so a
    NotFoundError degrades to ``None``; the client then builds and the legacy
    workspace/batch resolution below (``_legacy_*`` / ``resolve_batch_id``) uses
    the raw endpoints that DO exist (``workspaces`` / ``sample/batches`` /
    ``samples?sample_batch_id=``). No-op on a modern server (datasets.list works
    normally). Idempotent."""
    try:
        from mascope_sdk.resources.datasets import DatasetsResource
        from mascope_sdk.exceptions import NotFoundError
    except Exception:
        return
    if getattr(DatasetsResource.list, "_legacy_safe", False):
        return
    _orig = DatasetsResource.list

    def list(self):                       # noqa: A001  (mirror SDK method name)
        try:
            return _orig(self)
        except NotFoundError:
            return None
    list._legacy_safe = True
    DatasetsResource.list = list


def escape_batch(name: str) -> str:
    """The SDK resolves `batch=`/`batches=` as a case-insensitive REGEX via
    str.contains, so a literal batch name with regex metacharacters (e.g.
    'Sample run (Ur+ CIMS)' — parens + '+') silently fails to match. Escape
    it to match literally."""
    return re.escape(name)


# ---------------------------------------------------------------------------
# Legacy (workspace-based) server support
# ---------------------------------------------------------------------------
# Older Mascope builds expose `workspaces` + `sample/batches` + `samples` but NOT
# `datasets`, and `sample/batches` IGNORES its dataset_id param (returns every
# batch). These helpers hit those raw endpoints via the SDK's own http_get so the
# pipeline works on both server generations. `dataset` in this package's public
# API maps to a WORKSPACE name on a legacy server.
def _legacy_get(client, path: str, **params) -> list[dict]:
    from mascope_sdk._http import http_get
    r = http_get(client.url, path, client.access_token,
                 params=params or None, timeout=(15, 60))
    return (r.json() or {}).get("data", []) or []


def _legacy_workspaces(client) -> pd.DataFrame:
    return pd.DataFrame(_legacy_get(client, "workspaces"))


def _legacy_all_batches(client) -> pd.DataFrame:
    """Every sample batch on a legacy server (the dataset_id filter is ignored).
    Columns: workspace_id, sample_batch_name, sample_batch_id, polarity, status,
    sample_batch_utc_created, ..."""
    return pd.DataFrame(_legacy_get(client, "sample/batches"))


def resolve_batch_id(client, batch: str, *, dataset: str | None = None) -> str:
    """Legacy-server batch-name -> sample_batch_id. Exact case-insensitive name
    match first, then substring; if `dataset` is given it is treated as a
    WORKSPACE name and constrains the search. Raises with the candidate list on
    no/ambiguous match (so a CLI user can disambiguate)."""
    bdf = _legacy_all_batches(client)
    if bdf is None or not len(bdf):
        raise RuntimeError("no batches returned from /api/sample/batches")
    if dataset:
        ws = _legacy_workspaces(client)
        wid = ws.loc[ws["workspace_name"].str.casefold() == dataset.casefold(),
                     "workspace_id"]
        if len(wid):
            bdf = bdf[bdf["workspace_id"] == wid.iloc[0]]
    names = bdf["sample_batch_name"].astype(str)
    key = batch.casefold()
    hit = bdf[names.str.casefold() == key]
    if not len(hit):
        hit = bdf[names.str.casefold().str.contains(re.escape(key))]
    if len(hit) == 1:
        return str(hit.iloc[0]["sample_batch_id"])
    if not len(hit):
        avail = ", ".join(sorted(names)[:30])
        raise RuntimeError(f"no batch matching {batch!r}. Available (first 30): {avail}")
    opts = "; ".join(f"{r.sample_batch_name!r} [{r.polarity}] ws={r.workspace_id}"
                     for r in hit.itertuples())
    raise RuntimeError(
        f"{len(hit)} batches match {batch!r}; pass dataset=<workspace name> to "
        f"disambiguate: {opts}")


def _legacy_load_batch_peaks(client, batch: str, *, dataset: str | None = None,
                             matches: bool = False, areas: bool = True,
                             heights: bool = True, average: bool = True,
                             max_workers: int = 8) -> pd.DataFrame | None:
    """Legacy-server equivalent of `client.load_peaks(dataset, batches)` for ONE
    batch: resolve the `sample_batch_id` from the raw `sample/batches` endpoint
    (the modern loader's only break is its datasets-based batch resolution), then
    run the SDK's own per-sample fetch+enrich loop. Produces the SAME column shape
    as load_peaks (get_peaks columns + sample_batch_name/sample_item_name/
    datetime_utc). `matches=False` by default: the cluster/TS layer keys formulas
    off the merged ledger and only needs mz/height/datetime/sample_item_id from the
    TS, so the match tree is skipped (far lighter over ~1000 samples)."""
    from mascope_sdk._concurrent import run_concurrent
    bid = resolve_batch_id(client, batch, dataset=dataset)
    bdf = _legacy_all_batches(client)
    nm = bdf.loc[bdf["sample_batch_id"] == bid, "sample_batch_name"]
    batch_name = str(nm.iloc[0]) if len(nm) else batch
    samples = client.samples._list_by_id(bid)
    if samples is None or not len(samples):
        return None

    def _fetch(sample_row, batch_name):
        peaks = client.samples.get_peaks(sample_row["sample_item_id"], matches=matches,
                                         areas=areas, heights=heights, average=average)
        if peaks is None or peaks.empty:
            return None
        peaks.insert(0, "sample_batch_name", batch_name)
        peaks.insert(peaks.columns.get_loc("sample_item_id") + 1, "sample_item_name",
                     sample_row["sample_item_name"])
        if "datetime_utc" in sample_row.index:
            peaks.insert(peaks.columns.get_loc("sample_item_name") + 1, "datetime_utc",
                         sample_row["datetime_utc"])
        return peaks

    tasks = [(row, batch_name) for _, row in samples.iterrows()]
    frames = run_concurrent(_fetch, tasks, max_workers=max_workers,
                            desc="Loading peaks (legacy)", unit="sample")
    frames = [f.dropna(axis=1, how="all") for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


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
    """All datasets visible to the token. On a legacy server (no /api/datasets)
    this returns the WORKSPACES, reshaped to the dataset column names so the rest
    of the package/CLI is server-agnostic. Powers `list datasets` discovery."""
    ds = client.datasets.list()
    if ds is not None and len(ds):
        return ds
    ws = _legacy_workspaces(client)
    if ws is None or not len(ws):
        raise RuntimeError("no datasets/workspaces returned (check MASCOPE_URL / token)")
    return ws.rename(columns={"workspace_name": "dataset_name",
                              "workspace_id": "dataset_id",
                              "workspace_type": "dataset_type"})


def list_batches(client, dataset: str | None = None) -> pd.DataFrame:
    """Sample batches (optionally in one dataset/workspace). Columns include
    `sample_batch_name` / `polarity` / `sample_batch_id` / `status`. Falls back to
    the legacy `sample/batches` endpoint (filtered client-side by workspace name)
    when the server has no `datasets` concept."""
    try:
        bs = client.batches.list(dataset=dataset)
        if bs is not None and len(bs):
            return bs
    except Exception:
        pass
    bdf = _legacy_all_batches(client)
    if bdf is None or not len(bdf):
        raise RuntimeError("no batches returned from the server")
    if dataset:
        ws = _legacy_workspaces(client)
        wid = ws.loc[ws["workspace_name"].str.casefold() == dataset.casefold(),
                     "workspace_id"]
        if len(wid):
            bdf = bdf[bdf["workspace_id"] == wid.iloc[0]]
        if not len(bdf):
            raise RuntimeError(f"no batches for dataset/workspace {dataset!r}")
    return bdf.reset_index(drop=True)


def fetch_batch_samples(client, batch: str, *, dataset: str | None = None,
                        drop_columns=None) -> pd.DataFrame:
    """Per-sample table for a batch (one row per sample). Carries `sample_item_id`,
    `sample_item_name`, `datetime_utc`, `tic`, `polarity`, ... — enough for
    representative-sample selection WITHOUT loading every peak. Tries the modern
    `samples.list` (name resolution via datasets) and, on a legacy server where
    that resolution path 404s, resolves the `sample_batch_id` from the raw
    `sample/batches` endpoint and lists by id. Both paths return the SAME
    datetime-coerced frame (the modern single-batch path is itself `_list_by_id`)."""
    try:
        sl = client.samples.list(batch=escape_batch(batch), dataset=dataset,
                                 drop_columns=[] if drop_columns is None else drop_columns)
        if sl is not None and len(sl):
            return sl
    except Exception:
        pass  # legacy server -> resolve by sample_batch_id below
    bid = resolve_batch_id(client, batch, dataset=dataset)
    sl = client.samples._list_by_id(bid)
    if sl is None or not len(sl):
        raise RuntimeError(f"no samples for batch {batch!r} (id {bid})")
    return sl


# Cloudflare-WAF / origin-5xx / read-timeout signatures that CLEAR on retry. A
# Mascope origin behind Cloudflare throws these under burst load (521/522 origin
# down, 403 "Attention Required" WAF challenge, 429 rate-limit, gateway 5xx, read
# timeouts). Distinct from a genuinely-missing endpoint (404 on a legacy server),
# which must NOT be retried -- it falls through to the per-sample loader.
_TRANSIENT_SIGNS = ("403", "429", "500", "502", "503", "504", "521", "522",
                    "attention required", "timed out", "timeout",
                    "temporarily unavailable", "bad gateway")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(sig in s for sig in _TRANSIENT_SIGNS)


def _with_waf_retry(fn, *, tries: int = 4, base_delay: float = 2.0):
    """Call ``fn()``; on a transient WAF/5xx/timeout error back off (base_delay·2ⁿ:
    2s, 4s, 8s) and retry up to ``tries`` times. A NON-transient error re-raises
    immediately so the legacy-server fallback in the caller still fires. Tests pass
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
    SDK batch loader (dataset=, not the deprecated workspace=)."""
    # batches= is resolved as a case-insensitive REGEX (str.contains), so a literal
    # name with metacharacters (e.g. the ^ in '... ^Nitrate ...' or '(Ur+ CIMS)')
    # must be escaped or it silently matches nothing -- same as fetch_batch_samples.
    # confirm_above=None: never prompt (non-interactive; batches can exceed 100 samples).
    # Bulk load is the live default path (local scoring means match_compounds is off
    # by default); WAF-retry it so a burst 521/403 doesn't drop us onto the slow
    # per-sample legacy loader (which then hangs over ~2000 samples).
    try:
        peaks = _with_waf_retry(
            lambda: client.load_peaks(dataset=dataset, batches=escape_batch(batch),
                                      confirm_above=None))
    except Exception:
        peaks = None  # legacy server (no /api/datasets) or exhausted -> per-sample loader below
    if peaks is None or len(peaks) == 0:
        peaks = _legacy_load_batch_peaks(client, batch, dataset=dataset)
    if peaks is None or len(peaks) == 0:
        raise RuntimeError(f"no peaks for batch {batch!r} in dataset {dataset!r}")
    if save_path:
        peaks.to_parquet(os.path.expanduser(save_path))
    return peaks


def fetch_pooled_peaks(client, dataset: str, batches_regex: str, *,
                       save_path: str | None = None) -> pd.DataFrame:
    """Load the peak time-series of EVERY batch whose name matches `batches_regex`
    in ONE bulk call, pooling them into a single frame that keeps `sample_batch_name`
    so the caller can regroup. Unlike `fetch_batch_peaks`, the regex is passed to
    the SDK UNescaped — the whole point is a multi-batch match (e.g.
    `'HR-CIMS 100-500.*zone'` pools the per-zone batches of one mode x range).
    WAF-retried; confirm_above=None so a large pool never prompts."""
    peaks = _with_waf_retry(
        lambda: client.load_peaks(dataset=dataset, batches=batches_regex,
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
def fetch_peaks(client, sample_id: str, *, use_cache: bool = True,
                cache_root: Path = CACHE_ROOT) -> pd.DataFrame:
    """Pull the raw peak table (with Mascope's own matches flattened in) and
    cache it. Returns the full multi-row-per-peak frame; dedup is the ledger's
    job."""
    cdir = Path(cache_root) / sample_id
    cfile = cdir / "peaks.parquet"
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
    """Map ionization-mechanism names (e.g. '-H+', '+Br-') to their ids."""
    table = client.ionization.list()
    by_name = {r.ionization_mechanism: r.ionization_mechanism_id
               for r in table.itertuples()}
    out: dict[str, str] = {}
    for n in names:
        if n in by_name:
            out[n] = by_name[n]
    return out


# Map our adduct labels to the server's ionization-mechanism names.
ADDUCT_TO_MECH = {
    "[M-H]-": "-H+",
    "[M+Br]-": "+Br-",
    "[M+Cl]-": "+Cl-",
    "[M+I]-": "+I-",
    "[M+NO3]-": "+NO3-",
    "[M+^NO3]-": "+^NO3-",          # ¹⁵N-labelled nitrate reagent cluster
    "[M+HSO4]-": "+HSO4-",
    "[M+Br2]-": "+Br2-",
    "[M+Br3]-": "+Br3-",
    "[M+I2]-": "+I2-",
    "[M+I3]-": "+I3-",
    "[M+H]+": "+H+",
    "[M+Na]+": "+Na+",
    "[M+NH4]+": "+NH4+",
    "[M+CO3]-": "+CO3-",
    "[M+(CH4N2O)H]+": "+(CH4N2O)H+",   # protonated-urea (uronium) adduct channel
}
MECH_TO_ADDUCT = {v: k for k, v in ADDUCT_TO_MECH.items()}


def detect_adducts(peaks: pd.DataFrame) -> list[str]:
    """Infer the reagent/adduct system from the sample's own peak matches
    (the `ionization_mechanism` column). This is what makes a Br-CIMS sample
    get [M+Br]- offered as an interpretation instead of forcing Br into the
    neutral. Falls back to [M-H]- if nothing is recognised."""
    if peaks is None or "ionization_mechanism" not in peaks.columns:
        return ["[M-H]-"]
    out: list[str] = []
    for name in peaks["ionization_mechanism"].dropna().unique():
        a = MECH_TO_ADDUCT.get(str(name))
        if a and a not in out:
            out.append(a)
    return out or ["[M-H]-"]


def estimate_offset(peaks: pd.DataFrame, *, min_n: int = 8) -> float | None:
    """Rough median ppm mass-offset from the sample's OWN server matches (base
    ions only). The pass-1 self-calibration is the authoritative fit, but it runs
    AFTER pass 0 / pass 1 -- so a source with a large systematic offset (the
    instrument sits at e.g. -1.9 ppm) is blind to it in pass 0's |ppm|<=2 known-
    species gate, which then drops real contaminants whose on-trend mass is just
    past 2 ppm and lets pass 1 grab the peak with an off-trend mass-coincidence.
    This seeds those pre-calibration gates. None when too few matches to trust."""
    from peaky.chem import chemistry as C
    cols = {"target_compound_formula", "ionization_mechanism", "mz"}
    if peaks is None or not cols <= set(peaks.columns):
        return None
    iso_col = "target_isotope_formula" in peaks.columns
    ppms: list[float] = []
    for r in peaks.dropna(subset=["target_compound_formula", "mz",
                                  "ionization_mechanism"]).itertuples():
        if iso_col and "[" in str(getattr(r, "target_isotope_formula", "") or ""):
            continue                                 # heavy-isotope row, skip
        add = MECH_TO_ADDUCT.get(str(r.ionization_mechanism))
        if not add or add not in C.ADDUCT_SHIFTS:
            continue
        try:
            theo = C.ion_mz(str(r.target_compound_formula), add)
        except Exception:
            continue
        p = (float(r.mz) - theo) / theo * 1e6
        if abs(p) <= 10:                             # gross-outlier guard
            ppms.append(p)
    if len(ppms) < min_n:
        return None
    ppms.sort()
    n = len(ppms)
    return (ppms[n // 2] if n % 2 else (ppms[n // 2 - 1] + ppms[n // 2]) / 2)


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
            mz=mz, ionization_mechanism_ids=mechanism_ids,
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
    """Reverse-map resolved ionization-mechanism ids -> mascope mechanism strings
    that `mascope_tools.parse_ionization` charges CORRECTLY.

    The server's name trailing sign is the *added/removed species'* sign, not the
    net ion charge: deprotonation is named '-H+' (remove H+) yet yields an ANION.
    `parse_ionization` reads the trailing sign as the net charge, so '-H+' would
    score as a +1 cation and match nothing -- silently dropping the whole [M-H]-
    channel. The server disambiguates via `ionization_mechanism_polarity`, so we
    normalise the trailing sign to that polarity ('-H+' -> '-H-'); '+Br-'/'+NH4+'
    etc. already agree and are unchanged."""
    if not mechanism_ids:
        return []
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
        sign = _polarity_sign(pol)
        if sign and name and len(name) > 1 and name[-1] in "+-" and name[-1] != sign:
            name = name[:-1] + sign  # '-H+' (deprotonation, neg polarity) -> '-H-'
        out.append(name)
    return out


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
    fetch_peaks; channels from the reverse-mapped mechanism names."""
    from peaky.io import local_scoring

    raw = fetch_peaks(client, sample_id)                      # cached; mz/height/peak_id
    mechs = _mechanism_names(client, mechanism_ids)
    out = local_scoring.score_candidates_local(raw, formulas, mechanisms=mechs)
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
