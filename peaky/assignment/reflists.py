"""Reference peaklists — a curated catalog of known-molecule formula lists from
published chemical systems (oxidation HOM, contaminant families, ...), used as a
SOFT, context-gated prior.

Two uses, both decoupled from the expensive server scoring (one offline pass):
  * corroborate  -- a Candidate-tier neutral whose formula is on a list for the
                    sample's chemistry gets a literature-corroboration tag.
  * rescue/annotate -- an UNEXPLAINED peak (no formula) is matched BY MASS, under
                    the run's reagent adducts, against the list formulas, turning
                    the remainder into leads the reader can chase.

This is NOT the Pass-0 known-species lock (`passes._known_species`, for
contaminants the grid cannot reach). It is a post-hoc PRIOR: it never overrides an
isotope-scored Assigned, never fabricates a peak, and every match carries the
source list id for provenance. Masses are recomputed for whatever reagent adduct
the run uses (`chemistry.ion_mz`), so one list serves Br-/NO3-/I-/urea+ runs alike.

A list is one self-describing JSON in data/peaklists/ (see README.md). Add a
credible source = drop in one JSON; nothing else couples.

PINNING (peaky.lists). A list imported from an external source is a schema-v2
SNAPSHOT: it carries its provenance (`source`: DOI, record, file, the
publisher's md5, when it was retrieved, licence) and a content hash over its
species. The catalog directory's `catalog.lock` pins every snapshot's hash;
`load_catalog` recomputes each hash and refuses a mismatch
(`SnapshotIntegrityError`) -- a list is read exactly as pinned or not at all.
A run never fetches a list: importers (`peaky lists import`) do, offline.
Non-bundled snapshots live in a user directory ($PEAKY_LISTS_DIR, default
~/.cache/peaky/lists) with its own lock. `lists_record` is what the run
manifest records per active list (id, version, sha256, pinned, how activated).
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import warnings
from dataclasses import dataclass

from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.lists import lock as LK
from peaky.lists import snapshot as SN
from peaky.lists.lock import SnapshotIntegrityError  # noqa: F401  (re-exported for callers)
from peaky.paths import pkg_data

__version__ = "0.2.0"   # schema-v2 snapshots, catalog.lock verification, user dir, lists_record

# Bundled peaklist catalog. Resolved from the package root (paths.pkg_data), not
# this module's __file__, so it survives `reflists` moving into a sub-package.
_DIR = pkg_data("peaklists")

# batch-name / label keywords -> experimental-context tag (the metadata "unlock").
# Conservative: only fire on clearly source-diagnostic words.
CONTEXT_KEYWORDS = {
    "monoterpene_ox": ("monoterpene", "pinene", "a-pinene", "α-pinene", "apinene",
                       "b-pinene", "β-pinene", "limonene", "terpene", "orange",
                       "carene", "sabinene", "myrcene", "ap oxidation",
                       "ap-oxidation", "ap_oxidation"),
    "limonene_ox": ("limonene", "orange", "d-limonene"),
    # "ap oxidation" as a PHRASE: a dataset named for the chemistry ("AP oxidation
    # demo-set") must unlock the alpha-pinene lists; the bare abbreviation may not
    # (see below).
    "ap_ox": ("pinene", "a-pinene", "α-pinene", "apinene", "ap oxidation",
              "ap-oxidation", "ap_oxidation"),
    "isoprene_ox": ("isoprene", "isopn", "iepox", "isopooh", "methacrolein"),
    # NB a bare "AP" abbreviation (e.g. "AP Low temperature") is intentionally NOT a
    # keyword (too many false positives like "soap"/"grape"); tag such runs via the
    # explicit context/--reagent-config path instead.
}


@dataclass(frozen=True)
class ReferenceList:
    id: str
    system: str
    label: str
    data_version: str
    polarity: str
    native_detection: str
    applies_to_contexts: tuple
    references: tuple
    formulas: frozenset            # closed-shell neutral formulas (matchable)
    radicals: frozenset            # odd-electron (half-integer DBE) formulas (excluded by default)
    conditions_of: dict            # formula -> tuple(conditions)
    source_file: str
    always_active: bool = False    # universal lists (e.g. contaminants) ignore context gating
    meta_of: dict = None           # formula -> {name, origin, ...} (display extras)
    skipped: tuple = ()            # entries the loader refused: not a neutral molecule (`_not_a_neutral`)
    # ---- pinning (schema v2 snapshots; peaky.lists) ----
    schema_version: int = 1
    roles: tuple = ()              # declared uses (snapshot.ROLES); () on a v1 list
    sha256: str = ""               # content hash over the canonical species payload
    pinned: bool | None = None     # True: matched the directory's catalog.lock; False: no lock entry
    lock_version: str = ""         # the version the lock pins (== the file's when pinned)
    source: dict | None = None     # v2 provenance block (doi, record_id, file, md5, retrieved_utc, licence, ...)
    importer: dict | None = None   # {name, version} of the importer that wrote the snapshot
    family_scope: str = "compound" # how the evidence scale reads a named hit (compound: 3c reachable)
    directory: str = ""            # the catalog directory the file was read from

    def pool(self, include_radicals: bool = False) -> frozenset:
        return self.formulas | self.radicals if include_radicals else self.formulas

    @property
    def version(self) -> str:
        """The version the lock pins: a v2 snapshot's source version, a v1 list's `data_version`."""
        return self.data_version

    @property
    def effective_roles(self) -> tuple:
        """The declared roles; a v1 list reads `name` when any species is named, else `class`."""
        if self.roles:
            return tuple(self.roles)
        named = any((self.meta_of or {}).get(f, {}).get("name") for f in self.formulas)
        return ("name",) if named else ("class",)

    def cite(self) -> str:
        if not self.references:
            return self.label
        r = self.references[0]
        bits = [r.get("authors"), r.get("title"), r.get("publisher"),
                str(r.get("year", "")), r.get("section")]
        return ", ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
def is_radical(formula: str) -> bool:
    """True for an odd-electron neutral: a half-integer DBE, read by
    `chemistry.odd_electron` (the parity test the grid gate and the plausibility
    exemption share). Parity is the rule, not the H count -- an organic nitrate
    like C10H15NO8 has odd H and an integer DBE, so it is closed-shell."""
    return C.odd_electron(formula)


def _not_a_neutral(formula) -> str | None:
    """Why `formula` cannot be handed to the parity test, or None when it can.

    The parity test only means something for a neutral molecule written over the
    mass table: `chemistry.dbe` scores an element it does not know as divalent
    (sodium acetate C2H3NaO2 would come out DBE 1.5 and pool as a radical, with
    no message), and `parse_formula` drops charge and bracket notation (the ion
    '[C10H14NO8]-' would load as the closed-shell neutral C10H14NO8). So an entry
    must round-trip `format_formula(parse_formula(f)) == f` (Hill notation, no
    charge), use mass-table elements only, and have a non-negative DBE (a
    negative one is an ion or a salt: a quaternary-ammonium cation sits at -0.5)."""
    cnt = C.parse_formula(formula)
    unknown = sorted(e for e in cnt if e not in C.M)
    if unknown:
        return f"element {', '.join(unknown)} not in the mass table"
    if C.format_formula(cnt) != formula:
        return (f"does not read back as {C.format_formula(cnt) or 'a formula'} "
                "(charge, bracket or non-Hill notation)")
    d = C.dbe(cnt)
    if d < 0:
        return f"DBE {d:g} < 0, an ion or a salt"
    return None


#: the per-process catalog cache: {(dirs, file stamps): {id: ReferenceList}}. The
#: catalog is read per file in a batch's workers (the evidence stage's context
#: lists, the degeneracy space) and hashing every snapshot each time is wasted;
#: the key carries every file's (name, mtime_ns, size), so an edited or added
#: file is re-read.
_CATALOG_CACHE: dict = {}


def clear_catalog_cache() -> None:
    _CATALOG_CACHE.clear()


def catalog_dirs(directory: str | None = None, user_dir: str | None = "auto") -> list:
    """The directories a catalog is read from: `directory` (default the bundled
    data/peaklists) plus the user directory (`lock.user_lists_dir()`) when the
    default catalog is asked for and the user directory exists. An explicit
    `directory` (a test's tmp dir, a tool's target) is read alone unless a
    `user_dir` is named too; ``user_dir=None`` reads the bundled catalog alone."""
    dirs = [os.path.abspath(directory or _DIR)]
    if user_dir == "auto":
        u = LK.user_lists_dir() if directory is None else None
    else:
        u = os.path.expanduser(user_dir) if user_dir else None
    if u and os.path.isdir(u) and os.path.abspath(u) not in dirs:
        dirs.append(os.path.abspath(u))
    return dirs


def _dir_stamp(d: str) -> tuple:
    names = sorted(n for n in os.listdir(d) if n.endswith(".json") or n == LK.LOCK_NAME) if os.path.isdir(d) else ()
    out = []
    for n in names:
        st = os.stat(os.path.join(d, n))
        out.append((n, st.st_mtime_ns, st.st_size))
    return tuple(out)


def _join_names(names: list) -> str:
    """One display name for a formula several listed compounds share: the first
    two, then how many more ('Xylene; Ethylbenzene (+2 more)')."""
    names = list(dict.fromkeys(n for n in names if n))
    if not names:
        return ""
    head = "; ".join(names[:2])
    return head if len(names) <= 2 else f"{head} (+{len(names) - 2} more)"


def _load_one(p: str, lock: dict, lock_present: bool, *, verify: bool) -> ReferenceList:
    name = os.path.basename(p)
    d = SN.read_snapshot(p)
    if not d.get("id"):
        raise ValueError(f"{name}: a reference list needs an `id`")
    errs = SN.validate_snapshot(d)
    sha_errs = [e for e in errs if e.startswith("sha256 mismatch")]
    other = [e for e in errs if e not in sha_errs]
    if other:
        raise ValueError(f"{name}: not a valid schema-v{SN.schema_version(d)} snapshot: " + "; ".join(other))
    entry = (lock.get("lists") or {}).get(d["id"])
    if verify:
        sha = LK.verify(d, entry, name=name)        # raises SnapshotIntegrityError on any mismatch
    else:
        sha = SN.species_sha256(d.get("species", []))
    v2 = SN.schema_version(d) >= SN.SCHEMA_V2
    pinned = entry is not None
    if not pinned and (v2 or lock_present):
        warnings.warn(f"{name}: list {d['id']!r} is UNPINNED (no entry in {LK.LOCK_NAME}); "
                      "run `peaky lists lock` to pin it", stacklevel=3)
    sp = d.get("species", [])
    closed, rad, cond, meta, wrong, skipped, why = set(), set(), {}, {}, [], [], []
    names_of: dict = {}
    for s in sp:
        f = s.get("formula")
        if not f:
            continue
        reason = _not_a_neutral(f)
        if reason:
            skipped.append(f)
            why.append(f"{f}: {reason}")
            continue
        radical = is_radical(f)
        if bool(s.get("radical", False)) != radical:
            wrong.append(f)
        (rad if radical else closed).add(f)
        cond[f] = tuple(s.get("conditions", ()))
        extra = {k: s[k] for k in ("origin", "modes", "inchikey", "inchikey_block", "cas", "ids") if k in s}
        if s.get("name"):
            names_of.setdefault(f, []).append(str(s["name"]))
        if extra or f in names_of:
            m = meta.setdefault(f, {})
            for k, v in extra.items():
                m.setdefault(k, v)
            m["n_compounds"] = m.get("n_compounds", 0) + 1
            if f in names_of:
                m["name"] = _join_names(names_of[f])
                if len(names_of[f]) > 1:
                    m["names"] = list(dict.fromkeys(names_of[f]))
    if skipped:
        warnings.warn(
            f"{name}: {len(skipped)} species skipped, not a neutral "
            f"molecule over the mass table ({'; '.join(why[:5])}"
            f"{'; ...' if len(why) > 5 else ''})", stacklevel=3)
    if wrong:
        warnings.warn(
            f"{name}: the radical flag (false when absent) of "
            f"{len(wrong)} species disagrees with its formula's DBE parity "
            f"({', '.join(wrong[:5])}"
            f"{', ...' if len(wrong) > 5 else ''}); parity decides -- run "
            "tests/test_peaklists.py for the full list", stacklevel=3)
    return ReferenceList(
        id=d["id"], system=d.get("system", ""), label=d.get("label", d["id"]),
        data_version=LK.snapshot_version(d),
        polarity=d.get("polarity", ""), native_detection=d.get("native_detection", ""),
        applies_to_contexts=tuple(d.get("applies_to_contexts", ())),
        references=tuple(d.get("references", ())),
        formulas=frozenset(closed), radicals=frozenset(rad), conditions_of=cond,
        source_file=name,
        always_active=bool(d.get("always_active", False)), meta_of=meta,
        skipped=tuple(skipped),
        schema_version=SN.schema_version(d), roles=tuple(d.get("roles") or ()),
        sha256=sha, pinned=pinned, lock_version=str(entry.get("version", "")) if entry else "",
        source=dict(d["source"]) if isinstance(d.get("source"), dict) else None,
        importer=dict(d["importer"]) if isinstance(d.get("importer"), dict) else None,
        family_scope=str(d.get("family_scope") or "compound"),
        directory=os.path.dirname(os.path.abspath(p)))


def load_catalog(directory: str | None = None, *, user_dir: str | None = "auto",
                 verify: bool = True) -> dict:
    """Load every *.json reference list under `directory` (default: the packaged
    data/peaklists plus the user directory, `catalog_dirs`). Returns {id:
    ReferenceList}; a user-directory list of an id the bundled catalog also has
    wins, with a warning.

    A species goes to `formulas` or `radicals` by its formula's DBE parity
    (`is_radical`). Its `radical` value (false when absent) is the author's
    claim, checked but never deciding: a list whose claims disagree with parity
    loads with a warning naming them. An entry whose formula is not a neutral
    molecule over the mass table (`_not_a_neutral`) is skipped with a warning
    and recorded in `ReferenceList.skipped`, never pooled.

    PINNING: each directory's `catalog.lock` is read; with ``verify`` (the
    default) every list's species hash is recomputed and a mismatch with its own
    `sha256` or the lock's raises `SnapshotIntegrityError` naming the list. A
    pinned list reads `pinned=True`; a schema-v2 snapshot (or any list in a
    directory that has a lock) without a lock entry loads with an UNPINNED
    warning. A malformed v2 snapshot raises ValueError. Results are cached per
    process on the files' stamps (`clear_catalog_cache`)."""
    dirs = catalog_dirs(directory, user_dir)
    key = (tuple(dirs), tuple(_dir_stamp(d) for d in dirs), bool(verify))
    hit = _CATALOG_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    out: dict = {}
    for d in dirs:
        lock = LK.read_lock(d)
        lock_present = os.path.exists(LK.lock_path(d)) and bool(lock.get("lists"))
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            L = _load_one(p, lock, lock_present, verify=verify)
            if L.id in out:
                warnings.warn(f"list {L.id!r}: {p} shadows {os.path.join(out[L.id].directory, out[L.id].source_file)}",
                              stacklevel=2)
            out[L.id] = L
    _CATALOG_CACHE[key] = dict(out)
    return out


def resolve_context_tags(*texts: str) -> set:
    """Infer experimental-context tags from run metadata: the batch name, the
    DATASET name and the reagent label, all passed as `texts`. This is the
    'unlock with metadata' step — only the matched contexts' lists become active,
    keeping precision high on unrelated samples. Pass the dataset name too: on a
    campaign whose batch names describe instrument and reagent ("... TOF NO3_Br
    mixed ...") the chemistry lives only in the dataset name ("AP oxidation
    demo-set"), and a resolver that never saw it left the alpha-pinene reference
    list inactive on alpha-pinene data."""
    blob = " ".join(t for t in texts if t).lower()
    return {tag for tag, kws in CONTEXT_KEYWORDS.items()
            if any(k.lower() in blob for k in kws)}


def active_lists(catalog: dict, *, context_tags=()) -> list:
    """Select active lists: every `always_active` list (ubiquitous, e.g. lab
    contaminants) PLUS any whose `applies_to_contexts` intersect the run's context
    tags (the metadata unlock). Polarity is NOT a filter: a neutral-formula library
    transfers across reagent ion forms."""
    tags = set(context_tags)
    return [L for L in catalog.values()
            if L.always_active or (tags and set(L.applies_to_contexts) & tags)]


#: what `activate` calls its texts in the activation record, by position
#: (a batch passes its batch name, its dataset name and the reagent label)
ACTIVATION_FIELDS = ("batch", "dataset", "reagent label")


def keyword_matches(texts: dict) -> dict:
    """{context tag: [(field, keyword), ...]}: the keywords `resolve_context_tags`
    finds, per field of ``texts`` ({field: text}) -- the same lower-cased
    substring rule, so the tags are the ones `activate` resolves."""
    out: dict = {}
    for field, text in texts.items():
        blob = str(text or "").lower()
        if not blob:
            continue
        for tag, kws in CONTEXT_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in blob and (field, kw) not in out.setdefault(tag, []):
                    out[tag].append((field, kw))
    return {t: v for t, v in out.items() if v}


def activation_record(texts, fields=None, include=()) -> dict:
    """How `activate(*texts)` activated: {tags: [...], matched: {tag: [[field,
    keyword], ...]}, explicit: [...]} -- which keyword in which text unlocked
    each tag, and which list ids were asked for by name (`--lists`)."""
    texts = list(texts)
    fields = list(fields) if fields is not None else (
        list(ACTIVATION_FIELDS) if len(texts) == len(ACTIVATION_FIELDS) else [f"text {i + 1}" for i in range(len(texts))])
    m = keyword_matches(dict(zip(fields, texts)))
    tags = sorted(resolve_context_tags(*[t or "" for t in texts]))
    return dict(tags=tags, matched={t: [list(x) for x in m.get(t, [])] for t in tags},
                explicit=[str(x) for x in (include or ())])


#: the environment variable `activate` reads explicit list ids from (comma-separated),
#: the same as the CLI's --lists
LISTS_ENV = "PEAKY_LISTS"


def explicit_list_ids(include=None) -> list:
    """The list ids asked for by name: ``include`` (a sequence, or a
    comma-separated string) else $PEAKY_LISTS. Order kept, duplicates dropped."""
    if include is None:
        include = os.environ.get(LISTS_ENV, "")
    if isinstance(include, str):
        include = [x for x in include.split(",")]
    return list(dict.fromkeys(str(x).strip() for x in include if str(x).strip()))


def activate(*texts: str, record: bool = False, fields=None, include=None):
    """The one activation step every entry point runs: infer the context tags from
    whatever metadata the run has, then select the lists those tags unlock.
    Returns `(lists, tags)` so the caller can log what the metadata bought it.

    A run whose metadata names no chemistry — a single-sample `peaky assign`,
    whose texts are a context label like 'ambient-air' and the reagent label —
    still gets the `always_active` lists (the lab contaminants); only `peaky
    batch`, which has a batch name and a dataset name to read, can unlock a
    chemistry-specific list (on a campaign whose batch names describe the
    instrument and the reagent, the chemistry lives in the dataset name).

    ``include`` names lists to activate regardless of keywords (the CLI's
    `--lists`, else $PEAKY_LISTS): an imported suspect list whose contexts no
    keyword describes is switched on by id. An unknown id raises KeyError
    naming the catalog's ids -- a run must not silently proceed without a list
    it was told to use.

    ``record=True`` returns `(lists, tags, record)`, the record saying which
    keyword in which text (``fields`` names them; default batch / dataset /
    reagent label for three texts) unlocked each tag and which ids were
    explicit (`activation_record`); which lists activate is the same either way."""
    tags = resolve_context_tags(*texts)
    cat = load_catalog()
    lists = active_lists(cat, context_tags=tags)
    ids = explicit_list_ids(include)
    missing = [i for i in ids if i not in cat]
    if missing:
        raise KeyError(f"reference list(s) {missing} not in the catalog "
                       f"(bundled + {LK.user_lists_dir()}); known: {sorted(cat)}")
    have = {L.id for L in lists}
    lists = lists + [cat[i] for i in ids if i not in have]
    if record:
        return lists, tags, activation_record(texts, fields, include=ids)
    return lists, tags


def prior_formulas(lists) -> frozenset:
    """The SELECTION-PRIOR set: every closed-shell formula on the active lists
    (`assign.run` hands it to `cfg.reflist_formulas`, which `arbitrate` reads as
    a near-tie tie-break). Radicals are deliberately absent -- the prior may only
    nudge arbitration toward a neutral the pipeline is allowed to assign, and the
    closed-shell grid never offers a radical to nudge toward."""
    return frozenset().union(*(L.formulas for L in lists)) if lists else frozenset()


def active_versions(lists) -> list:
    """`[(id, data_version), ...]` of the lists a run had active -- the manifest
    line that says which list versions shaped its selection prior and rescue
    (a list's split changes with its `data_version`, e.g. the HOM list's 2024.2
    parity correction moved 118 nitrates into the matched pool)."""
    return [(L.id, L.data_version) for L in (lists or ())]


def lists_record(lists, activation=None) -> list:
    """What the run manifest records per active list: id, version, the content
    hash it was read at, its declared roles, whether the catalog lock pinned it,
    and how it was activated. A run re-executed later can check it read the
    same content (`sha256`) or say which list changed."""
    from peaky.assignment.levels.lists import activation_how
    out = []
    explicit = set((activation or {}).get("explicit") or ())
    for L in (lists or ()):
        how = "explicit (--lists)" if L.id in explicit else activation_how(L, activation)
        out.append({"id": L.id, "version": L.data_version, "sha256": L.sha256,
                    "roles": list(L.effective_roles), "pinned": L.pinned,
                    "schema_version": int(L.schema_version), "file": L.source_file,
                    "how_activated": how})
    return out


# ---------------------------------------------------------------------------
def match_assigned(neutral_formulas, lists, *, include_radicals: bool = False) -> dict:
    """Formula-membership corroboration: which assigned neutrals appear on a list.
    Returns {formula: [{list, conditions}]}."""
    res: dict = {}
    for f in {str(x) for x in neutral_formulas if x and str(x) != "nan"}:
        hits = [{"list": L.id, "conditions": list(L.conditions_of.get(f, ()))}
                for L in lists if f in L.pool(include_radicals)]
        if hits:
            res[f] = hits
    return res


# The bare (covalent) channels: a deprotonated / protonated molecular ion with
# no reagent cluster. Two list entries can name the SAME ION under two of the
# run's adducts -- an acid's reagent cluster and the deprotonated organonitrate
# one HNO3 heavier (C5H6O6 [M+NO3]- and C5H7NO9 [M-H]- are both C5H6NO9-) --
# and then neither the mass nor the isotopes can separate them. At such a tie
# the CLUSTER reading is kept: it is the reagent's native detection (the
# list's own `native_detection`) and the tier engine's standing policy for a
# same-ion decomposition alias (tiers._drop_decomposition_aliases, 2026-06-11);
# the covalent reading is recorded on the row as the alias. Before this the
# winner was whichever entry the formula frozenset iterated first -- hash order,
# different in every interpreter -- and three TOF peaks flipped reading between
# byte-identical runs.
BARE_ADDUCTS = frozenset({"[M-H]-", "[M+H]+", "[M-H]+", "[M]-.", "[M]+."})
#: two targets within this many ppm of each other are one ion (a same-ion pair
#: differs by float rounding only, ~1e-9 ppm)
TIE_PPM = 1e-6


def _adduct_rank(adduct: str, adducts) -> tuple:
    """Order among same-ion entries: a cluster channel before a bare one, then
    the run's own channel order, then the text."""
    try:
        pos = list(adducts).index(adduct)
    except ValueError:
        pos = len(adducts)
    return (1 if adduct in BARE_ADDUCTS else 0, pos, adduct)


def _target_table(lists, adducts, include_radicals: bool):
    """Sorted [(mz, formula, adduct, list_id, rank)] of every list formula under
    every reagent adduct -- built once per match call. Fully ordered (mass, then
    `_adduct_rank`, then the formula and list text), so the same catalog gives
    the same table in every interpreter."""
    rows = []
    for L in lists:
        for f in sorted(L.pool(include_radicals)):
            for a in adducts:
                try:
                    rows.append((C.ion_mz(f, a), f, a, L.id, _adduct_rank(a, adducts)))
                except Exception:
                    continue
    rows.sort(key=lambda r: (r[0], r[4], r[1], r[3]))
    return rows


def rescue_unexplained_by_reflist(client, sample_id, ledger, profile, cfg, lists,
                                  adducts, *, score_fn=None, tol_ppm: float = 4.0,
                                  log=print) -> dict:
    """RESCUE-VERIFY: match UNEXPLAINED peaks by mass to active reference-list
    formulas, then SCORE those specific formulas with the server (match_compounds).

    Decision per matched peak (mass gate: server ion_score >= tau_low AND on-cal z):
      * isotope-CONFIRMED  -> commit literature-anchored M0 (Good/Assigned-grade);
      * too DIM to confirm (the predicted 13C M+1 falls below height_cutoff, so no
        satellite COULD show) -> commit a low-quality Candidate + tentative_lead
        so the lead is never lost back to 'unexplained' (the user's small-peak rule);
      * isotopes EXPECTED (bright enough) but absent, or off-cal / poor score
        -> leave unexplained (a real mass coincidence, not corroborated).

    A soft, provenance-tagged rescue: never overrides an existing assignment (only
    unlocked ROLE_UNEXPLAINED peaks are touched: a locked one was emptied for good,
    the element-evidence clear), every commit records the source list."""
    import pandas as pd

    from peaky.io import io_mascope as IO
    from peaky.assignment import ledger as L
    score_fn = score_fn or IO.score_candidates
    if not lists:
        return {"rescued": 0, "tentative": 0}
    # a LOCKED unexplained peak is one a stage emptied for good (the element-
    # evidence stage's clear: the file's own peak list contradicted the reading
    # there) -- it stays unexplained, no list match refills it
    free = (~ledger["locked"].map(L._truthy).astype(bool) if "locked" in ledger.columns
            else pd.Series(True, index=ledger.index))
    un = ledger[(ledger["role"] == L.ROLE_UNEXPLAINED) & free].dropna(subset=["mz"])
    if not len(un):
        return {"rescued": 0, "tentative": 0}
    mz_by_pid = {pid: float(m) for pid, m in zip(un["peak_id"], un["mz"])}
    matches = match_by_mass(list(mz_by_pid.values()), lists, adducts, tol_ppm=tol_ppm)
    by_mz = {round(m, 5): pid for pid, m in mz_by_pid.items()}
    want, allf = {}, set()
    for m in matches:
        pid = by_mz.get(round(m["obs_mz"], 5))
        if pid is not None and pid not in want:
            want[pid] = (m["formula"], m["adduct"], m["list"], m.get("aliases") or [])
            allf.add(m["formula"])
    if not allf:
        return {"rescued": 0, "tentative": 0}
    fr = score_fn(client, sample_id, sorted(allf), allow_partial=True,
                  mechanism_ids=getattr(cfg, "mechanism_ids", None))
    if fr is None or not len(fr):
        return {"rescued": 0, "tentative": 0}
    fr = fr[fr["sample_peak_id"].notna()]
    mu = getattr(cfg, "cal_mu", None)
    sigma = getattr(cfg, "cal_sigma", None) or 0.5
    z_acc = getattr(cfg, "cal_z_accept", 2.0)
    floor = getattr(cfg, "tau_low", 0.70)
    hcut = cfg.height_cutoff          # resolved gate (raises if unresolved: fail closed)
    # the counting-detector floor of the tier pass (C46): this stage runs AFTER
    # apply_tiers and stamps tiers itself, so it has to honour the floor too --
    # a sub-floor TOF centroid matched to a list formula is a lead, never Assigned
    from peaky.assignment import tiers as _T
    tof_floor = _T.tof_assign_floor(cfg)
    rescued = tentative = 0
    for pid, (formula, adduct, lid, aliases) in want.items():
        idx = ledger.index[ledger["peak_id"] == pid]
        if not len(idx) or ledger.at[idx[0], "role"] != L.ROLE_UNEXPLAINED:
            continue
        i = idx[0]
        mz = float(ledger.at[i, "mz"])
        h = float(ledger.at[i, "height"]) if pd.notna(ledger.at[i, "height"]) else 0.0
        sub = fr[(fr["compound_formula"] == formula)
                 & ((fr["sample_peak_mz"] - mz).abs() < 0.006)]
        base = sub[sub["is_base"] & sub["ion_score"].notna()]
        if base.empty:
            continue
        top = base.sort_values("ion_score", ascending=False).iloc[0]
        score = float(top["ion_score"])
        ppm = float(top["ppm_error"]) if pd.notna(top["ppm_error"]) else None
        z = abs((ppm - mu) / sigma) if (mu is not None and ppm is not None) else 0.0
        if score < floor or ppm is None or z > z_acc:
            continue                                   # poor mass match / off-cal -> leave
        # isotope confirmation: a confirmed satellite sits at the M+1/M+2 mass, not
        # the base m/z -- scan the full scored frame for this formula, not `sub`.
        iso_ok = bool(len(fr[(fr["compound_formula"] == formula) & (~fr["is_base"])
                             & (pd.to_numeric(fr["iso_score"], errors="coerce").fillna(0) > 0.4)]))
        nC = C.parse_formula(formula).get("C", 0)
        # predicted 13C M+1 vs the floor -- the shared observability predicate
        # (chem/isotopes.satellite_observable), which passes.core's het-iso gate
        # asks of 34S/37Cl/81Br/29Si for the same reason. h is 0.0 when the
        # height is unknown, which reads as "not observable" here on purpose: an
        # unverifiable rescue must land tentative, never Assigned.
        iso_observable = ISO.satellite_observable("C", nC, h, hcut)
        srcs = next((Ls.cite().split(",")[0] for Ls in lists if Ls.id == lid), lid)
        # a same-ion alias on the list (see BARE_ADDUCTS): recorded on the row,
        # in `alternatives` and in the commentary, never a competing candidate
        alts = [{"formula": al["formula"], "adduct": al["adduct"], "list": al["list"],
                 "note": "same ion (exact-mass alias): not separable by mass or isotopes"}
                for al in aliases] or None
        alias_note = ("" if not aliases else
                      " Same ion as " + ", ".join(f"{al['formula']} {al['adduct']} ({al['list']})"
                                                  for al in aliases)
                      + ": the reagent-cluster reading is kept (the list's native detection; "
                      "the decomposition-alias policy), the covalent reading is its alias -- "
                      "no mass or isotope evidence can separate them.")
        sub_floor = tof_floor is not None and h < tof_floor
        # runs AFTER apply_tiers (like the F/carbon demotes), so set tier explicitly.
        if iso_ok and not sub_floor:
            L.commit_assignment(ledger, pid, neutral_formula=formula, adduct=adduct,
                                ion_formula=str(top["ion_formula"]), ion_score=score,
                                compound_score=score, ppm_error=ppm, pass_no=8,
                                method=f"reflist-rescue:{lid}", confidence="Good (literature)",
                                alternatives=alts,
                                commentary=(f"Reference-list match ({srcs}); server score "
                                            f"{score:.2f}, isotope-confirmed, z={z:.1f}."
                                            + alias_note))
            ledger.at[i, "tier"] = "Assigned"
            rescued += 1
        elif not iso_observable or sub_floor:          # too dim to confirm -> tentative
            L.commit_assignment(ledger, pid, neutral_formula=formula, adduct=adduct,
                                ion_formula=str(top["ion_formula"]), ion_score=score,
                                compound_score=score, ppm_error=ppm, pass_no=8,
                                method=f"reflist-rescue:{lid}",
                                confidence="Candidate (literature, dim)",
                                alternatives=alts,
                                commentary=(f"Reference-list match ({srcs}); server score "
                                            f"{score:.2f}, z={z:.1f}. Too dim ({h:.0f} cps) to "
                                            "confirm isotopes -- tentative lead, not confirmed."
                                            + (f" Under the counting-detector floor ({tof_floor:.3g} cps): "
                                               "a sub-edge centroid, whatever matched its 13C window."
                                               if sub_floor else "")
                                            + alias_note))
            ledger.at[i, "tier"] = "Candidate"
            # a lead, not a contradiction: nothing could test the formula here
            # (C19(c)); the flags are created where the tier stage has not
            L.ensure_flags(ledger)
            L.mark_lead(ledger, i, "reflist_dim")
            tentative += 1
        # else: bright enough to show isotopes but none confirmed -> mass coincidence, leave
    log(f"[reflist] rescue-verify: {rescued} confirmed + {tentative} tentative "
        f"(of {len(want)} reference-matched unexplained)")
    return {"rescued": rescued, "tentative": tentative}


def match_by_mass(mz_values, lists, adducts, *, tol_ppm: float = 5.0,
                  include_radicals: bool = False) -> list:
    """Rescue/annotate UNEXPLAINED peaks (which have no formula) BY MASS: for each
    observed m/z, find the closest list-formula ion (any reagent adduct) within
    `tol_ppm`. Returns one dict per matched observed peak (best match only):
    {obs_mz, formula, adduct, list, ppm, target_mz, aliases}, where `aliases`
    lists the OTHER list entries that name the same ion (within `TIE_PPM` of the
    winner: an acid's reagent cluster vs the deprotonated organonitrate of the
    same list), the cluster reading having won the tie (`_adduct_rank`)."""
    targets = _target_table(lists, adducts, include_radicals)
    if not targets:
        return []
    tmz = [t[0] for t in targets]
    out = []
    for obs in mz_values:
        try:
            obs = float(obs)
        except (TypeError, ValueError):
            continue
        if obs <= 0:
            continue
        w = obs * tol_ppm * 1e-6
        lo = bisect.bisect_left(tmz, obs - w)
        hi = bisect.bisect_right(tmz, obs + w)
        hits = []
        for j in range(lo, hi):
            m, f, a, lid, rank = targets[j]
            ppm = (obs - m) / obs * 1e6
            if abs(ppm) <= tol_ppm:
                hits.append((abs(ppm), rank, f, lid, a, ppm, m))
        if not hits:
            continue
        best_abs = min(h[0] for h in hits)
        # the nearest target wins; targets within TIE_PPM of it are the same ion,
        # ranked by channel (cluster first), then formula and list text
        ties = sorted((h for h in hits if h[0] <= best_abs + TIE_PPM),
                      key=lambda h: (h[1], h[2], h[3]))
        _, _, f, lid, a, ppm, m = ties[0]
        out.append({"obs_mz": round(obs, 5), "formula": f, "adduct": a,
                    "list": lid, "ppm": round(ppm, 2), "target_mz": round(m, 5),
                    "aliases": [{"formula": t[2], "adduct": t[4], "list": t[3]}
                                for t in ties[1:] if (t[2], t[4]) != (f, a)]})
    return out
