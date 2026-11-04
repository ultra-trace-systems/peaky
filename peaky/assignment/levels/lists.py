"""The context lists of a run (step 3 of the evidence scale) and where they came from.

A run's context lists are its ACTIVE reference lists (closed-shell formulas;
`batch_summary["reflists_active"]`) and the pass-0 registry of its polarity and
context. An entry is NAMED when it names a compound: a reference-list entry with
a name (or, on a list that has names at all, its `origin` where the name is
empty), or a registry entry of a compound-scope family. Formula-only lists and
class-scope registry families give CLASS entries. A neutral matches an entry by
its formula (element counts), on any adduct; a labelled neutral (``^``) never
matches. Only a named entry can make level 3c; a class entry is a tag.

`context` / `context_source` say which entries matched the neutral and how the
lists that matched were activated (always active, a keyword in the batch /
dataset / reagent label name, the pass-0 registry).

`mode_flag` says where a named entry's recorded source mode or ion form
contradicts or differs from the pair (a flag: it never blocks 3c).
"""
from __future__ import annotations

from collections import defaultdict

from peaky.assignment import reflists as RL
from peaky.assignment.levels.routes import fcounts, fkey

# the activation fields RL.activate reads, in its argument order, as the context source names them
ACTIVATION_FIELDS = ("batch", "dataset", "reagent label")
# the in-pass list text's activation note for a keyword-activated list (internal `lists` column only)
KEYWORD_SOURCE = "names (keywords)"


def keyword_matches(texts: dict) -> dict:
    """{context tag: [(field, keyword), ...]} (`reflists.keyword_matches`)."""
    return RL.keyword_matches(texts)


def activation_record(batch: str = "", dataset: str = "", label: str = "") -> dict:
    """The record of how a run's reference lists were activated, from the
    texts `RL.activate(batch, dataset, label)` reads: {tags: [...], matched:
    {tag: [[field, keyword], ...]}}. Which lists activate is unchanged."""
    return RL.activation_record((batch or "", dataset or "", label or ""), ACTIVATION_FIELDS)


def activation_how(L, activation) -> str:
    """How one active list was activated: 'always active' | "keyword '<kw>' in
    the <field> name, ..." | 'activation not recorded'."""
    if L.always_active:
        return "always active"
    if not activation:
        return "activation not recorded"
    if L.id in set(activation.get("explicit") or ()):
        return "explicit (--lists)"
    matched = activation.get("matched") or {}
    pairs = []
    for tag in L.applies_to_contexts:
        for fk in matched.get(tag, []):
            fk = (str(fk[0]), str(fk[1]))
            if fk not in pairs:
                pairs.append(fk)
    if not pairs:
        return "activation not recorded"
    return ", ".join(f"keyword '{kw}' in the {field} name" for field, kw in pairs)


def reflists_context(lists, activation) -> dict:
    """batch_summary["reflists_context"]: the activation record ({tags,
    matched}) plus every active list as [id, data_version, how]."""
    rec = dict(activation or {"tags": [], "matched": {}})
    rec["active"] = [[L.id, L.data_version, activation_how(L, activation)] for L in (lists or ())]
    return rec


class ContextLists:
    """The run's context lists, indexed by formula key; ``activation`` = the
    run's activation record (`activation_record`; None when the run did not
    record it)."""

    def __init__(self, polarity: str, context: str, active, *, catalog=None, activation=None):
        from peaky.assignment import evidence as EV
        from peaky.assignment.passes.directors import _known_species
        self.by_key = defaultdict(list)
        self.active = [(str(x[0]), str(x[1]) if len(x) > 1 else "") for x in (active or [])]
        self.activation = activation
        self.pol = "positive" if str(polarity).startswith("pos") else "negative"
        cat = RL.load_catalog() if catalog is None else catalog
        self._cat = cat
        self._meta: dict = {}
        self.how_of: dict = {}
        explicit = set((activation or {}).get("explicit") or ())
        for lid, _ver in self.active:
            L = cat.get(lid)
            if L is None:
                continue
            meta = L.meta_of or {}
            how = ("always active (universal list)" if L.always_active
                   else "explicit (--lists)" if lid in explicit
                   else f"context source {KEYWORD_SOURCE} -> {','.join(L.applies_to_contexts)}")
            self.how_of[f"reflist:{lid}"] = self._source_text(lid, L)
            # a list WITH names gives a named entry for every entry; where an entry's `name` is empty its
            # `origin` names the compound. A snapshot of `family_scope` class names a family, not the
            # neutral: its entries are class entries whatever they are called.
            has_names = any((meta.get(f) or {}).get("name") for f in L.formulas)
            class_only = str(getattr(L, "family_scope", "compound")) == "class"
            for f in L.formulas:
                mf = meta.get(f) or {}
                nm = mf.get("name") or ""
                if not nm and has_names and mf.get("origin"):
                    nm = str(mf["origin"])
                if class_only:
                    nm = ""
                k = fkey(fcounts(f))
                if k:
                    self.by_key[k].append(dict(id=f"reflist:{lid}", named=bool(nm), name=nm or "", how=how,
                                               kind="named" if nm else "class"))
        for fam, table in _known_species(self.pol, context).items():
            if str(fam).startswith("contaminant:"):
                continue          # a pass-0 contaminant loop is not a context list
            scope = EV.family_scope(fam)
            self.how_of[f"registry:{fam}"] = f"registry:{fam}: pass-0 registry ({self.pol})"
            for f, label in table.items():
                if RL.is_radical(f):
                    continue
                k = fkey(fcounts(f))
                if not k:
                    continue
                named = scope == "compound"
                self.by_key[k].append(dict(id=f"registry:{fam}", named=named, name=str(label) if named else "",
                                           how=f"pass-0 registry ({self.pol})", kind="named" if named else "class"))

    # ---- activation ----
    def _source_text(self, lid: str, L) -> str:
        return f"reflist:{lid}: " + activation_how(L, self.activation)

    # ---- size-aware naming (schema-v2 snapshots) ----
    #: the formula steps a list's neighbourhood is read over: a homologue (CH2),
    #: a saturation step (H2), an oxidation step (O), both ways
    NEIGHBOUR_DELTAS = (("CH2", {"C": 1, "H": 2}), ("H2", {"H": 2}), ("O", {"O": 1}))

    def specificity(self, hit, neutral) -> dict | None:
        """How specific a named reference-list hit is: the hit's list may also
        name formula NEIGHBOURS of `neutral` (+-CH2, +-H2, +-O). A list that
        names every neighbour names everything -- a big suspect list hits a
        formula by size, not by knowledge -- so the count rides the level text as
        a tag (no level effect yet; a floor is the obvious next step). Only for
        a `reflist:` hit of a schema-v2 snapshot; None otherwise."""
        from peaky.chem import chemistry as C
        if not hit["id"].startswith("reflist:"):
            return None
        L = self._cat.get(hit["id"].split(":", 1)[1])
        if L is None or int(getattr(L, "schema_version", 1) or 1) < 2 or not isinstance(neutral, str):
            return None
        base = C.parse_formula(neutral)
        found = []
        for lab, delta in self.NEIGHBOUR_DELTAS:
            for sign in (1, -1):
                cnt = dict(base)
                for el, n in delta.items():
                    cnt[el] = cnt.get(el, 0) + sign * n
                if any(v < 0 for v in cnt.values()):
                    continue
                f = C.format_formula({k: v for k, v in cnt.items() if v > 0})
                if f and f != neutral and f in L.formulas:
                    found.append(f"{'+' if sign > 0 else '-'}{lab} {f}")
        return {"n": len(found), "neighbours": found, "n_listed": len(L.formulas)}

    # ---- lookups ----
    def hits(self, neutral) -> list:
        if not isinstance(neutral, str) or "^" in neutral:
            return []
        return self.by_key.get(fkey(fcounts(neutral)), [])

    @staticmethod
    def text(hits) -> str:
        return "; ".join(f"{h['id']} [{'named: ' + h['name'] if h['named'] else 'class'}; {h['how']}]" for h in hits)

    def context(self, hits) -> str:
        """The ``context`` column: the matching entries, named and class."""
        return "; ".join(f"{h['id']} = {h['name']}" if h["named"] else h["id"] for h in hits)

    def context_source(self, hits) -> str:
        """The ``context_source`` column: how the lists that matched were
        activated; when nothing matched, how the run's lists were."""
        if hits:
            ids = list(dict.fromkeys(h["id"] for h in hits))
        else:
            ids = [f"reflist:{lid}" for lid, _v in self.active if f"reflist:{lid}" in self.how_of]
            return "; ".join([self.how_of[i] for i in ids] + [f"registry: pass-0 registry ({self.pol})"])
        return "; ".join(self.how_of.get(i, f"{i}: activation not recorded") for i in ids)

    # ---- mode / ion-form flags ----
    def list_meta(self, list_id: str, neutral):
        lid = list_id.split(":", 1)[1] if list_id.startswith("reflist:") else None
        if lid is None or lid not in self._cat:
            return None
        m = self._meta.get(lid)
        if m is None:
            L = self._cat[lid]
            mm = {}
            for f in L.formulas:
                k = fkey(fcounts(f))
                if k:
                    mm[k] = dict(L.meta_of.get(f) or {}) if L.meta_of else {}
            m = self._meta[lid] = dict(meta=mm, polarity=str(L.polarity or ""), native=str(L.native_detection or ""))
        return dict(entry=m["meta"].get(fkey(fcounts(neutral)), {}), polarity=m["polarity"], native=m["native"])

    def mode_flag(self, hit, neutral, pol, adduct) -> str:
        """'' or where the entry's recorded source mode / ion form contradicts or
        differs from this pair. The registry is read per polarity: never flagged.
        The anion / cation tests are substring tests on name + origin."""
        if hit["id"].startswith("registry:"):
            return ""
        m = self.list_meta(hit["id"], neutral)
        if not m:
            return ""
        e = m["entry"]
        esi = "ESI-" if pol == "negative" else "ESI+"
        modes = e.get("modes") or []
        out = []
        if modes and esi not in modes:
            out.append(f"MODE CONTRADICTS: entry recorded in {'/'.join(modes)}, this run is {esi}")
        text = f"{e.get('name', '')} {e.get('origin', '')}".lower()
        if pol == "positive" and "anion" in text:
            out.append("ion form: the entry is an anion")
        if pol == "negative" and "cation" in text:
            out.append("ion form: the entry is a cation")
        # a list recorded for one polarity contradicts a run of the other; a list
        # of both (a suspect list of neutrals, not of detected ions) contradicts nothing
        if not modes and m["polarity"] and m["polarity"] not in (pol, "both"):
            out.append(f"MODE CONTRADICTS: list is {m['polarity']} ({m['native']}), this run is {pol}")
        if "anion" in text and pol == "negative" and adduct != "[M-H]-":
            out.append(f"ion form differs: entry = the anion [M-H]-, this pair = {adduct}")
        return "; ".join(out)


def named_hits(hits) -> list:
    return [h for h in hits if h["named"]]
