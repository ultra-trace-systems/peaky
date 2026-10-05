"""The decision of the evidence scale: one level per pooled (neutral, adduct) pair.

    reagent  the neutral is a combination of the reagent's own molecules, read
             on a reagent adduct (not levelled)
    5b       rejected (an isotope check / the label twin refutes the reading,
             Low / Suspect, below assignability by an implausible-chemistry
             setter -- O>=11 alone is a degeneracy statement, not a rejection --
             or the committed reading's own isotopes contradict it), or nothing
             could be enumerated or tested
    5a       a competitor ion is left in the calibrated window (step 1)
    4b       ion established; the split is open, or pinned without a positive
             fact, or the channel reads the ion only
    4a       ion established, split pinned (step 2) and a positive fact (an own
             in-band isotope line of the neutral's elements, the 15N label, an
             NH4 adduct tracking its parent)
    3c       ion established, split pinned and a NAMED context-list entry

How it runs (`decide_source`): step 0 + step 1 + step 2 are decided by an
internal pass (`inpass`) that also still reads two routes, other-source partners
and CH2 / CF2 ladders as route / ladder levels: those internal levels are never
output, but the route-anchored pairs anchor the series exclusion of step 1,
which is iterated with them to a fixed point (<= 20 passes). `relevel` then
reads 3c / 4a / 4b off the facts; reagent, 5b, 5a and the ion-only 4b pass
through. `records` writes the per-pair record: the level, the evidence string,
what would lift it, the competitors left, the tags and the fact columns.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from peaky.assignment.levels import lists as LS
from peaky.assignment.levels import routes as RT
from peaky.assignment.levels import space as SP
from peaky.assignment.levels import split as SPL
from peaky.chem import chemistry as C

# ---------------------------------------------------------------------------
# step 0: the reagent bucket (hard-coded for the nitrate and uronium reagents; see BACKLOG)
# ---------------------------------------------------------------------------
REAGENT_MOL = {"negative": {"HNO3": {"H": 1, "N": 1, "O": 3}, "H^NO3": {"H": 1, "^N": 1, "O": 3},
                            "HBr": {"H": 1, "Br": 1}, "H2O": {"H": 2, "O": 1}},
               "positive": {"CH4N2O": {"C": 1, "H": 4, "N": 2, "O": 1}, "NH3": {"N": 1, "H": 3},
                            "H2O": {"H": 2, "O": 1}}}
REAGENT_ADDUCTS = {"negative": {"[M-H]-", "[M+NO3]-", "[M+^NO3]-", "[M+Br]-"},
                   "positive": {"[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"}}
CLUSTER_ADDUCTS = {"[M+NO3]-", "[M+^NO3]-", "[M+Br]-", "[M+(CH4N2O)H]+", "[M+NH4]+"}


def reagent_identity(n, a, pol) -> str:
    """The reagent decomposition text ('HBr', 'urea + H2O', ...) when the
    neutral is a non-negative combination of the reagent's own molecules with
    >= 1 non-water molecule (water alone only on a cluster adduct), read on a
    reagent adduct; else ''."""
    if a not in REAGENT_ADDUCTS[pol]:
        return ""
    c = RT.fcounts(n)
    if not c:
        return ""
    c = dict(c)
    if pol == "negative":
        if set(c) - {"H", "N", "^N", "O", "Br"}:
            return ""
        x14, x15, xb = c.get("N", 0), c.get("^N", 0), c.get("Br", 0)
        w = c.get("O", 0) - 3 * (x14 + x15)
        if w < 0 or c.get("H", 0) != x14 + x15 + xb + 2 * w:
            return ""
        parts = [(x14, "HNO3"), (x15, "H^NO3"), (xb, "HBr"), (w, "H2O")]
        main = x14 + x15 + xb
    else:
        if set(c) - {"C", "H", "N", "O"}:
            return ""
        u = c.get("C", 0)
        nh3 = c.get("N", 0) - 2 * u
        w = c.get("O", 0) - u
        if nh3 < 0 or w < 0 or c.get("H", 0) != 4 * u + 3 * nh3 + 2 * w:
            return ""
        parts = [(u, "urea"), (nh3, "NH3"), (w, "H2O")]
        main = u + nh3
    if main == 0 and not (w > 0 and a in CLUSTER_ADDUCTS):
        return ""
    return " + ".join(f"{k if k > 1 else ''}{m}" if k > 1 else m for k, m in parts if k)


# ---------------------------------------------------------------------------
# step 0: below_assignability setters (parsed from tier_reason || commentary; see BACKLOG)
# ---------------------------------------------------------------------------
O11_SETTER = "O-count beyond validated chemistry (O>=11, mass-saturated)"
O11_NOTE = "O>=11 on a mass-saturated window (degeneracy statement, not a rejection)"
BELOW_SETTERS = [
    (O11_SETTER, r"below-assignability \(O>=11", "implausible"),
    ("O-monster", r"oxygen-lattice monster", "implausible"),
    ("carbon cluster", r"carbon cluster \(DBE/C", "implausible"),
    ("(H+F)/C", r"implausibly carbon-rich", "implausible"),
    ("implausible ionization", r"pure hydrocarbon via|N-only neutral via", "implausible"),
    ("off-calibration residual", r"speculative residual fit -- off-calibration", "implausible"),
    ("unconfirmed fluorine", r"unconfirmed fluorine", "implausible"),
]
DEGEN_TEXTS = [
    ("mass-degenerate", r"mass-degenerate: \d+ plausible formulas share this mass|mass-saturated window \(degeneracy audit\)"),
    ("near-tie", r"alternative\(s\) within \S+ effective score and no isotope / cross-channel|near-tie: best alternative"),
]


def _txt(v) -> str:
    """A text cell: NaN / NA / None / 'nan' / '<NA>' read as ''."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return "" if s in ("nan", "<NA>") else s


def classify_below_row(tier_reason, commentary):
    t = f"{_txt(tier_reason)} || {_txt(commentary)}"
    setters = [lab for lab, pat, _k in BELOW_SETTERS if re.search(pat, t)]
    degen = [lab for lab, pat in DEGEN_TEXTS if re.search(pat, _txt(tier_reason))]
    klass = "implausible" if setters else "degeneracy/tie" if degen else "unclassified"
    return klass, setters, degen


def below_classes(per_file: dict) -> dict:
    """{(neutral, adduct): dict(klass, setters, degen, n_rows)} over the pair's
    below_assignability M0 rows in every file; setters = the sorted union."""
    from peaky.assignment import evidence as EV
    acc = defaultdict(lambda: dict(klass=set(), setters=set(), degen=set(), n_rows=0))
    for _sid, led in per_file.items():
        if "below_assignability" not in led.columns:
            continue
        m = led[(led["role"] == "M0") & led["below_assignability"].map(EV.truthy)]
        for r in m.itertuples(index=False):
            klass, setters, degen = classify_below_row(getattr(r, "tier_reason", ""), getattr(r, "commentary", ""))
            a = acc[(str(r.neutral_formula), str(r.adduct))]
            a["klass"].add(klass)
            a["setters"] |= set(setters)
            a["degen"] |= set(degen)
            a["n_rows"] += 1
    out = {}
    for k, a in acc.items():
        ks = a["klass"]
        out[k] = dict(klass="implausible" if "implausible" in ks else "unclassified" if "unclassified" in ks
                      else "degeneracy/tie", setters=sorted(a["setters"]), degen=sorted(a["degen"]), n_rows=a["n_rows"])
    return out


def _b(v) -> bool:
    from peaky.assignment import evidence as EV
    return bool(EV.truthy(v)) if not isinstance(v, (bool, np.bool_)) else bool(v)


# ---------------------------------------------------------------------------
# the internal pass (steps 0-2 + the route / ladder anchors of the series exclusion)
# ---------------------------------------------------------------------------
# the in-pass route aliases (Y = X - L explains both route ions); a locked side channel's alias never applies
ROUTE_ALIAS = {"negative": [("formate", {"C": 1, "H": 2, "O": 2}, "deprotonation")],
               "positive": [("NH4+", {"N": 1, "H": 3}, "protonation")]}
MAX_PASSES = 20


class Prepared:
    """One source after pass B, in the form the decision reads: ``P`` (one row
    per pair: the pair facts + pass B), ``comps`` (competitor rows), ``pf``
    ({sid: ledger}), ``ctx`` (levels.context.RunContext), ``lists``
    (ContextLists), ``ts`` (route co-variation groups or None), ``tol_ppm``
    (the ladder spacing window), ``pol``, ``run_classes`` (the source's own
    route classes, locked side classes removed), ``below`` (below_classes),
    ``alien`` (rule K 14N pairs), ``arm`` (one-file minima: route co-files 1,
    ladder files 1, no partners), ``skip_m0``, ``gate`` (AmineGate | None),
    ``lead_by`` ({(n, a): setter text})."""

    def __init__(self, *, name, P, comps, pf, ctx, lists, ts, tol_ppm, pol, run_classes, below, alien, arm,
                 skip_m0, gate=None, lead_by=None):
        self.name, self.P, self.comps, self.pf, self.ctx, self.lists = name, P.reset_index(drop=True), comps, pf, ctx, lists
        self.ts, self.tol_ppm, self.pol, self.run_classes = ts, tol_ppm, pol, run_classes
        self.below, self.alien, self.arm, self.skip_m0, self.gate = below, alien, arm, skip_m0, gate
        self.lead_by = lead_by or {}


def run_classes_of(engine_channels) -> set:
    """The source's own route classes: those of its reagent profile's adducts,
    minus a locked side channel's class."""
    rc = {RT.route_class(x) for x in engine_channels} - {None}
    return rc - {cl for cl, nm in RT.SIDE_CLASSES.items() if SPL.is_locked(nm)}


# C47 (rule changes the user accepted on 2026-10-05 after the scale landed):
#  * `lowconf` ALONE (every row of the pair Low / Suspect, no other rejection) is
#    a 5a CEILING, not a rejection: the pair is levelled on its facts and reads at
#    most 5a -- "not established", where 5b says "refuted". It still anchors
#    nothing (no member of the series exclusion, no route, no ladder). On the two
#    Orbitrap runs the scale was validated on, lowconf alone was the only reason
#    of 86 (labelled nitrate) and 174 (uronium; 174 of its 185 5b pairs) 5b pairs,
#    and the C42 recentring alone moved 40 of them out and 13 in -- the flag
#    tracks the calibration centre more than the chemistry.
#  * a reference list that rescued a dim reading (a tentative lead whose setter
#    is `reflist_dim`) may not also certify it at 3c through its named entry:
#    the pair takes the level its other facts give (tag `3c withheld`).
LOWCONF_REASON = "lowconf"
LOWCONF_CEILING_WHY = ("engine confidence Low/Suspect in every file: not established (5a ceiling; a file at "
                       "Good or High lifts it), not refuted")
LEAD_RESCUE_SETTER = "reflist_dim"
LIFT_3C_WITHHELD = ("3c withheld: the named entry is on the list that rescued this reading -- an independent named "
                    "list, or MS2 / standards")


def lead_rescued(lead_by) -> bool:
    """Did a reference list rescue this pair (a `lead_by` text naming the
    `reflist_dim` setter; the pooled text joins setters with ',' or '|')?"""
    t = _txt(lead_by)
    return any(x.strip() == LEAD_RESCUE_SETTER for x in re.split(r"[,|]", t)) if t else False


def inpass(S: Prepared, partners=None) -> dict:
    """The internal pass over a source. ``partners`` = {neutral: {route class:
    [partner text]}} (other-source partners; None on a partner pass and on a
    one-file source)."""
    P = S.P
    n = len(P)
    pol = S.pol
    ctx = S.ctx
    arm = S.arm
    keys = list(zip(P["neutral_formula"].astype(str), P["adduct"].astype(str)))
    nfiles = pd.to_numeric(P["n_files_obs"], errors="coerce").fillna(0).astype(int).to_numpy()
    # ---- step 0 ----
    reag = [reagent_identity(nn, aa, pol) for nn, aa in keys]
    below = [S.below.get(k, {}) for k in keys]
    rej = []
    o11_note = [""] * n
    for i, r in enumerate(P.itertuples(index=False)):
        b = [x for x in ("iso_veto", "label_veto", "lowconf") if _b(getattr(r, x))]
        if _b(r.below):
            sets = below[i].get("setters", [])
            if sets and set(sets) <= {O11_SETTER}:
                o11_note[i] = O11_NOTE
            else:
                # a below flag no setter text explains is rejected too (as built; see BACKLOG)
                b.append("below: " + (" & ".join(sets) if sets else "setter not found"))
        if _b(r.committed_contradicted):
            b.append("own isotopes: " + _txt(r.committed_reasons)[:160])
        rej.append(b)
    no_info = P["no_comp_info"].map(_b).to_numpy()
    ion_only = P["ion_only_reading"].map(_b).to_numpy()
    matched_els = [set(x for x in _txt(v).split(",") if x) for v in P["committed_matched"]]
    # ---- step 1: the competitors left by the isotope tests (route-excluded ones are never excluded) ----
    pos = {k: i for i, k in enumerate(keys)}
    left_of = defaultdict(list)
    if S.comps is not None and len(S.comps):
        for r in S.comps.itertuples(index=False):
            i = pos.get((str(r.neutral_formula), str(r.adduct)))
            if i is None or str(r.status) != "left":
                continue
            left_of[i].append(dict(name=str(r.competitor), kind=str(r.kind), neutral=_txt(r.comp_neutral),
                                   adduct=str(r.comp_adduct) if _txt(r.comp_adduct) else str(r.adduct)))
    # ---- step 2 ----
    sp = [SPL.split(S, r) for r in P.itertuples(index=False)]
    pinned = np.array([x["pinned"] for x in sp], dtype=bool)
    label_pin = np.array([x["label_pin"] for x in sp], dtype=bool)
    elements = [set((RT.fcounts(nn) or {}).keys()) for nn, _ in keys]
    posfact = np.array([bool(label_pin[i]) or bool(matched_els[i] & elements[i]) for i in range(n)], dtype=bool)
    hits = [S.lists.hits(nn) for nn, _ in keys]
    listed = np.array([bool(h) for h in hits], dtype=bool)
    # C47: the list that rescued a dim reading may not certify it at 3c
    lead_reflist = np.array([lead_rescued(S.lead_by.get(k, "")) for k in keys], dtype=bool)
    # C47: lowconf alone is a 5a ceiling -- the pair is still no anchor for others (rej stays)
    lowconf_only = np.array([bool(x) and all(y == LOWCONF_REASON for y in x) for x in rej], dtype=bool)
    # ---- routes (in-pass anchors; tags in the record) ----
    rejected_mask = np.array([bool(x) for x in rej], dtype=bool)
    nonrej = ~rejected_mask & ~no_info & np.array([not x for x in reag], dtype=bool)
    committed_nonrej = {keys[i][0] for i in range(n) if nonrej[i]}
    route_info = [None] * n
    routes2 = np.zeros(n, bool)
    excl = {k for k, v in zip(keys, P["committed_contradicted"].map(_b)) if v} | set(S.alien)
    excl |= {k for k, a_, b_ in zip(keys, P["iso_veto"].map(_b), P["label_veto"].map(_b)) if a_ or b_}
    excl |= {keys[i] for i in range(n) if rejected_mask[i] or reag[i]}
    cof = 1 if arm else RT.ROUTE_COFILES
    rt, byfile = RT.routes_table(S.pf, S.run_classes, excl, cof, S.skip_m0)
    aliases = [x for x in ROUTE_ALIAS.get(pol, []) if not SPL.is_locked(x[0])]
    for i, (nn, aa) in enumerate(keys):
        own_cls = RT.route_class(aa)
        info = dict(own=[], cross=[], alias=[], r={}, failed=[])
        route_info[i] = info
        if own_cls is None or (nn, aa) in excl or ion_only[i]:
            continue
        t = rt.get(nn)
        if t is not None:
            for (A, B), nco in t["co"].items():
                if own_cls not in (A, B):
                    continue
                if nco < cof:
                    info["failed"].append(f"{A}+{B} co-seen in {nco} file(s)")
                    continue
                al = ""
                for aname, L, base in aliases:
                    if base not in (A, B):
                        continue
                    Y = RT.fsub(RT.fcounts(nn), L)
                    if not Y:
                        continue
                    okY, _w = ctx.space.plausible_neutral(Y)
                    yf = C.format_formula(Y)
                    if okY and yf in committed_nonrej:
                        al = f"{aname} alias {yf} (X - {C.format_formula(L)}) explains both route ions"
                        break
                rbest, nbest = np.nan, 0
                for a1 in sorted(t["adducts"].get(A, ())):
                    for a2 in sorted(t["adducts"].get(B, ())):
                        rr_, nn_ = RT.covary(S.ts, (nn, a1), (nn, a2))
                        if nn_ > 0 and (not np.isfinite(rbest) or (np.isfinite(rr_) and rr_ > rbest)):
                            rbest, nbest = rr_, nn_
                info["r"][(A, B)] = (rbest, nbest, nco)
                if al:
                    info["alias"].append(((A, B), al))
                else:
                    info["own"].append((A, B))
        if partners and nfiles[i] >= RT.CROSS_MIN_FILES and not arm and pinned[i]:
            for cl, srcs in partners.get(nn, {}).items():
                if cl in S.run_classes:
                    continue
                info["cross"].append((cl, "; ".join(srcs[:2])))
        elif partners and nfiles[i] >= RT.CROSS_MIN_FILES and not arm and not pinned[i]:
            lost = [cl for cl in partners.get(nn, {}) if cl not in S.run_classes]
            if lost:
                info["cross_refused"] = ("other-source route (" + ", ".join(sorted(lost)) + ") not counted: "
                                         "this ion's own split is open")
        self_ok = _b(P.at[i, "committed_plausible"])
        routes2[i] = bool((info["own"] or info["cross"]) and self_ok)
        if (info["own"] or info["cross"]) and not self_ok:
            info["blocked"] = "routes not counted: the committed neutral is not a plausible decomposition"
    heights = pd.to_numeric(P["height"], errors="coerce").fillna(0.0).to_numpy()
    br_own = RT.base_rate_own(pol, S.run_classes, byfile, keys, heights, cof)
    elig = np.array([nonrej[i] and not ion_only[i] and RT.route_class(keys[i][1]) is not None
                     and nfiles[i] >= RT.CROSS_MIN_FILES and pinned[i] for i in range(n)])
    br_cross = RT.base_rate_cross(pol, S.run_classes, partners, keys, elig) if (partners and not arm) else None
    # ---- iterate: series exclusion + ladder (anchors = route-anchored pairs) ----
    member = nonrej
    member_ions = set()
    for i in np.flatnonzero(member):
        k = RT.ion_key(keys[i][0], keys[i][1], P.at[i, "ion"] if "ion" in P.columns else None)
        if k:
            member_ions.add(k)
    D = P[["neutral_formula", "adduct", "mz"]].copy()
    lmin = 1 if arm else RT.LADDER_MIN_FILES

    def inpass_level(i, left, homo):
        """The internal pass's natural (level, why) of pair i."""
        if reag[i]:
            return "reagent", "reagent identity (" + reag[i] + ")"
        if rej[i] and not lowconf_only[i]:
            return "5b", "rejected: " + "; ".join(rej[i])
        # a tentative lead has no level effect (the lead gate is off)
        if no_info[i]:
            return "5b", "untestable"
        if left:
            return "5a", f"competitors left ({len(left)})"
        if ion_only[i]:
            return "4b", "ion formula only"
        if routes2[i]:
            return ("3a" if listed[i] else "3b"), "routes"
        h = homo.get(i)
        if h and h["anchored"] and pinned[i] and nfiles[i] >= lmin:
            return ("3a" if listed[i] else "3d"), "ladder"
        if pinned[i] and listed[i] and not lead_reflist[i]:
            return "3c", "split pinned + listed"
        if pinned[i] and posfact[i]:
            return "4a", "split pinned + positive fact"
        return "4b", ("split pinned, no positive fact" if pinned[i] else "split open")

    def one_pass(excl_s, homo):
        lv = [""] * n
        why = [""] * n
        lefts = [None] * n
        for i in range(n):
            left = [c for c in left_of.get(i, []) if c["name"] not in excl_s.get(i, {})]
            lefts[i] = left
            lv[i], why[i] = inpass_level(i, left, homo)
            if lowconf_only[i] and lv[i] not in ("reagent", "5b", "5a"):
                # C47: the ceiling -- its step-1/2 facts stand, the level reads 5a
                lv[i], why[i] = "5a", LOWCONF_CEILING_WHY
        return lv, why, lefts

    excl_s, homo = {}, {}
    lv, why, lefts = one_pass(excl_s, homo)
    it = 0
    for it in range(1, MAX_PASSES + 1):
        anchor = np.array([l_ in ("3a", "3b") and w == "routes" for l_, w in zip(lv, why)])
        D["_lv"] = lv
        homo = RT.homologue_facts(D, member, anchor, tol_ppm=S.tol_ppm, units=RT.LADDER_UNITS)
        left_nonrej = {i: v for i, v in left_of.items() if member[i]}
        new_excl = RT.series_exclusions(D, left_nonrej, homo, member_ions)
        lv2, why2, lefts2 = one_pass(new_excl, homo)
        conv = (new_excl == excl_s) and (lv2 == lv)
        excl_s, lv, why, lefts = new_excl, lv2, why2, lefts2
        if conv:
            break
    return dict(level=lv, why=why, lefts=lefts, excl_s=excl_s, homo=homo, routes2=routes2, route_info=route_info,
                sp=sp, pinned=pinned, posfact=posfact, listed=listed, hits=hits, reag=reag, rej=rej,
                lead_reflist=lead_reflist, lowconf_only=lowconf_only,
                o11_note=o11_note, iterations=it, nfiles=nfiles, br_own=br_own, br_cross=br_cross)


def inpass_texts(S: Prepared, res: dict) -> list[dict]:
    """Per pair: the ion text, the internal pass's blocked text (5a / 5b), its
    tags (O>=11, series exclusion, 5a's suspect match / tie, an open 4b's
    'list favours X') and its competitor list after series exclusion."""
    P = S.P
    out = []
    for i, r in enumerate(P.itertuples(index=False)):
        L = res["level"][i]
        why = res["why"][i]
        nn, aa = str(r.neutral_formula), str(r.adduct)
        left = res["lefts"][i] or []
        nser = len(res["excl_s"].get(i, {}))
        n_iso = int(pd.to_numeric(r.n_excl_iso, errors="coerce") or 0)
        ncomp = int(pd.to_numeric(r.n_competitors, errors="coerce") or 0)
        if _b(r.no_comp_info):
            ion_txt = "untestable (no ion composition / outside the enumerable space)"
        elif ncomp == 0:
            ion_txt = "unique in the calibrated window"
        else:
            parts = [f"{ncomp} competitor(s)"]
            if n_iso:
                parts.append(f"isotopes exclude {n_iso}")
            if nser:
                parts.append(f"series excludes {nser}")
            parts.append(f"{len(left)} left" if left else "none left")
            ion_txt = "; ".join(parts)
        lines = _txt(r.committed_matched_lines)
        if lines:
            ion_txt += f"; own lines in band: {lines}"
        if _b(r.committed_contradicted):
            ion_txt += " [committed contradicted]"
        pinned = res["sp"][i]["pinned"]
        plaus = res["sp"][i]["plaus"]
        pre, post = [], []          # the internal pass's tags before / after the (moved) lead tag
        if L == "5a":
            cands = [dict(neutral=nn, name=f"{nn} {aa}")] + [c for c in left if c.get("neutral")]
            on = [(c, S.lists.hits(c["neutral"])) for c in cands]
            on = [(c, hh) for c, hh in on if hh]
            if len(on) == 1:
                pre.append(f"suspect match: {on[0][0]['name']} [{'; '.join(x['id'] for x in on[0][1])}]")
            if _b(r.tied):
                post.append("tie")
        if res["o11_note"][i]:
            post.append(res["o11_note"][i])
        if nser:
            post.append(f"series exclusion x{nser}")
        # 'list favours X' reads the INTERNAL level (an open in-pass 4b), as built
        if L == "4b" and not pinned and not _b(r.ion_only_reading):
            onl = {d["neutral"] for d in plaus if S.lists.hits(d["neutral"])}
            if len(onl) == 1:
                post.append(f"list favours {next(iter(onl))}")
        if L == "reagent":
            blk = "reagent ion / reagent cluster (not levelled)"
        elif L == "5b":
            blk = why.replace("rejected: ", "refuted: ") if why.startswith("rejected") else \
                "nothing could be enumerated or tested"
        elif L == "5a":
            blk = ("competitors left: " + "; ".join(c["name"] for c in left[:6])
                   + (f" (+{len(left) - 6} more)" if len(left) > 6 else "")) if left else why
        else:
            blk = ""
        out.append(dict(ion_txt=ion_txt, blocked=blk, tags_pre=pre, tags_post=post,
                        competitors_left="; ".join(c["name"] for c in left), n_left=len(left), n_series_excl=nser))
    return out


# ---------------------------------------------------------------------------
# the scale's level
# ---------------------------------------------------------------------------
def relevel(S: Prepared, res: dict) -> list[str]:
    """3c / 4a / 4b from the facts; reagent, 5b, 5a and the ion-only 4b pass
    through from the internal pass."""
    out = []
    for i in range(len(S.P)):
        L, why = res["level"][i], res["why"][i]
        if L in ("reagent", "5b", "5a") or why == "ion formula only":
            out.append(L)
            continue
        pin = bool(res["pinned"][i])
        if pin and LS.named_hits(res["hits"][i]) and not res["lead_reflist"][i]:
            out.append("3c")
            continue
        if pin and (res["posfact"][i] or res["sp"][i]["gi"].get("track", False)):
            out.append("4a")
            continue
        out.append("4b")
    return out


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
# the scale's v1 route aliases (tag texts): Y = X - L explains both route ions
ROUTE_ALIAS_TAG = {"negative": [("formate", {"C": 1, "H": 2, "O": 2}, "deprotonation")],
                   "positive": [("NH4+", {"N": 1, "H": 3}, "protonation")]}
LEAD_TAG = "a tentative lead: not answered by an own isotope line"
LEVEL_TEXT = {"3c": "3c (split pinned + named context-list entry)", "4a": "4a (split pinned + positive fact)",
              "4b": "4b", "5a": "5a", "5b": "5b", "reagent": "reagent bucket"}
MODE_FLAG_ALL = (" [FLAG: every named entry's source mode contradicts this run (MODE CONTRADICTS); flag only, "
                 "not a block]")


#: the record's columns after (neutral_formula, adduct), in order: the scale's columns but `claim`, then the step
#: facts (`iterations` last); a source with no pair levels to an empty frame with them (source.empty_levels)
RECORD_COLUMNS = ("evidence_level", "evidence", "would_lift", "competitors_left", "tags", "context", "context_source",
                  "tag_kinds", "split_pinned", "split_how", "positive_fact", "named_list", "named_mode_flag",
                  "window_only", "window_isobar", "chloride_open", "nh4_gate", "nh4_admissible", "nh4_inadmissible",
                  "nh4_gate_detail", "side_aliases", "route_alias", "anchor_kind", "anchor_why", "n_left_inpass",
                  "n_series_excl", "inpass_level", "inpass_why", "iterations")


def records(S: Prepared, res: dict, texts: list[dict], lv: list[str]) -> pd.DataFrame:
    """The per-pair record: the scale's columns + the step facts."""
    P = S.P
    n = len(P)
    pol = S.pol
    gate = S.gate
    keys = list(zip(P["neutral_formula"].astype(str), P["adduct"].astype(str)))
    no_info = P["no_comp_info"].map(_b).to_numpy()
    nonrej = [not res["rej"][i] and not no_info[i] and not res["reag"][i] for i in range(n)]
    committed_nonrej = {keys[i][0] for i in range(n) if nonrej[i]}
    br_own_txt = RT.br_text((res.get("br_own") or {}).get("all"), "base rate of 'seen in both channels'") \
        or "base rate n/a"
    br_cross_txt = RT.br_text((res.get("br_cross") or {}).get("all"), "base rate of 'other-source partner'") \
        or "base rate n/a"
    cols = defaultdict(list)
    for i, (nn, aa) in enumerate(keys):
        L = lv[i]
        r = P.iloc[i]
        t = texts[i]
        ri = res["route_info"][i] or {}
        s = res["sp"][i]
        pinned = bool(s["pinned"])
        split_how = s["how"]
        gi = s["gi"]
        tags, tagk = [], []
        # --- routes ---
        own_ok, alias_txt = [], []
        for (A, B) in ri.get("own", []):
            al = ""
            for aname, Lf, base in ROUTE_ALIAS_TAG.get(pol, []):
                if base not in (A, B):
                    continue
                Y = RT.fsub(RT.fcounts(nn), Lf)
                if not Y:
                    continue
                okY, _w = S.ctx.space.plausible_neutral(Y)
                yf = C.format_formula(Y)
                if not (okY and yf in committed_nonrej):
                    continue
                if aname == "formate":
                    al = f"formate alias {yf} (X - CH2O2) would explain both route ions (side channel locked: not applied)"
                    tagk.append("route alias locked")
                elif gate is not None:
                    g = gate(yf)
                    if g["kept"]:
                        al = f"NH4+ alias {yf} (X - NH3) explains both route ions (amine gate keeps its NH4 reading: {g['how']})"
                    else:
                        al = f"NH4+ alias {yf} (X - NH3) not kept by the amine gate ({g['how']})"
                break
            rr = ri.get("r", {}).get((A, B))
            rtxt = f"{rr[2]} co-files; r {rr[0]:.2f}/{rr[1]}" if rr and np.isfinite(rr[0]) else \
                (f"{rr[2]} co-files; r n/a" if rr else "")
            vetoed = al.startswith("NH4+ alias") and "explains both" in al
            if al:
                alias_txt.append(al)
            if vetoed:
                tags.append(f"two routes alias-vetoed: {RT.route_label(A, B)} ({rtxt}); {al}")
                tagk.append("routes alias-vetoed")
            else:
                own_ok.append((A, B))
                tags.append(f"two routes: {RT.route_label(A, B)} ({rtxt}){' [' + al + ']' if al else ''} -- tag; "
                            f"{br_own_txt}")
        if own_ok:
            tagk.append("routes")
        # --- other-source partner ---
        own_cls = RT.route_class(aa)
        if ri.get("cross"):
            for cl, src in ri["cross"]:
                tags.append(f"other-source partner: {RT.route_label(own_cls or '?', cl, ' (other source)')}"
                            + (f" [{src}]" if src else "") + f" -- tag; {br_cross_txt}")
            tagk.append("other source")
        if ri.get("cross_refused"):
            tags.append(ri["cross_refused"])
            tagk.append("other source refused (split open)")
        # --- ladder ---
        h = res["homo"].get(i)
        if h:
            anc = [x for x in h["anchors"] if x["ok"]]
            lad = (f"{h['unit']} chain {len(h['chain'])}, {len(anc)} route-tagged anchor(s)" +
                   (": " + ", ".join(f"{x['n']} {RT.kfmt(x['k'], h['unit'])}"
                                     for x in sorted(anc, key=lambda x: x["k"])[:5]) if anc else ""))
            if h["anchored"] and pinned and res["nfiles"][i] >= RT.LADDER_MIN_FILES:
                tags.append(f"ladder: {lad} -- tag (anchors are route tags at the chance rate)")
                tagk.append("ladder")
            else:
                tags.append(f"chain: {lad} ({'not anchored' if not h['anchored'] else 'split open' if not pinned else 'seen in < 2 files'})")
        # --- lists ---
        hits = res["hits"][i]
        named = LS.named_hits(hits)
        mflags = [S.lists.mode_flag(hh, nn, pol, aa) for hh in named]
        named_txt = [f"{hh['id']} = {hh['name']}" + (f" [{mfl}]" if mfl else "") for hh, mfl in zip(named, mflags)]
        # the level text needs EVERY named entry flagged, the tag ANY (as built; see BACKLOG)
        mode_block = L == "3c" and bool(named) and all("MODE CONTRADICTS" in x for x in mflags)
        cls = [hh for hh in hits if not hh["named"]]
        if cls:
            ctx_txt = []
            for hh in cls:
                mfl = S.lists.mode_flag(hh, nn, pol, aa)
                ctx_txt.append(f"{hh['id']}" + (f" [{mfl}]" if mfl else ""))
            tags.append("class list: " + "; ".join(ctx_txt) + " -- tag (formula-only / class entry)")
            tagk.append("class list")
        if named and L != "3c" and pinned and res["lead_reflist"][i] and L in ("4a", "4b"):
            # C47: the list that rescued the reading (lead_by reflist_dim) cannot also certify it
            tags.append("3c withheld: the named entry is on the list that rescued this reading (lead: "
                        f"{LEAD_RESCUE_SETTER}); the pair takes the level its other facts give: "
                        + "; ".join(named_txt))
            tagk.append("3c withheld (list rescued the lead)")
        elif named and L != "3c":
            tags.append("named list (not used: " + ("split open" if not pinned else "ion not established") + "): "
                        + "; ".join(named_txt))
            tagk.append("named list, not 3c")
        if L == "3c" and any("MODE CONTRADICTS" in x for x in mflags):
            tagk.append("3c name: mode contradicts")
            tags.append("FLAG, named entry's source mode contradicts this run: " + "; ".join(
                f"{hh['name']} ({x})" for hh, x in zip(named, mflags) if "MODE CONTRADICTS" in x)
                + " -- flag only, does not block 3c")
        # --- split ---
        if "side channels locked" in split_how:
            tagk.append("side channels locked (split)")
            tags.append("side channels locked")
        if gi.get("xside_open"):
            tagk.append("chloride would open the split (locked)")
        if gi.get("isobar"):
            tagk.append("X+reagent isobar admitted over the window")
            tags.append(f"X+reagent isobar admitted over the context window ({gi['isobar']}) -- the context window "
                        "does not exclude the X+reagent isobar")
        adm_other = [(Y, ga) for Y, ga in gi.get("adm", []) if ga["present"] and not ga["own_name"]]
        if adm_other:
            tagk.append("NH4 reading admissible via Y's ion under another reading")
            tags.append("NH4 reading admissible (NH4 admissibility rule): Y's uronium adduct ion is present under "
                        "another reading: " + "; ".join(
                            f"{Y} -- {ga['why'].replace('uronium adduct ion present: ', '')}" for Y, ga in adm_other))
        if gi.get("inadm"):
            tagk.append("NH4 reading inadmissible")
            tags.append("NH4 reading inadmissible (NH4 admissibility rule): "
                        + "; ".join(f"{Y} [M+NH4]+ -- {w}" for Y, w, _g in gi["inadm"]))
        if gi.get("window") and pinned:
            tagk.append("pinned only by the context window")
            tags.append(f"pinned only by the context window ({gi['window']}) -- a window lifted would open the split")
        k = gi.get("kind", "")
        if k == "amine default":
            tagk.append("amine default")
            tags.append("amine default (NH4 reading unconfirmed)")
        elif k.startswith("NH4 kept"):
            tagk.append("NH4 reading kept by the gate")
            tags.append("NH4 reading kept by the amine gate")
        if k in ("NH4 dropped", "NH4 kept+dropped"):
            tagk.append("NH4 reading dropped by the gate")
        if gi.get("track"):
            tagk.append("NH4 adduct tracks its parent")
        # --- the internal pass's tags; the tentative lead on every level ---
        tags += t["tags_pre"]
        if res["lowconf_only"][i] and L == "5a":
            # C47: lowconf alone -- the ceiling, printed whether or not it bit
            tags.append("engine confidence Low/Suspect in every file: 5a ceiling (not established, not refuted)")
            tagk.append("lowconf 5a ceiling")
        if _b(r["lead_fact"]):
            lb = S.lead_by.get((nn, aa), "")
            tags.append(f"lead: {lb} ({LEAD_TAG})" if lb else f"lead ({LEAD_TAG})")
        tags += t["tags_post"]
        # --- positive fact ---
        pf = []
        if s["label_pin"]:
            pf.append("15N label")
        lines = _txt(r["committed_matched_lines"])
        els = {x for x in _txt(r["committed_matched"]).split(",") if x} & set((RT.fcounts(nn) or {}).keys())
        if els:
            pf.append(f"own in-band isotope line(s) {lines} ({','.join(sorted(els))} of the neutral)")
        if gi.get("track"):
            pf.append("NH4 adduct tracks its parent")
        pf_txt = "; ".join(pf)
        # --- context ---
        context = S.lists.context(hits)
        csrc = S.lists.context_source(hits)
        lvl_txt = LEVEL_TEXT[L] + (MODE_FLAG_ALL if (L == "3c" and mode_block) else "")
        if L in ("5a", "5b", "reagent"):
            ev = f"{lvl_txt} · {res['why'][i]}" + (f" · ion: {t['ion_txt']}" if t["ion_txt"] else "") \
                + f" · split: {split_how}"
        else:
            ev = (f"{lvl_txt} · ion: {t['ion_txt']} · split: {'PINNED' if pinned else 'open'} -- {split_how}"
                  f" · positive fact: {pf_txt or 'none'} · named list: {'; '.join(named_txt) if named_txt else 'none'}")
        ev += f" · context source: {csrc}"
        ev += " · tags: " + (" | ".join(tags) if tags else "none")
        # --- what would lift it ---
        if L == "reagent":
            blk = "reagent ion / reagent cluster (not levelled)"
        elif L in ("5b", "5a"):
            blk = t["blocked"]
        elif L == "4b":
            if _b(r["ion_only_reading"]) or res["why"][i] == "ion formula only":
                blk = "ion-only channel: the neutral and the process stay open"
            elif not pinned:
                blk = "split not pinned: " + split_how
            else:
                blk = ("4a needs a positive fact (an own in-band isotope line of the neutral's elements, the 15N "
                       "label, or NH4 tracking)")
                if not named:
                    blk += "; 3c needs a named context-list entry"
        elif L == "4a":
            if named and res["lead_reflist"][i]:
                blk = LIFT_3C_WITHHELD
            else:
                blk = "3c needs a NAMED context-list entry naming the neutral" + (
                    " (the class-list match is a tag only)" if cls else "")
        else:
            blk = "level 2 (MS2 / standards) is not automatic"
        cols["evidence_level"].append(L)
        cols["evidence"].append(ev)
        cols["would_lift"].append(blk)
        cols["competitors_left"].append(t["competitors_left"])
        cols["tags"].append(" | ".join(tags))
        cols["context"].append(context)
        cols["context_source"].append(csrc)
        cols["tag_kinds"].append("|".join(sorted(set(tagk))))
        cols["split_pinned"].append(pinned)
        cols["split_how"].append(split_how)
        cols["positive_fact"].append(pf_txt)
        cols["named_list"].append("; ".join(named_txt))
        cols["named_mode_flag"].append("; ".join(x for x in mflags if x))
        cols["window_only"].append(gi.get("window", "") if pinned else "")
        cols["window_isobar"].append(gi.get("isobar", ""))
        cols["chloride_open"].append(gi.get("xside_open", ""))
        cols["nh4_gate"].append(k)
        cols["nh4_admissible"].append("; ".join(f"{Y}: {ga['why']}" for Y, ga in gi.get("adm", [])))
        cols["nh4_inadmissible"].append("; ".join(f"{Y}: {w} (gate: {g['verdict']})" for Y, w, g in gi.get("inadm", [])))
        cols["nh4_gate_detail"].append("; ".join(f"{Y}: {g['verdict']} ({g['how']})" for Y, g in gi.get("nh4", [])))
        cols["side_aliases"].append(s["flags"])
        cols["route_alias"].append("; ".join(alias_txt))
        cols["anchor_kind"].append(RT.anchor_kind(res["level"][i], res["why"][i]))
        cols["anchor_why"].append(RT.anchor_why(res["level"][i], res["why"][i]))
        cols["n_left_inpass"].append(t["n_left"])
        cols["n_series_excl"].append(t["n_series_excl"])
        # the raw in-pass tokens: in memory only (scale.INTERNAL_COLUMNS; partners_from reads them)
        cols["inpass_level"].append(res["level"][i])
        cols["inpass_why"].append(res["why"][i])
    extra = set(cols) - set(RECORD_COLUMNS)
    if extra:
        raise AssertionError(f"record columns missing from RECORD_COLUMNS: {sorted(extra)}")
    out = pd.DataFrame({c: cols[c] for c in RECORD_COLUMNS if c != "iterations"})
    out.insert(0, "adduct", [k[1] for k in keys])
    out.insert(0, "neutral_formula", [k[0] for k in keys])
    out["iterations"] = res["iterations"]
    return out
