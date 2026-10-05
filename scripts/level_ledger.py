"""The evidence scale of peaky 0.10.0, restated: the executable reference.

The engine levels every committed (neutral_formula, adduct) pair of a source
(`peaky.assignment.evidence`: the batch's pooled files, one file alone, or one
file in a main run's context). This script shares the engine's FACT layer --
the source, the run context, the competitors and their isotope tests, the
decompositions, the amine gate's verdicts, the context lists, the routes table
and the CH2 / CF2 series -- and states the DECISION layer again, on its own,
from the build document of the scale: the order of step 0, the step-1 outcome,
the split's outcome per rule, the level and the would-lift texts. It never
calls `peaky.assignment.levels.decide`. The engine must give the same level and
the same would-lift text as this script on every pair (tests/test_level_ledger.py
on synthetic sources and their mutants; outside the repo on real runs).

    python scripts/level_ledger.py <run dir>... [--corroborate <run dir>...]
                                   [--mode run|adapted|strict] [--main <run dir>]
                                   [--resolving-power R --reagent NAME [--context NAME]]
                                   [--out levels.csv] [--vector]

A run dir holds merged_ledger.csv, per_file/*_ledger.csv,
per_file/_batch_ts.parquet, batch_summary.json and tables/{iso_checks,
label_twins}.csv; it is levelled as one pooled source ('run', every file-count
minimum 3). A ledger CSV is one file: levelled alone ('adapted', the minima 1;
'strict', the batch minima) or, with --main, in that run's context exactly as a
decoy arm. A lone ledger has no width model: it reads NA unless
--resolving-power (R at m/z 200) and --reagent (the profile) are given; it is
then levelled as the per-file stage of a single-sample `peaky assign`: the
profile's context (--context overrides it), the lists that context and the
profile label activate, the profile channels' halogen and the --window
calibration (MU,SIGMA; else refitted from its degeneracy counts). Lists a
batch or dataset name activated are not known to a lone ledger.
--corroborate names other-source partners: each Orbitrap-class run dir is
levelled once with no partners and its route / ladder / listed pairs become a
TAG on the levelled pairs (they can still anchor the series exclusion of step
1, as in the engine); a TOF-class or class-less source gives none.

Output: one row per pair -- source, neutral_formula, adduct, evidence_level,
claim, would_lift, competitors_left and the decisive facts (reagent identity,
rejections, split rule and text, positive fact, named entry, the internal
pass's level). --vector prints, per source, ``n 3c/4a/4b/5a/5b/reagent/NA`` and
the share of the committed per-file M0 height per level: every per-file M0
reading takes its pair's level, a reading no merged row carries is
'unmatched', and the denominator holds them all.

The scale, in the order the outcomes are tried (the first that applies):

    reagent  the neutral is made only of the reagent's own molecules and is
             read on a reagent adduct (a bucket, not a level)
    5b       rejected (iso_veto, label_veto, lowconf, an implausible-chemistry
             below-assignability setter -- O>=11 alone is a tag -- or the
             committed reading's own isotope lines contradict it), or nothing
             could be enumerated or tested
    5a       a competitor ion is left in the calibrated window (after the
             isotope tests and the CH2 / CF2 series exclusion)
    4b       the channel reads the ion only
    3c       split pinned and a NAMED context-list entry names the neutral
    4a       split pinned and a positive fact (an own in-band isotope line of
             an element of the neutral, the 15N label, an NH4 adduct that
             tracks its parent)
    4b       everything else (split open, or pinned with no positive fact)
    NA       not assessed: the source's width model does not resolve
             >= 50 000 at m/z 200, or there is none

The pre-0.10.0 B-series reference this script used to be is kept below it,
frozen, for the merge vote's private evidence class and its tests
(`series_run`, `series_main`, `series_level_of`, `SERIES_LEVEL_ORDER`,
`measure_source`, `assign_levels`, ...). No user-facing level reads it.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from peaky.assignment import evidence as EV  # noqa: E402
from peaky.assignment.levels import routes as RT  # noqa: E402
from peaky.assignment.levels import source as SRC  # noqa: E402
from peaky.assignment.levels import space as SP  # noqa: E402
from peaky.assignment.levels import split as SPL  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402

# ===========================================================================
# the scale
# ===========================================================================
LEVEL_ORDER = list(EV.LEVEL_ORDER)          # 1 2 3c 4a 4b 5a 5b (1 and 2 never fire)
LEVELS = list(EV.LEVELS)                    # 3c 4a 4b 5a 5b
BUCKETS = list(EV.BUCKETS)                  # reagent NA
VECTOR_KEYS = LEVELS + BUCKETS              # the --vector order
MODES = ("run", "adapted", "strict")

# ===========================================================================
# step 0: sort out
# ===========================================================================
#: the reagent's own molecules, each with the element that marks it, read off in this order; water (marked by
#: O) takes the oxygen left over. Keyed by polarity, as the engine keys them.
REAGENT_MOLECULES = {
    "negative": (("HNO3", "N", {"H": 1, "N": 1, "O": 3}), ("H^NO3", "^N", {"H": 1, "^N": 1, "O": 3}),
                 ("HBr", "Br", {"H": 1, "Br": 1}), ("H2O", "O", {"H": 2, "O": 1})),
    "positive": (("urea", "C", {"C": 1, "H": 4, "N": 2, "O": 1}), ("NH3", "N", {"N": 1, "H": 3}),
                 ("H2O", "O", {"H": 2, "O": 1})),
}
#: the reagent adducts a reagent molecule is read on; water alone counts only on a cluster adduct
REAGENT_ON = {"negative": {"[M-H]-", "[M+NO3]-", "[M+^NO3]-", "[M+Br]-"},
              "positive": {"[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"}}
WATER_ON = {"[M+NO3]-", "[M+^NO3]-", "[M+Br]-", "[M+(CH4N2O)H]+", "[M+NH4]+"}
#: the rejections, in the order the evidence lists them
REJECT_FLAGS = ("iso_veto", "label_veto", "lowconf")
#: the one below-assignability setter that is a degeneracy statement (a tag), not a rejection
O11_SETTER = "O-count beyond validated chemistry (O>=11, mass-saturated)"
OWN_REASONS_MAX = 160          # the contradicting isotope reasons printed in a rejection


def txt(v) -> str:
    """A text cell; NaN / NA / None / 'nan' / '<NA>' read as ''."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return "" if s in ("nan", "<NA>") else s


def on(v) -> bool:
    """A flag cell (NaN-safe)."""
    return bool(v) if isinstance(v, (bool, np.bool_)) else bool(EV.truthy(v))


def counts_of(formula) -> dict:
    """Element counts, zero counts dropped; {} when unreadable."""
    try:
        return {k: int(v) for k, v in C.parse_formula(str(formula)).items() if v}
    except Exception:  # noqa: BLE001
        return {}


def reagent_identity(neutral, adduct, pol) -> str:
    """'HNO3', '2HNO3 + H2O', 'urea + NH3' ... when the neutral is a sum of the
    reagent's own molecules read on a reagent adduct, with >= 1 molecule that
    is not water (water alone only on a cluster adduct); else ''."""
    if adduct not in REAGENT_ON.get(pol, ()):
        return ""
    rest = counts_of(neutral)
    if not rest:
        return ""
    taken = []
    for name, marker, mol in REAGENT_MOLECULES[pol]:
        k = rest.get(marker, 0)
        for el, v in mol.items():
            rest[el] = rest.get(el, 0) - k * v
        if any(v < 0 for v in rest.values()):
            return ""
        taken.append((name, k))
    if any(v for v in rest.values()):
        return ""
    water = dict(taken).get("H2O", 0)
    if not any(k for name, k in taken if name != "H2O") and not (water > 0 and adduct in WATER_ON):
        return ""
    return " + ".join((f"{k}{name}" if k > 1 else name) for name, k in taken if k)


def rejections(row: dict, below: dict) -> tuple[list[str], bool]:
    """(the rejections of a pair, in order; whether O>=11 is its only below setter -- a tag)."""
    out = [flag for flag in REJECT_FLAGS if on(row.get(flag))]
    o11 = False
    if on(row.get("below")):
        setters = list((below or {}).get("setters") or [])
        if setters and set(setters) <= {O11_SETTER}:
            o11 = True
        else:
            # a below flag no setter explains is rejected too (as built)
            out.append("below: " + (" & ".join(setters) if setters else "setter not found"))
    if on(row.get("committed_contradicted")):
        out.append("own isotopes: " + txt(row.get("committed_reasons"))[:OWN_REASONS_MAX])
    return out, o11


# ===========================================================================
# step 2: the split -- its outcome per rule
# ===========================================================================
#: the rules in the order they are tried; the first that decides wins
SPLIT_RULES = (
    ("no ion composition", "open"),
    ("ion-only channel", "open"),
    ("label", "PINNED"),                      # the ion carries ^N on a ^N adduct (no label veto, not rule K)
    ("label on an unlabelled adduct", "open"),
    ("amine gate: committed NH4", "gate"),    # PINNED when the gate keeps it, else open (amine default)
    ("amine gate: admissible NH4 reading", "open"),
    ("only decomposition", "PINNED"),
    ("halogen count", "PINNED"),              # rule (iii): as coded it is subsumed by 'only decomposition'
    ("committed neutral not plausible", "open"),
    ("k decompositions", "open"),
)
#: the X+reagent isobar's molecules (adduct a = adduct d + R), keyed by polarity
ISOBAR_MOLECULES = {"positive": (("urea", {"C": 1, "H": 4, "N": 2, "O": 1}),),
                    "negative": (("HNO3", {"H": 1, "N": 1, "O": 3}), ("H^NO3", {"H": 1, "^N": 1, "O": 3}),
                                 ("HBr", {"H": 1, "Br": 1}))}
NH4 = "[M+NH4]+"
PROTONATED = "[M+H]+"
WINDOW_WHY = "context filter"


def grid_of(S) -> list[str]:
    """The split grid: the decomposition adducts and the side channels, minus every LOCKED side channel (the
    uronium run's own [M+NH4]+ is not a side channel there); chloride joins only when unlocked."""
    lk = {a for a, nm in SPL.SIDE_CHANNEL_NAMES.items() if SPL.is_locked(nm)} \
        - SPL.OWN_REAGENT_CHANNEL.get(S.pol, set())
    grid = [a for a in S.ctx.decomp_adducts if a not in lk]
    for a in SPL.SIDE[S.pol]:
        if a not in lk and a not in grid and SP.adduct_delta(a) is not None:
            grid.append(a)
    for a, nm in SPL.V1_SIDE_EXTRA.get(S.pol, {}).items():
        if not SPL.is_locked(nm) and a not in grid:
            grid.append(a)
    return grid


def _readings(ds, k=5) -> str:
    return "; ".join(f"{d['neutral']} {d['adduct']}" for d in ds[:k]) + (f" (+{len(ds) - k})" if len(ds) > k else "")


def _window_readings(ws) -> str:
    return "; ".join(f"{d['neutral']} {d['adduct']} ({str(d['why']).replace(WINDOW_WHY + ': ', '')})" for d in ws)


def _is_committed(d, n, a) -> bool:
    return d["adduct"] == a and SP.same_formula(d["neutral"], n)


def _isobar_molecule(pol, a, d) -> str:
    """R when reading d is X + R on an adduct without R (a = d's adduct + R); else ''."""
    da, dd = SP.adduct_delta(a), SP.adduct_delta(d["adduct"])
    if da is None or dd is None:
        return ""
    diff = {k: da.get(k, 0) - dd.get(k, 0) for k in set(da) | set(dd)}
    diff = {k: v for k, v in diff.items() if v}
    return next((name for name, mol in ISOBAR_MOLECULES.get(pol, ()) if diff == mol), "")


def split_rules(S, row: dict, grid) -> dict:
    """The split of one pair: dict(pinned, rule, text, label_pin, track). ``text`` is the split text the
    evidence and the would-lift print (without the locked-channel note a pinned split may carry)."""
    n, a = str(row["neutral_formula"]), str(row["adduct"])

    def out(rule, text, plaus=(), label_pin=False, track=False, kept=False):
        kind = dict(SPLIT_RULES)[rule]
        pinned = kind == "PINNED" or (kind == "gate" and kept)
        return dict(pinned=pinned, rule=rule, text=text, label_pin=label_pin, track=track, n_plausible=len(plaus))

    counts = SP.ion_counts_of(n, a, row.get("ion"))
    if not counts:
        return out("no ion composition", "no ion composition")
    decs = S.ctx.decompositions(counts, n, a, adducts=grid)
    plaus = [d for d in decs if d["ok"] and d["neutral"]]
    if on(row.get("ion_only")) or a in SP.ION_ONLY:
        return out("ion-only channel", "ion-only channel (process open)", plaus)
    labelled_ion = bool(S.ctx.labelled) and counts.get("^N", 0) > 0
    if labelled_ion and "^" in a and not on(row.get("label_veto")) and (n, a) not in S.alien:
        return out("label", "label (the ion carries the reagent's ^N)", plaus, label_pin=True)
    if labelled_ion and "^" not in a:
        lab = [d for d in plaus if "^" in d["adduct"]]
        return out("label on an unlabelled adduct",
                   ("label points to the cluster reading: " + _readings(lab, 3)) if lab
                   else "the ion carries ^N but no plausible cluster reading", plaus)
    # ---- the amine gate (a positive run whose grid holds [M+NH4]+) ----
    gate = S.gate if (S.gate is not None and NH4 in grid) else None
    note, pre, adm_note, amine = "", "", "", False
    if gate is not None:
        if a == NH4:
            # a committed NH4 reading: the gate alone decides it (the admissibility rule does not apply)
            g = gate(n)
            if g["kept"]:
                text = (f"NH4 adduct tracks its parent ({g['how']})" if g["track"]
                        else f"NH4 adduct kept by the amine gate: {g['how']}")
                return out("amine gate: committed NH4", text, plaus, track=bool(g["track"]), kept=True)
            return out("amine gate: committed NH4",
                       f"amine default (NH4 reading unconfirmed: {g['how']}; the engine's gate reads this ion as "
                       f"{g.get('amine', '?')} [M+H]+)", plaus)
        amine = a == PROTONATED          # the NH4 admissibility rule asks only about an [M+H]+ (amine) reading
        nh4 = [d for d in plaus if d["adduct"] == NH4]
        bad = [d for d in nh4 if amine and not gate.nh4_admissibility(d["neutral"])["ok"]]
        live = [d for d in nh4 if not any(d is x for x in bad)]
        bad_txt = "; ".join(f"{d['neutral']} [M+NH4]+ -- {gate.nh4_admissibility(d['neutral'])['why']}" for d in bad)
        plaus = [d for d in plaus if not any(d is x for x in bad)]
        verdict = [(d, gate(d["neutral"])) for d in live]
        kept = [(d, g) for d, g in verdict if g["kept"]]
        dropped = [(d, g) for d, g in verdict if not g["kept"]]
        if amine:
            other_reading = [(d["neutral"], gate.nh4_admissibility(d["neutral"])) for d in live]
            other_reading = [(y, ga) for y, ga in other_reading if ga["present"] and not ga["own_name"]]
            if other_reading:
                adm_note = (" [NH4 admissibility rule: Y's uronium adduct ion is present under another reading: "
                            + "; ".join(f"{y} -- {ga['why'].replace('uronium adduct ion present: ', '')}"
                                        for y, ga in other_reading) + "]")
        plaus = [d for d in plaus if not (d["adduct"] == NH4 and any(d is x for x, _ in dropped))]
        others = [d for d in plaus if not _is_committed(d, n, a) and d["adduct"] != NH4]
        also = f"; also {len(others)} other decomposition(s): {_readings(others, 3)}" if others else ""
        if amine and verdict:
            # an [M+H]+ with an admissible NH4 reading stays open
            if kept:
                return out("amine gate: admissible NH4 reading", "NH4 reading kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ ({g['how']})" for d, g in kept)
                    + " -- the committed [M+H]+ (amine) reading is contested" + also + adm_note, plaus)
            return out("amine gate: admissible NH4 reading", "amine default (NH4 reading unconfirmed: " + "; ".join(
                f"{d['neutral']} [M+NH4]+ -- {g['how']}" for d, g in dropped) + ")" + also + adm_note, plaus)
        if not amine and (verdict or bad):
            parts = []
            if dropped:
                parts.append("NH4 reading not kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ -- {g['how']}" for d, g in dropped))
            if kept:
                parts.append("NH4 reading kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ ({g['how']})" for d, g in kept))
            note = "; ".join(parts)
        if amine and bad_txt:
            pre = f"NH4 reading inadmissible: no uronium adduct of Y ({bad_txt}); "
    sfx = (f" [{note}]" if note else "") + adm_note
    # ---- the context window: a reading only the window excludes; the X+reagent isobar stays live ----
    window = []
    for d in decs:
        if d["ok"] or not d["neutral"] or not str(d["why"]).startswith(WINDOW_WHY) or _is_committed(d, n, a):
            continue
        if gate is not None and d["adduct"] == NH4 and (
                (amine and not gate.nh4_admissibility(d["neutral"])["ok"]) or not gate(d["neutral"])["kept"]):
            continue          # the gate or the admissibility rule removes it, not the window
        window.append(d)
    isobar = [(d, _isobar_molecule(S.pol, a, d)) for d in window]
    isobar = [(d, m) for d, m in isobar if m]
    if isobar:
        plaus = plaus + [d for d, _m in isobar]
        window = [d for d in window if not any(d is x for x, _ in isobar)]
        sfx += " [X+reagent isobar admitted over the context window: " + "; ".join(
            f"{d['neutral']} {d['adduct']} (X+{m}; window {str(d['why']).replace(WINDOW_WHY + ': ', '')})"
            for d, m in isobar) + "]"
    self_ok = any(_is_committed(d, n, a) for d in plaus)
    if len(plaus) == 1 and self_ok:
        wtxt = f" [pinned only by the context window: {_window_readings(window)}]" if window else ""
        return out("only decomposition", pre + "only decomposition" + sfx + wtxt, plaus)
    matched = {x for x in txt(row.get("committed_matched")).split(",") if x}
    for el in ("Br", "Cl"):
        if counts.get(el, 0) and el in matched and counts[el] > SP.reagent_supply(grid, el):
            carriers = [d for d in plaus if d["counts"].get(el, 0) > 0]
            if len(carriers) == 1 and carriers[0]["adduct"] == a:
                wk = [d for d in window if d["counts"].get(el, 0) > 0]
                wtxt = f" [pinned only by the context window: {_window_readings(wk)}]" if wk else ""
                return out("halogen count", pre + "halogen count" + sfx + wtxt, plaus)
    if not self_ok:
        why = next((d["why"] for d in decs if d["adduct"] == a), "")
        return out("committed neutral not plausible", f"committed neutral not plausible ({why}); {len(plaus)} "
                   "plausible: " + _readings(plaus, 4) + sfx, plaus)
    return out("k decompositions", pre + f"{len(plaus)} decompositions: " + _readings(plaus) + sfx, plaus)


# ===========================================================================
# step 1 + the internal pass: competitors left after the series exclusion
# ===========================================================================
MAX_PASSES = 20                 # the series exclusion and its route anchors iterate to a fixed point
ROUTE_ALIASES = {"negative": (("formate", {"C": 1, "H": 2, "O": 2}, "deprotonation"),),
                 "positive": (("NH4+", {"N": 1, "H": 3}, "protonation"),)}


def _left(S, keys) -> dict:
    """{pair index: [competitor left by the isotope tests]} (route-excluded competitors count as left)."""
    pos = {k: i for i, k in enumerate(keys)}
    out = defaultdict(list)
    if S.comps is None or not len(S.comps):
        return out
    for r in S.comps.to_dict("records"):
        i = pos.get((str(r["neutral_formula"]), str(r["adduct"])))
        if i is None or str(r["status"]) != "left":
            continue
        out[i].append(dict(name=str(r["competitor"]), kind=str(r["kind"]), neutral=txt(r["comp_neutral"]),
                           adduct=str(r["comp_adduct"]) if txt(r["comp_adduct"]) else str(r["adduct"])))
    return out


def _route_anchored(S, rows, keys, f, partners) -> np.ndarray:
    """Pairs with two routes (both route ions of an own route-class pair in the same file in >= 2 files, 1 on a
    one-file source; no alias of a committed neutral explains both -- the aliases are locked side channels by
    default) or an other-source partner (split pinned, >= 2 files), whose committed neutral is plausible."""
    n = len(rows)
    out = np.zeros(n, bool)
    excl = {k for k, x in zip(keys, f["contradicted"]) if x} | set(S.alien)
    excl |= {k for k, r in zip(keys, rows) if on(r.get("iso_veto")) or on(r.get("label_veto"))}
    excl |= {keys[i] for i in range(n) if f["rej"][i] or f["reagent"][i]}
    cof = 1 if S.arm else RT.ROUTE_COFILES
    table, _byfile = RT.routes_table(S.pf, S.run_classes, excl, cof, S.skip_m0)
    aliases = [x for x in ROUTE_ALIASES.get(S.pol, ()) if not SPL.is_locked(x[0])]
    committed_ok = {keys[i][0] for i in range(n) if f["member"][i]}
    for i, (nn, aa) in enumerate(keys):
        own_cls = RT.route_class(aa)
        if own_cls is None or (nn, aa) in excl or f["ion_only"][i]:
            continue
        own = False
        t = table.get(nn)
        for (A, B), nco in (t["co"].items() if t is not None else ()):
            if own_cls not in (A, B) or nco < cof:
                continue
            explained = False
            for _name, L, base in aliases:
                if base not in (A, B):
                    continue
                Y = RT.fsub(RT.fcounts(nn), L)
                if Y and S.ctx.space.plausible_neutral(Y)[0] and C.format_formula(Y) in committed_ok:
                    explained = True
                    break
            own = own or not explained
        cross = bool(partners) and f["nfiles"][i] >= RT.CROSS_MIN_FILES and not S.arm and f["pinned"][i] and any(
            cl not in S.run_classes for cl in partners.get(nn, {}))
        out[i] = (own or cross) and on(rows[i].get("committed_plausible"))
    return out


def _inpass(i, f, left, routes, homo, lmin):
    """The internal pass's (level, why) of pair i: the scale's step 0 + step 1, then route / ladder / listed
    levels (never output: they anchor the series exclusion and make other-source partners)."""
    if f["reagent"][i]:
        return "reagent", "reagent identity (" + f["reagent"][i] + ")"
    if f["rej"][i]:
        return "5b", "rejected: " + "; ".join(f["rej"][i])
    if f["untestable"][i]:
        return "5b", "untestable"
    if left:
        return "5a", f"competitors left ({len(left)})"
    if f["ion_only"][i]:
        return "4b", "ion formula only"
    if routes[i]:
        return ("3a" if f["listed"][i] else "3b"), "routes"
    h = homo.get(i)
    if h and h["anchored"] and f["pinned"][i] and f["nfiles"][i] >= lmin:
        return ("3a" if f["listed"][i] else "3d"), "ladder"
    if f["pinned"][i] and f["listed"][i]:
        return "3c", "split pinned + listed"
    if f["pinned"][i] and f["posfact"][i]:
        return "4a", "split pinned + positive fact"
    return "4b", ("split pinned, no positive fact" if f["pinned"][i] else "split open")


#: what anchored a pair in the internal pass, as the tables print it (the raw in-pass level stays in memory):
#: (anchor_kind, anchor_why) per in-pass (level, why); any other in-pass level is no anchor -- kind 'none', why
#: the pass's own reason
ANCHOR_TEXT = {("3b", "routes"): ("two routes", "two routes"), ("3a", "routes"): ("two routes", "two routes + listed"),
               ("3d", "ladder"): ("ladder", "ladder"), ("3a", "ladder"): ("ladder", "ladder + listed"),
               ("3c", "split pinned + listed"): ("listed", "split pinned + listed")}


def anchor_of(level, why) -> tuple[str, str]:
    """(anchor_kind, anchor_why) of an in-pass (level, why)."""
    return ANCHOR_TEXT.get((level, why), ("none", why))


# ===========================================================================
# steps 3 + 4: the level, and what would lift it
# ===========================================================================
LIFT = {
    "reagent": "reagent ion / reagent cluster (not levelled)",
    "untestable": "nothing could be enumerated or tested",
    "ion-only": "ion-only channel: the neutral and the process stay open",
    "4b pinned": ("4a needs a positive fact (an own in-band isotope line of the neutral's elements, the 15N label, "
                  "or NH4 tracking)"),
    "4b pinned, no name": "; 3c needs a named context-list entry",
    "4a": "3c needs a NAMED context-list entry naming the neutral",
    "4a, class entry": " (the class-list match is a tag only)",
    "3c": "level 2 (MS2 / standards) is not automatic",
}
LEFT_SHOWN = 6                  # the competitors a 5a would-lift names


def level_of(f: dict) -> str:
    """The level of one pair from its facts: the first outcome that applies."""
    if f["reagent"]:
        return "reagent"
    if f["rejected"] or f["untestable"]:
        return "5b"
    if f["left"]:
        return "5a"
    if f["ion_only"]:
        return "4b"
    if f["pinned"] and f["named"]:
        return "3c"
    if f["pinned"] and (f["posfact"] or f["track"]):
        return "4a"
    return "4b"


def would_lift(level: str, f: dict) -> str:
    if level == "reagent":
        return LIFT["reagent"]
    if level == "5b":
        # the rejection text with its 'rejected: ' read as 'refuted: ' (every occurrence, as built)
        return ("rejected: " + "; ".join(f["rejected"])).replace("rejected: ", "refuted: ") if f["rejected"] \
            else LIFT["untestable"]
    if level == "5a":
        names = [c["name"] for c in f["left"]]
        return "competitors left: " + "; ".join(names[:LEFT_SHOWN]) + (
            f" (+{len(names) - LEFT_SHOWN} more)" if len(names) > LEFT_SHOWN else "")
    if level == "4b":
        if f["ion_only"]:
            return LIFT["ion-only"]
        if not f["pinned"]:
            return "split not pinned: " + f["split_text"]
        return LIFT["4b pinned"] + ("" if f["named"] else LIFT["4b pinned, no name"])
    if level == "4a":
        return LIFT["4a"] + (LIFT["4a, class entry"] if f["class_entry"] else "")
    return LIFT["3c"]


# ===========================================================================
# a prepared source -> one row per pair
# ===========================================================================
DECISION_COLUMNS = ["neutral_formula", "adduct", "evidence_level", "claim", "would_lift", "competitors_left",
                    "n_left", "reagent_identity", "rejected_by", "o11_tag", "untestable", "ion_only", "split_pinned",
                    "split_rule", "split_text", "positive_fact", "named_entry", "class_entry", "anchor_kind",
                    "anchor_why", "n_series_excl", "iterations", "n_files_obs", "mz", "height",
                    *EV.INTERNAL_COLUMNS]


def decide(S, partners=None) -> pd.DataFrame:
    """The scale's decision over a prepared source (`peaky.assignment.levels.source.prepared`): one row per pair
    with the level, its claim, what would lift it and the facts that decided it. ``partners`` = {neutral: {route
    class: [text]}} (other-source partners: they anchor the series exclusion of step 1; never a level)."""
    P = S.P.reset_index(drop=True)
    rows = P.to_dict("records")
    n = len(rows)
    keys = [(str(r["neutral_formula"]), str(r["adduct"])) for r in rows]
    nfiles = pd.to_numeric(P["n_files_obs"], errors="coerce").fillna(0).astype(int).to_numpy() if n else []
    grid = grid_of(S) if n else []
    f = defaultdict(list)
    for i, r in enumerate(rows):
        nn, aa = keys[i]
        rej, o11 = rejections(r, S.below.get(keys[i], {}))
        sp = split_rules(S, r, grid)
        elements = set(counts_of(nn))
        matched = {x for x in txt(r.get("committed_matched")).split(",") if x}
        hits = S.lists.hits(nn)
        f["reagent"].append(reagent_identity(nn, aa, S.pol))
        f["rej"].append(rej)
        f["o11"].append(o11)
        f["untestable"].append(on(r.get("no_comp_info")))
        f["ion_only"].append(on(r.get("ion_only_reading")))
        f["contradicted"].append(on(r.get("committed_contradicted")))
        f["split"].append(sp)
        f["pinned"].append(sp["pinned"])
        f["posfact"].append(sp["label_pin"] or bool(matched & elements))
        f["named"].append([h for h in hits if h["named"]])
        f["class_entry"].append([h for h in hits if not h["named"]])
        f["listed"].append(bool(hits))
        f["nfiles"].append(int(nfiles[i]))
    f["member"] = [not f["rej"][i] and not f["untestable"][i] and not f["reagent"][i] for i in range(n)]
    if S.arm:
        partners = None
    left_of = _left(S, keys)
    routes = _route_anchored(S, rows, keys, f, partners) if n else np.zeros(0, bool)
    member = np.array(f["member"], dtype=bool)
    member_ions = {k for i in np.flatnonzero(member)
                   for k in [RT.ion_key(keys[i][0], keys[i][1], rows[i].get("ion"))] if k}
    D = P[["neutral_formula", "adduct", "mz"]].copy()
    lmin = 1 if S.arm else RT.LADDER_MIN_FILES

    def one_pass(excluded, homo):
        lefts = [[c for c in left_of.get(i, []) if c["name"] not in excluded.get(i, {})] for i in range(n)]
        return [_inpass(i, f, lefts[i], routes, homo, lmin) for i in range(n)], lefts

    excluded, homo = {}, {}
    inp, lefts = one_pass(excluded, homo)
    iterations = 0
    for iterations in range(1, MAX_PASSES + 1):
        anchor = np.array([lv in ("3a", "3b") and why == "routes" for lv, why in inp], dtype=bool)
        D["_lv"] = [lv for lv, _w in inp]
        homo = RT.homologue_facts(D, member, anchor, tol_ppm=S.tol_ppm, units=RT.LADDER_UNITS)
        new = RT.series_exclusions(D, {i: v for i, v in left_of.items() if member[i]}, homo, member_ions)
        inp2, lefts2 = one_pass(new, homo)
        done = new == excluded and [lv for lv, _w in inp2] == [lv for lv, _w in inp]
        excluded, inp, lefts = new, inp2, lefts2
        if done:
            break
    recs = []
    for i in range(n):
        sp = f["split"][i]
        g = dict(reagent=f["reagent"][i], rejected=f["rej"][i], untestable=f["untestable"][i], left=lefts[i],
                 ion_only=f["ion_only"][i], pinned=sp["pinned"], named=f["named"][i], posfact=f["posfact"][i],
                 track=sp["track"], split_text=sp["text"], class_entry=f["class_entry"][i])
        lv = level_of(g)
        pf = []
        if sp["label_pin"]:
            pf.append("15N label")
        els = {x for x in txt(rows[i].get("committed_matched")).split(",") if x} & set(counts_of(keys[i][0]))
        if els:
            pf.append(f"own in-band isotope line(s) {txt(rows[i].get('committed_matched_lines'))} "
                      f"({','.join(sorted(els))} of the neutral)")
        if sp["track"]:
            pf.append("NH4 adduct tracks its parent")
        recs.append(dict(
            neutral_formula=keys[i][0], adduct=keys[i][1], evidence_level=lv, claim=EV.claim_class(lv),
            would_lift=would_lift(lv, g), competitors_left="; ".join(c["name"] for c in lefts[i]),
            n_left=len(lefts[i]), reagent_identity=f["reagent"][i], rejected_by="; ".join(f["rej"][i]),
            o11_tag=f["o11"][i], untestable=f["untestable"][i], ion_only=f["ion_only"][i],
            split_pinned=sp["pinned"], split_rule=sp["rule"], split_text=sp["text"], positive_fact="; ".join(pf),
            named_entry="; ".join(f"{h['id']} = {h['name']}" for h in f["named"][i]),
            class_entry="; ".join(h["id"] for h in f["class_entry"][i]),
            anchor_kind=anchor_of(*inp[i])[0], anchor_why=anchor_of(*inp[i])[1],
            inpass_level=inp[i][0], inpass_why=inp[i][1], n_series_excl=len(excluded.get(i, {})),
            iterations=iterations, n_files_obs=f["nfiles"][i], mz=rows[i].get("mz"), height=rows[i].get("height")))
    return pd.DataFrame(recs, columns=DECISION_COLUMNS)


def _blank_rows(src) -> pd.DataFrame:
    keys = set()
    for led in src.per_file.values():
        m0 = led[led["role"].astype(str) == "M0"]
        keys |= {(nn, aa) for nn, aa in zip(m0["neutral_formula"].fillna("").astype(str),
                                            m0["adduct"].fillna("").astype(str)) if nn and aa}
    out = pd.DataFrame([dict(neutral_formula=nn, adduct=aa) for nn, aa in sorted(keys)],
                       columns=["neutral_formula", "adduct"])
    return out.reindex(columns=DECISION_COLUMNS)


def na_rows(src) -> pd.DataFrame:
    """Every committed (neutral, adduct) of the source's per-file M0 rows, NA: the class gate comes before any
    fact work."""
    out = _blank_rows(src)
    out["evidence_level"] = "NA"
    out["claim"] = EV.CLAIM_NA
    out["would_lift"] = ""
    return out


def no_window_rows(src, why: str = EV.NO_WINDOW_TEXT) -> pd.DataFrame:
    """A lone file with no calibrated window (no calibration of its own, no run sigma to borrow), or a pooled
    source none of whose files is calibrated (``why`` = EV.NO_RUN_WINDOW_TEXT): no level, claim tentative, and why
    (the engine's per-file and pooled stages say the same in `evidence`)."""
    out = _blank_rows(src)
    out["evidence_level"] = ""
    out["claim"] = "tentative"
    out["would_lift"] = why
    return out


def decide_source(src, partners=None) -> pd.DataFrame:
    """Level a Source (`evidence.source_from_run_dir` / `source_from_frames`) with this script's decision: NA
    unless the source's width model is Orbitrap-class; partners never on a one-file adapted source."""
    klass, _r200 = src.instrument()
    if klass != "orbitrap":
        return na_rows(src)
    if not SRC.committed_pairs(src.per_file):
        return _blank_rows(src)          # nothing committed: no pair to level (no row, the columns)
    if SRC.no_run_window(src):
        return no_window_rows(src, EV.NO_RUN_WINDOW_TEXT)   # no file calibrated: no window to enumerate in
    return decide(SRC.prepared(src), None if src.one_file_minima else partners)


# ===========================================================================
# sources from disk, partners, the vector
# ===========================================================================
def _is_run_dir(path) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "batch_summary.json")) and \
        os.path.isdir(os.path.join(path, "per_file"))


def run_dir_of(path) -> str | None:
    """The run dir itself, or the out dir holding exactly one; None for anything else."""
    path = os.path.expanduser(str(path).rstrip("/"))
    if _is_run_dir(path):
        return path
    if os.path.isdir(path):
        inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if _is_run_dir(d)]
        if len(inner) == 1:
            return inner[0]
    return None


def resolution_for(r200: float) -> dict:
    """An Orbitrap-shaped width model (FWHM ~ m^1.5) resolving ``r200`` at m/z 200, as a batch summary records it."""
    from peaky.chem.resolution import Resolution
    return Resolution(coef=(200.0 / float(r200)) / 200.0 ** 1.5, exponent=1.5, offset=0.0, source="declared").as_dict()


def lone_run_inputs(sid, *, reagent, context=None, resolution=None, **kw):
    """The run inputs of a lone ledger CSV, built as the per-file stage of a single-sample run builds its own
    (assign._stage_evidence after cli's assign): the profile's context unless ``context`` names one, the reference
    lists that context and the profile label activate (the always-active lists included) with their activation
    record, and the halogen of the profile's declared channels. What a lone CSV cannot know -- a batch's or
    dataset's name (keyword-activated lists), the file's height gate -- falls back as the stage's own fallbacks do."""
    from peaky.assignment import reflists as RL
    from peaky.chem import profiles as PR
    try:
        prof = PR.resolve(str(reagent))
    except (KeyError, ValueError) as e:
        raise SystemExit(f"--reagent {reagent!r}: {e}") from None
    ctx_name = context or prof.context
    lists, _tags, record = RL.activate(ctx_name, prof.label or "", record=True, fields=("context", "reagent label"))
    return EV.file_run_inputs(sample_id=sid, reagent=reagent, context=ctx_name, resolution=resolution,
                              reflists_active=RL.active_versions(lists), activation=record,
                              reagent_halogen=EV.channel_halogen(prof.adducts), **kw)


def source_of(path, *, mode=None, main=None, resolving_power=None, reagent=None, context=None, window=None):
    """(label, Source) of one argument: a run dir (or its out dir), or one ledger CSV (alone, or in ``main``'s
    context)."""
    rd = run_dir_of(path)
    if rd is not None:
        return os.path.basename(rd), EV.source_from_run_dir(rd, mode=mode or "run")
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: neither a run dir (batch_summary.json + per_file/) nor a ledger CSV")
    if (mode or "adapted") == "run":
        raise SystemExit(f"{path}: a ledger CSV is one file: --mode adapted or strict")
    if main is not None:
        return os.path.basename(path), EV.source_from_run_dir(path, main=main, mode=mode or "adapted")
    src = EV.source_from_run_dir(path, mode=mode or "adapted")
    if resolving_power:
        if not reagent:
            raise SystemExit(f"{path}: --resolving-power on a lone ledger needs --reagent (its profile)")
        sid = next(iter(src.per_file))
        kw = {} if window is None else {"degeneracy_cal": tuple(window)}
        ri = lone_run_inputs(sid, reagent=reagent, context=context, resolution=resolution_for(resolving_power), **kw)
        src = EV.source_from_frames(src.per_file, run_inputs=ri, mode=mode or "adapted", name=sid, label=sid)
        # the step-1 window: the file's calibration when given, else refitted from its degeneracy counts
        from peaky.assignment.levels import context as CX
        src.win = CX.run_windows(src.summary, src.per_file)
        if not np.isfinite(CX.window_table(src.win)[1][1]):
            src.no_window = True
    return os.path.basename(path), src


def partners_of(paths, log=print) -> dict:
    """Other-source partners from --corroborate run dirs: each Orbitrap-class run dir levelled once with no
    partners (this script's decision); a TOF-class / class-less source, a CSV or an Orbitrap run dir none of whose
    files is calibrated (no run window) gives none (logged, with the batch's reason)."""
    merged = {}
    for p in paths:
        rd = run_dir_of(p)
        if rd is None:
            log(f"  --corroborate {p}: not a run dir -- no partners")
            continue
        src = EV.source_from_run_dir(rd)
        klass, r200 = src.instrument()
        if klass != "orbitrap":
            log(f"  --corroborate {os.path.basename(rd)}: {klass or 'class-less'} source -- no partners")
            continue
        if SRC.no_run_window(src):
            log(f"  --corroborate {os.path.basename(rd)}: {EV.NO_RUN_WINDOW_PARTNERS}")
            continue
        got = RT.partners_from(decide_source(src), os.path.basename(rd))
        log(f"  --corroborate {os.path.basename(rd)}: {len(got)} partner neutral(s)")
        merged = RT.merge_partners(merged, got)
    return merged


def run(paths, corroborate=(), *, mode=None, main=None, resolving_power=None, reagent=None, context=None,
        window=None, log=print) -> pd.DataFrame:
    """Level every path with this script's decision; one row per (source, pair). The --corroborate partners are
    computed only when a source can take them (Orbitrap-class, not a one-file adapted source)."""
    main_src = EV.source_from_run_dir(run_dir_of(main) or main) if main else None
    sources = [source_of(p, mode=mode, main=main_src, resolving_power=resolving_power, reagent=reagent,
                         context=context, window=window) for p in paths]
    takes = any(src.instrument()[0] == "orbitrap" and not src.one_file_minima for _l, src in sources)
    partners = partners_of(corroborate, log=log) if (corroborate and takes) else None
    out = []
    for label, src in sources:
        df = no_window_rows(src) if getattr(src, "no_window", False) else decide_source(src, partners)
        df.insert(0, "source", label)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["source"] + DECISION_COLUMNS)


def vector(levels: pd.DataFrame) -> dict:
    """{level: n pairs} over 3c 4a 4b 5a 5b reagent NA (zeros kept)."""
    c = levels["evidence_level"].astype(str).value_counts() if len(levels) else pd.Series(dtype=int)
    return {k: int(c.get(k, 0)) for k in VECTOR_KEYS}


def height_share(levels: pd.DataFrame, per_file: dict, merged: pd.DataFrame | None) -> dict:
    """% of the committed per-file M0 height per level: every per-file M0 reading takes its pair's level; a
    reading no merged row carries is 'unmatched' (no merged ledger: none is); the denominator holds them all."""
    lv = dict(zip(zip(levels["neutral_formula"].astype(str), levels["adduct"].astype(str)),
                  levels["evidence_level"].astype(str)))
    mkeys = None if merged is None else set(zip(merged["neutral_formula"].fillna("").astype(str),
                                                merged["adduct"].fillna("").astype(str)))
    acc = defaultdict(float)
    for led in per_file.values():
        m0 = led[led["role"].astype(str) == "M0"]
        h = pd.to_numeric(m0["height"], errors="coerce").fillna(0.0).to_numpy()
        for hh, k in zip(h, zip(m0["neutral_formula"].fillna("").astype(str), m0["adduct"].fillna("").astype(str))):
            acc["unmatched" if (mkeys is not None and k not in mkeys) else lv.get(k, "(no pair)")] += float(hh)
    tot = sum(acc.values())
    keys = VECTOR_KEYS + ["unmatched", "(no pair)"]
    return {k: (100.0 * acc.get(k, 0.0) / tot if tot else float("nan")) for k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The evidence scale of peaky " + EV.SCALE_RELEASE
                                 + ", restated: level run dirs or ledgers post hoc.")
    ap.add_argument("paths", nargs="+", help="run dirs (or the out dir holding one) or ledger CSVs")
    ap.add_argument("--corroborate", action="append", default=[],
                    help="an Orbitrap-class run dir whose route / ladder / listed pairs become other-source partners "
                         "(a tag; they can anchor the series exclusion); repeatable")
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="run (a run dir, every minimum 3; the default), adapted (one file, minima 1; the default for "
                         "a CSV) or strict (one file, the batch minima)")
    ap.add_argument("--main", help="a run dir: level each ledger CSV in its context, like a decoy arm")
    ap.add_argument("--resolving-power", type=float, help="R at m/z 200 for a lone ledger CSV (else it reads NA)")
    ap.add_argument("--reagent", help="the reagent profile of a lone ledger CSV levelled with --resolving-power")
    ap.add_argument("--context", help="the context of a lone ledger CSV (default: the reagent profile's context); "
                                      "it and the profile label activate the reference lists, as `peaky assign` does")
    ap.add_argument("--window", help="MU,SIGMA (ppm): a lone ledger's calibration (its batch_summary per_file "
                                     "degeneracy_cal); default: refitted from its degeneracy counts")
    ap.add_argument("--out", help="write the levelled pairs here as CSV")
    ap.add_argument("--vector", action="store_true",
                    help="print n 3c/4a/4b/5a/5b/reagent/NA and the %% of committed M0 height per level")
    args = ap.parse_args(argv)
    log = (lambda *a: print(*a, file=sys.stderr))
    window = tuple(float(x) for x in args.window.split(",")) if args.window else None
    df = run(args.paths, args.corroborate, mode=args.mode, main=args.main, resolving_power=args.resolving_power,
             reagent=args.reagent, context=args.context, window=window, log=log)
    if df.empty:
        print("nothing to level", file=sys.stderr)
        return 1
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        EV.for_output(df).to_csv(args.out, index=False)      # the raw in-pass tokens stay in memory
    order = "/".join(VECTOR_KEYS)
    for (label, group), path in zip(df.groupby("source", sort=False), args.paths):
        v = vector(group)
        print(f"{label}  n={len(group)}")
        print(f"  {order}")
        print("  " + "/".join(str(v[k]) for k in VECTOR_KEYS))
        if args.vector:
            rd = run_dir_of(path)
            if rd is not None:
                per_file = {os.path.basename(p)[: -len("_ledger.csv")]: pd.read_csv(p, low_memory=False)
                            for p in sorted(glob.glob(os.path.join(rd, "per_file", "*_ledger.csv")))}
                mp = os.path.join(rd, "merged_ledger.csv")
                merged = pd.read_csv(mp, low_memory=False) if os.path.isfile(mp) else None
            else:
                per_file, merged = {label: pd.read_csv(os.path.expanduser(path), low_memory=False)}, None
            share = height_share(group, per_file, merged)
            print("  % of committed M0 height: " + "; ".join(
                f"{k} {share[k]:.2f}" for k in VECTOR_KEYS + ["unmatched", "(no pair)"] if np.isfinite(share[k])))
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


# ===========================================================================
# the pre-0.10.0 B-series reference (frozen): the merge vote's private class and its tests
# ===========================================================================
# The pre-0.10.0 B-series predicates, frozen (the merge vote's private evidence class reads the same
# facts in core: `peaky.assignment.evidence.vote_classes`). Renamed where the scale above took the name:
# `series_run` (was `run`), `series_main` (was `main`), `series_level_of` (was `level_of`),
# `SERIES_LEVEL_ORDER` (was `LEVEL_ORDER`). No user-facing level reads anything below.
#

SERIES_LEVEL_ORDER = ["1", "2a", "2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]

# Natural abundance of the heavy isotope relative to the light one, for the
# satellite the audit looks for. 13C is per carbon and computed from the ion.
ISOTOPE_ABUNDANCE = {
    "34S": 0.0443,
    "37Cl": 0.3196,
    "81Br": 0.9728,
    "29Si": 0.0508,
    "30Si": 0.0335,
}
C13_PER_CARBON = 0.0107
# 15N and 18O are expected per atom of the ion (heavy / light natural abundance;
# a caret '^N' atom is already 15N and is not counted)
PER_ATOM_ABUNDANCE = {"15N": ("N", 0.00368 / 0.99632), "18O": ("O", 0.00205 / 0.99757)}
RATIO_LO, RATIO_HI = 0.5, 2.0
# the element an isotope child's tag measures ('13C2' -> C, '2x81Br' -> Br,
# '81Br(pair)' -> Br, '14N' -> N); 'M' (a generic M+n child) names none
TAG_ELEMENT = re.compile(r"^(?:\d+x)?\d+([A-Z][a-z]?)\d*(?:\(pair\))?$")
# an ion carrying Br or Cl owns its M+2 region: its 81Br / 37Cl line buries an
# 18O satellite, so an '18O' line there is not measured
M2_OWNERS = ("Br", "Cl")
FORMULA_TOKEN = re.compile(r"(\^?)([A-Z][a-z]?)(\d*)")

# The scope of each pass-0 family (docs/EVIDENCE_LEVELS.md section 4.1): a family
# whose entries are hand-listed compounds asserts a COMPOUND, one generated from a
# formula loop asserts a CLASS. 2b needs compound scope AND a one-structure formula
# in peaky/data/isomer_space.csv; any other curated commit is 3a. A family not
# listed is read as a class. (Until 2026-09-22 this script kept two hand-made sets
# -- {atmospheric, reactive_iodine} always 2b, six others always 3a -- which
# agreed with the spec on every golden row but not on the positive-mode families:
# cyclosiloxane D3/D5/D7 read 3a here and 2b in core, and contaminant:silanediol
# was in neither set. The spec is the design; the sets were the bug.)
KNOWN_FAMILY_SCOPE = {
    "atmospheric": "compound",
    "reactive_iodine": "compound",
    "ambient_inorganic": "compound",
    "nitroaromatic": "compound",
    "cyclosiloxane": "compound",
    "indoor_sulfur": "compound",
    "organophosphate": "compound",
    "organothiophosphate": "compound",
    "easyic_hydride": "compound",
    "perfluoroacid": "class",
    "chlorinated_paraffin": "class",
    "contaminant:silanediol": "class",
}
ISOMER_SPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "peaky", "data", "isomer_space.csv")
_STRUCTURES: dict | None = None


def plausible_structures(formula: str) -> int | None:
    """The isomer space's structure count for `formula`, None when absent."""
    global _STRUCTURES
    if _STRUCTURES is None:
        _STRUCTURES = {}
        if os.path.isfile(ISOMER_SPACE):
            space = pd.read_csv(ISOMER_SPACE)
            for f, n in zip(space["formula"].astype(str).str.strip(),
                            pd.to_numeric(space["n_plausible_structures"], errors="coerce")):
                if pd.notna(n):
                    _STRUCTURES[f] = int(n)
    return _STRUCTURES.get(str(formula).strip())

BARE_ADDUCTS = {"[M-H]-"}
CLUSTER_ADDUCTS = {
    "[M+NO3]-",
    "[M+15NO3]-",
    "[M+^NO3]-",
    "[M+Br]-",
    "[M+HBr+Br]-",
    "[M+CO3]-",
}
RESOLVED = {"resolved", "isolated"}
LOW_CONFIDENCE = {"Low", "Suspect"}
# ION-ONLY rows (the engine's `ion_only` stage): the electron-attachment line
# beside a committed [M-H]- acid, on "[M]-." with method `ion_only:*` (a merged
# ledger carries the `ion_only_of` link instead). Levelled on their own
# satellite alone (4d with one, 5a without); never a second channel or a
# corroboration for anything, in either direction -- as evidence.py does.
ION_ONLY_ADDUCTS = {"[M]-."}
ION_ONLY_METHOD_PREFIX = "ion_only:"

# The heavy satellite a reagent halogen contributes. Iodine is monoisotopic, so
# an iodide reagent can never produce a reagent-only isotope pattern.
HALOGEN_SATELLITE = {"Br": "81Br", "Cl": "37Cl", "I": None}


def truthy(value) -> bool:
    """Null-safe truthiness: NaN, NA, None and 'false'/'' are all False."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def first_word(value) -> str:
    """The grade a `confidence` cell opens with ('High (0.91)' -> 'High')."""
    if not isinstance(value, str):
        return ""
    parts = value.split()
    return parts[0] if parts else ""


def as_list(value) -> list:
    """A ledger cell holding a JSON or repr'd list, or nothing at all.

    The ledger writes the satellite list with json.dumps, so a line without a
    per-line score carries `null`, which `ast.literal_eval` cannot read: JSON
    is tried first, as the engine's `evidence.as_list` does (C11+c; before it
    such a list read EMPTY here). A cell neither parser reads is empty."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def to_float(value) -> float:
    """A cell as a float; NaN when it is none."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def count_element(formula, element: str) -> int:
    """How many of `element` a formula carries; 0 when it carries none."""
    if not isinstance(formula, str):
        return 0
    match = re.search(element + r"(\d*)(?![a-z])", formula)
    if not match:
        return 0
    return int(match.group(1)) if match.group(1) else 1


def composition(formula, fold: bool = True, labelled: bool = False) -> dict:
    """Element counts of a formula; a caret isotope ('^N') folds into its
    element unless `fold` is False, when it is left out (it is already heavy),
    or `labelled` is True, when it is its own key ('^N')."""
    out: dict = {}
    if not isinstance(formula, str):
        return out
    for caret, element, n in FORMULA_TOKEN.findall(formula):
        if caret and labelled:
            element = caret + element
        elif caret and not fold:
            continue
        out[element] = out.get(element, 0) + (int(n) if n else 1)
    return out


ADDUCT_TOKEN = re.compile(r"([+-])\^?([A-Za-z0-9]+)")


def ion_composition(neutral, adduct, ion) -> dict:
    """The ION's element counts: its stored ion formula when that carries a
    charge sign, else neutral + adduct ('[M+HBr+Br]-' adds H and two Br; a
    ledger row can hold the NEUTRAL in ion_formula)."""
    s = str(ion).strip() if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return composition(s)
    a = str(adduct).strip() if isinstance(adduct, str) else ""
    if not neutral or not a.startswith("[M"):
        return composition(s)
    counts = composition(neutral)
    inner = a.split("]")[0][2:].replace("(", "").replace(")", "")
    for sign, token in ADDUCT_TOKEN.findall(inner):
        for element, n in composition(token).items():
            counts[element] = counts.get(element, 0) + (n if sign == "+" else -n)
    return {k: v for k, v in counts.items() if v} or composition(s)


ADDUCT_TOKEN_LABELLED = re.compile(r"([+-])(\^?[A-Za-z0-9]+)")


def ion_counts(neutral, adduct, ion) -> dict:
    """The ION's element counts as the engine's evidence.ion_composition reads
    them for the isotope lines, the labelled '^N' kept as its own key: the
    stored ion formula when it carries a charge sign, else neutral + adduct (a
    labelled reagent atom kept too: '[M+^NO3]-' adds 15N, as the engine's
    tiers._ion_counts(labelled=True) reads it), else the stored string.
    `ion_composition` above folds '^N' into N (the reagent halogen count it
    serves does not care)."""
    s = str(ion).strip() if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return composition(s, labelled=True)
    a = str(adduct).strip() if isinstance(adduct, str) else ""
    n = "" if neutral is None else str(neutral)
    counts: dict = {}
    if n and a.startswith("[M"):
        counts = composition(n, labelled=True)
        inner = a.split("]")[0][2:].replace("(", "").replace(")", "")
        for sign, token in ADDUCT_TOKEN_LABELLED.findall(inner):
            for element, k in composition(token, labelled=True).items():
                counts[element] = counts.get(element, 0) + (k if sign == "+" else -k)
    return {k: v for k, v in counts.items() if v} or composition(s, labelled=True)


def carries_reagent(neutral, adduct, ion, halogen) -> bool:
    """The ION carries more of the reagent halogen than the neutral: only then
    can a line of it be the reagent's (the reagent-only flag reads this alone
    since C11+c released the Br-free hold)."""
    if not halogen:
        return False
    return ion_composition(neutral, adduct, ion).get(halogen, 0) > composition(neutral).get(halogen, 0)


def expected_ratio(tag: str, ion_formula, committed: dict | None = None) -> float:
    """The natural height ratio of an isotope line to its parent (C11+c: the
    count-aware expectation relative to the COMMITTED parent line, `committed`
    its heavy configuration, {} / None = the mono line); 0.0 when there is none
    -- the engine's evidence.expected_ratio. The levels read every child through
    `resolve_child` (below); this is its parent-relative reading."""
    counts = {e: v for e, v in composition(ion_formula, labelled=True).items() if v}
    shift = resolve_child(tag, counts, {}, 0.0)["shift"]
    r = resolve_child(tag, counts, dict(committed or {}), shift if np.isfinite(shift) else 0.0)
    return float(r["expected"])


def neutral_elements(neutral, ion) -> set:
    """Elements whose isotope line speaks for the neutral: the neutral supplies
    more than half of the ion's atoms of the element (else the line measures
    the reagent -- 15N on a urea adduct of an N-free neutral, 81Br on a bromide
    adduct)."""
    own, whole = composition(neutral, labelled=True), composition(ion, labelled=True)
    return {e for e, n in own.items() if n > 0 and 2 * n > whole.get(e, 0)}


# ===========================================================================
# ISOTOPE CHILDREN JUDGED AGAINST THE COMMITTED LINE (C11+c) -- the standalone
# twin of peaky/chem/isotopes.py's section of the same name (this B-series section uses
# no peaky code). The committed line is read off the parent's m/z, each child
# label both ways (parent-relative / mono-counted), a line counts within
# max(1 ppm, 4 sigma(h)) of its exact spacing (sigma(h) self-fitted on the
# source's '13C' children; pcal and N1 rescue it, 'M+n' is exempt) and in band
# under its count-aware expectation. tests/test_isotope_children.py pins every
# function here to the engine's, text and values.
# ===========================================================================

#: exact mass (NIST / AME2020) of the heavy isotope a child label can name and
#: the element it replaces; the light isotope's mass is chemistry.M's
HEAVY_ISOTOPES: dict[str, tuple[str, float]] = {
    "13C": ("C", 13.0033548378), "15N": ("N", 15.0001088984), "17O": ("O", 16.9991317565),
    "18O": ("O", 17.9991596129), "33S": ("S", 32.9714589098), "34S": ("S", 33.967867004),
    "37Cl": ("Cl", 36.965902602), "81Br": ("Br", 80.9162906), "29Si": ("Si", 28.9764946649),
    "30Si": ("Si", 29.973770136), "2H": ("H", 2.0141017781),
}
#: monoisotopic element masses an ion's mono m/z is summed from (chemistry.M
#: plus the two alkali adduct metals -- a copy: this B-series section uses no peaky code)
ELEMENT_MASS: dict[str, float] = {
    "C": 12.0, "H": 1.0078250319, "O": 15.9949146221, "N": 14.0030740052, "S": 31.97207069, "P": 30.97376163,
    "Si": 27.976926535, "F": 18.9984031627, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.9044719,
    "^N": 15.0001088984, "Na": 22.989769282, "K": 38.9637064864,
}
ELECTRON_MASS = 0.0005485799
#: exact heavy - light spacing (Da); '14N' is the labelled reagent's light line,
#: one 15N - 14N spacing BELOW a '^N' atom
ISOTOPE_SPACING: dict[str, float] = {iso: m - ELEMENT_MASS[el] for iso, (el, m) in HEAVY_ISOTOPES.items()}
ISOTOPE_SPACING["14N"] = -ISOTOPE_SPACING["15N"]
#: the composition key each heavy isotope counts atoms of ('14N' counts the
#: labelled '^N' atoms)
ISOTOPE_ELEMENT: dict[str, str] = {iso: el for iso, (el, _m) in HEAVY_ISOTOPES.items()}
ISOTOPE_ELEMENT["14N"] = "^N"
#: per-atom heavy / light ratio -- the evidence band's (evidence.C13_PER_CARBON,
#: ISOTOPE_ABUNDANCE, PER_ATOM_ABUNDANCE); 33S, 17O, 2H natural; '14N' the
#: labelled reagent's impurity at LABEL_PURITY_15N. Card B7 re-rounds 30Si.
ISOTOPE_RATIO: dict[str, float] = {
    "13C": 0.0107, "81Br": 0.9728, "37Cl": 0.3196, "34S": 0.0443, "29Si": 0.0508, "30Si": 0.0335,
    "15N": 0.00368 / 0.99632, "18O": 0.00205 / 0.99757, "33S": 0.0075 / 0.9499, "17O": 0.00038 / 0.99757,
    "2H": 0.000115, "14N": 0.02 / 0.98,
}
#: an ion carrying Br or Cl owns its M+2 region: an '18O' line there is not
#: measured (evidence.M2_OWNERS, C17) -- its expectation reads 0
M2_OWNERS = ("Br", "Cl")

#: the heavy isotopes the committed line may carry (the M+2 drivers; S and Si
#: to two atoms), and the least probability a candidate configuration needs
COMMITTED_ISOTOPES = ("81Br", "37Cl", "34S", "30Si", "29Si")
COMMITTED_MIN_P = 1e-4
#: how far (ppm of the parent m/z) the parent may sit from a configuration's
#: exact position and still be read as committed on it: 5 ppm on an
#: Orbitrap-class width model, 20 ppm on a TOF-class one and without one
COMMITTED_TOL_PPM = {"orbitrap": 5.0, "tof": 20.0}
COMMITTED_TOL_CLASSLESS_PPM = 20.0
#: a generic 'M+n' child: the ion's lines within this half-width (Da) of its
#: measured shift, or half the width model's FWHM there if wider
GENERIC_HALF_WIDTH_DA = 0.012

#: the position test: a child counts within max(POSITION_MIN_PPM, POSITION_K x
#: sigma(h)) of its exact position, sigma(h)^2 = a^2 + b^2 / h fitted on the
#: source's own '13C' children -- pre-clipped at max(SIGMA_CLIP_PPM,
#: SIGMA_CLIP_K x the global 1.4826 x MAD), SIGMA_BINS height-quantile bins of
#: >= SIGMA_MIN_PER_BIN children, per-bin 1.4826 x MAD, weighted least squares
#: of s^2 on 1/h (weights = bin counts), a >= SIGMA_A_FLOOR_PPM, b >= 0; fewer
#: than SIGMA_MIN_CHILDREN such children and the source is not tested at all
POSITION_MIN_PPM = 1.0
POSITION_K = 4.0
SIGMA_MIN_CHILDREN = 40
SIGMA_BINS = 8
SIGMA_MIN_PER_BIN = 5
SIGMA_A_FLOOR_PPM = 0.02
SIGMA_CLIP_PPM = 5.0
SIGMA_CLIP_K = 6.0
MAD_TO_SIGMA = 1.4826
#: N1, the neighbour's pull: a child outside the window still counts when a row
#: the levelling reads (an M0 or iso child of its file, not the child's own
#: parent) >= NEIGHBOUR_RATIO x its height sits on the side it is displaced
#: toward, within NEIGHBOUR_REACH_PPM of its exact position, and the child's
#: residual is <= NEIGHBOUR_FRACTION of the distance to it
NEIGHBOUR_RATIO = 3.0
NEIGHBOUR_REACH_PPM = 25.0
NEIGHBOUR_FRACTION = 0.5

_PART_KX = re.compile(r"^(\d+)x(\d+)([A-Z][a-z]?)$")                     # '2x81Br'
_PART_TWO = re.compile(r"^(\d+)([A-Z][a-z]?)(\d+)([A-Z][a-z]?)\(pair\)$")  # '81Br37Cl(pair)': one of each
_PART_ONE = re.compile(r"^(\d+)([A-Z][a-z]?)(\d*)(\(pair\))?$")          # '81Br', '81Br2', '13C2', '37Cl(pair)'
_GENERIC = re.compile(r"^M\+(\d+)$")                                      # 'M+5'


def split_label(label) -> list[str]:
    """The '+' parts of a child label: 'M+n' is one part ('13C+M+4' -> ['13C',
    'M+4']); a bare number after an isotope part is the synthetic tests'
    nominal-shift note ('13C+1', '81Br+2') and is no part."""
    out: list[str] = []
    for tok in (t.strip() for t in str(label).split("+")):
        if tok.isdigit() and out:
            if out[-1] == "M":
                out[-1] = "M+" + tok
            continue
        out.append(tok)
    return [t for t in out if t]


def parse_label_part(part: str) -> tuple[str, object]:
    """One '+' part -> ('set', {isotope: count}) | ('alt', [{..}, {..}]) (either
    line, '81Br/37Cl(pair)') | ('gen', n) ('M+n') | ('mono', {}) ('M0') |
    ('bad', None). 'kx' multiplies ('2x81Br'), a trailing count counts
    ('81Br2', '37Cl3', '13C2'), '(pair)' is one atom ('81Br(pair)'),
    '81Br37Cl(pair)' is one of each."""
    p = str(part).strip()
    if p == "M0":
        return "mono", {}
    m = _GENERIC.match(p)
    if m:
        return "gen", int(m.group(1))
    if "/" in p and p.endswith("(pair)"):
        alts = []
        for a in p[:-6].split("/"):
            kind, v = parse_label_part(a)
            if kind != "set":
                return "bad", None
            alts.append(v)
        return "alt", alts
    m = _PART_KX.match(p)
    if m:
        iso = m.group(2) + m.group(3)
        return ("set", {iso: int(m.group(1))}) if iso in ISOTOPE_SPACING else ("bad", None)
    m = _PART_TWO.match(p)
    if m:
        a, b = m.group(1) + m.group(2), m.group(3) + m.group(4)
        return ("set", {a: 1, b: 1}) if a in ISOTOPE_SPACING and b in ISOTOPE_SPACING else ("bad", None)
    m = _PART_ONE.match(p)
    if m:
        iso = m.group(1) + m.group(2)
        return ("set", {iso: int(m.group(3) or 1)}) if iso in ISOTOPE_SPACING else ("bad", None)
    return "bad", None


def heavy_key(h: dict) -> tuple:
    """A heavy configuration {isotope: count} as a sorted tuple (zeros dropped)."""
    return tuple(sorted((k, int(v)) for k, v in h.items() if v))


def heavy_shift(h) -> float:
    """The exact mass shift of a heavy configuration from the all-light line."""
    return sum(ISOTOPE_SPACING[i] * k for i, k in (h.items() if isinstance(h, dict) else h))


def _heavy_nominal(h) -> int:
    return sum(int(round(ISOTOPE_SPACING[i])) * k for i, k in (h.items() if isinstance(h, dict) else h))


def _heavy_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def heavy_probability(h: dict, counts: dict) -> float:
    """P(configuration h) relative to the all-light line of an ion with element
    `counts`: per element the multinomial C(n; k1, k2, ..) x prod ratio^k over
    its n atoms. 0.0 when the ion cannot form it (an element it lacks, more
    heavy atoms than it has). '13C2' is C(nC, 2) x 0.0107^2."""
    p = 1.0
    per_el: dict[str, dict] = {}
    for iso, k in h.items():
        if k < 0:
            return 0.0
        per_el.setdefault(ISOTOPE_ELEMENT[iso], {})[iso] = k
    for el, ks in per_el.items():
        n = int(counts.get(el, 0))
        tot = sum(ks.values())
        if tot > n:
            return 0.0
        mult = math.factorial(n) / (math.factorial(n - tot) * math.prod(math.factorial(k) for k in ks.values()))
        p *= mult * math.prod(ISOTOPE_RATIO[iso] ** k for iso, k in ks.items())
    return p


def ion_sign(ion, adduct) -> str:
    """'+' / '-' of an ion: its stored ion formula's sign, else its adduct's;
    '' when neither carries one."""
    s = ion.strip().rstrip(".") if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return s[-1]
    a = adduct.strip().rstrip(".") if isinstance(adduct, str) else ""
    return a[-1] if a.endswith(("+", "-")) else ""


def mono_mz(counts: dict, sign: str) -> float:
    """The monoisotopic m/z of a singly charged ion (NaN for an unknown element
    or no sign)."""
    if sign not in ("+", "-"):
        return float("nan")
    mass = sum(ELEMENT_MASS.get(el, float("nan")) * n for el, n in counts.items())
    return mass + (ELECTRON_MASS if sign == "-" else -ELECTRON_MASS)


@lru_cache(maxsize=200000)
def _committed_candidates(ckey: tuple) -> tuple:
    counts = dict(ckey)
    opts: list[dict] = [{}]
    for iso in COMMITTED_ISOTOPES:
        el = ISOTOPE_ELEMENT[iso]
        n = int(counts.get(el, 0))
        kmax = min(n, 2) if iso in ("34S", "30Si", "29Si") else n
        new = []
        for o in opts:
            used = sum(v for k, v in o.items() if ISOTOPE_ELEMENT[k] == el)
            for k in range(0, min(kmax, n - used) + 1):
                new.append(_heavy_add(o, {iso: k}) if k else dict(o))
        opts = new
    out = []
    for o in opts:
        p = heavy_probability(o, counts)
        if p >= COMMITTED_MIN_P:
            out.append((heavy_key(o), heavy_shift(o), p))
    return tuple(out)


def committed_tolerance_ppm(klass: str | None) -> float:
    """COMMITTED_TOL_PPM of an instrument class ('orbitrap' / 'tof'); 20 ppm
    without one."""
    return COMMITTED_TOL_PPM.get(klass, COMMITTED_TOL_CLASSLESS_PPM) if klass else COMMITTED_TOL_CLASSLESS_PPM


def committed_configuration(counts: dict, d: float, mz: float, tol_ppm: float) -> dict:
    """The heavy configuration of the committed parent line that sits `d` Da
    above the ion's mono m/z: among the ion's Br / Cl / S / Si configurations
    (P >= COMMITTED_MIN_P) whose shift lies within `tol_ppm` of `d` (ppm of
    `mz`), the most probable; none within -> the mono line ({})."""
    if not (np.isfinite(d) and np.isfinite(mz)):
        return {}
    ck = heavy_key({k: v for k, v in counts.items() if k in ("Br", "Cl", "S", "Si")})
    tol = tol_ppm * 1e-6 * mz
    inside = [c for c in _committed_candidates(ck) if abs(d - c[1]) <= tol]
    if not inside:
        return {}
    return dict(max(inside, key=lambda c: c[2])[0])


_ENUM_ISOTOPES = ("13C", "81Br", "37Cl", "34S", "33S", "29Si", "30Si", "18O", "17O", "15N")


@lru_cache(maxsize=50000)
def _ion_lines(ckey: tuple, max_nominal: int = 12, floor: float = 1e-7) -> tuple:
    """Every heavy configuration of the ion (vs its mono line) with P >= floor
    and nominal shift <= max_nominal: ((key, shift, P), ...); 13C to four
    atoms, the halogens to their count, S / Si / O / N to two."""
    counts = dict(ckey)
    lines: list[tuple[dict, float]] = [({}, 1.0)]
    for iso in _ENUM_ISOTOPES:
        el = ISOTOPE_ELEMENT[iso]
        n = int(counts.get(el, 0))
        if n <= 0:
            continue
        kmax = {"13C": 4, "81Br": n, "37Cl": n}.get(iso, 2)
        new = []
        for h, _p in lines:
            used = sum(v for k, v in h.items() if ISOTOPE_ELEMENT[k] == el)
            for k in range(0, min(kmax, n - used) + 1):
                h2 = _heavy_add(h, {iso: k}) if k else dict(h)
                if _heavy_nominal(h2) > max_nominal:
                    break
                p = heavy_probability(h2, counts)
                if p < floor:
                    break
                new.append((h2, p))
        lines = new
    return tuple((heavy_key(h), heavy_shift(h), p) for h, p in lines)


def generic_expectation(counts: dict, hp: dict, delta: float, half_width: float) -> tuple[float, float]:
    """A generic 'M+n' child `delta` Da from its parent: (the ion's lines within
    `half_width` Da of that shift summed, relative to the committed line; their
    probability-weighted shift). (0.0, the nearest line's shift) when none."""
    sp = heavy_shift(hp)
    pp = heavy_probability(hp, counts)
    lines = _ion_lines(heavy_key(counts))
    near = [(s - sp, p) for _k, s, p in lines if abs((s - sp) - delta) <= half_width]
    if near and pp > 0:
        tot = sum(p for _s, p in near)
        return tot / pp, sum(s * p for s, p in near) / tot
    if not lines:
        return 0.0, delta
    return 0.0, min((s - sp for _k, s, _p in lines), key=lambda x: abs(x - delta))


def _label_elements(parsed) -> list[str]:
    els = []
    for kind, v in parsed:
        if kind == "set":
            els.extend(ISOTOPE_ELEMENT[i] for i in v)
        elif kind == "alt":
            for a in v:
                els.extend(ISOTOPE_ELEMENT[i] for i in a)
    return sorted(set(els))


def _expand(parsed) -> list[dict]:
    outs: list[dict] = [{}]
    for kind, v in parsed:
        if kind == "set":
            outs = [_heavy_add(o, v) for o in outs]
        elif kind == "alt":
            outs = [_heavy_add(o, a) for o in outs for a in v]
    return outs


def resolve_child(label, counts: dict, hp: dict, delta: float, *,
                  generic_half_width: float = GENERIC_HALF_WIDTH_DA) -> dict:
    """Read one child label against its committed parent (heavy set `hp`) at
    the measured child - parent shift `delta` (Da). Returns

      kind      'set' | 'mono' (a pure 'M0' label) | 'gen' ('M+n') | 'bad'
      shift     the exact child - parent shift of the reading nearer `delta`
      readings  the exact shifts the position test may place it at: both
                conventions (parent-relative, mono-counted); a pure 'M0' label
                only the mono line; none for 'M+n' (exempt) or an unreadable label
      expected  P(line) / P(committed line) under the nearer reading, the '+'
                parts' joint probability; 0.0 for a line the ion cannot make, a
                '14N' line (rule K judges it), an '18O' line of a Br / Cl ion,
                an 'M0' child of a mono-committed parent, an unreadable label
      elements  the elements whose heavy-atom count differs between the line and
                the committed line (a pure 'M0' line: the parent's heavy
                elements); none for 'M+n'
      parts     the label's parts other than 'M0' (their '+'-join is the label
                `iso_labels` lists)
      heavy     the line's own heavy configuration under the nearer reading
                ({} for a pure 'M0' line: the mono line); None for 'M+n' or an
                unreadable label (no configuration)"""
    raw = split_label(label)
    parsed = [parse_label_part(p) for p in raw]
    parts = [p for p, (k, _v) in zip(raw, parsed) if k != "mono"]
    kinds = {k for k, _v in parsed}
    out = dict(kind="bad", shift=float("nan"), readings=(), expected=0.0, elements=[], parts=parts, heavy=None)
    if not parsed or "bad" in kinds:
        return out
    if "gen" in kinds:
        if len(parsed) != 1:
            return out
        e, s = generic_expectation(counts, hp, delta, generic_half_width)
        return dict(out, kind="gen", shift=s, expected=e)
    sp = heavy_shift(hp)
    pp = heavy_probability(hp, counts) if hp else 1.0
    alts = _expand(parsed)
    mono_only = not parts
    if not hp:
        conventions = ("same",)
    elif mono_only:
        conventions = ("M",)          # 'M0' names the mono line, never the parent itself
    else:
        conventions = ("P", "M")
    best = None
    readings = []
    for conv in conventions:
        cands = [_heavy_add(hp, a) for a in alts] if conv in ("same", "P") else alts
        ps = [heavy_probability(h, counts) for h in cands]
        tot = sum(ps)
        shift = (sum(heavy_shift(h) * p for h, p in zip(cands, ps)) / tot if tot > 0 else heavy_shift(cands[0])) - sp
        expected = tot / pp if pp > 0 else 0.0
        if any("14N" in h for h in cands):
            expected = 0.0
        if any("18O" in h for h in cands) and any(counts.get(e, 0) for e in M2_OWNERS):
            expected = 0.0
        readings.append(shift)
        cand = dict(shift=shift, expected=expected, heavy=cands[0])
        if best is None or abs(delta - shift) < abs(delta - best["shift"]):
            best = cand
    if mono_only:
        elements = sorted({ISOTOPE_ELEMENT[i] for i in hp})
        expected = best["expected"] if hp else 0.0
    else:
        diff = dict(best["heavy"])
        for k, v in hp.items():
            diff[k] = diff.get(k, 0) - v
        elements = sorted({ISOTOPE_ELEMENT[i] for i, v in diff.items() if v}) or _label_elements(parsed)
        expected = best["expected"]
    return dict(kind="mono" if mono_only else "set", shift=best["shift"], readings=tuple(readings),
                expected=expected, elements=elements, parts=parts, heavy=dict(best["heavy"]))


def reagent_part(part: str, satellite: str | None) -> bool:
    """A label part naming the reagent halogen's heavy line alone ('81Br',
    '81Br2', '2x81Br', '81Br(pair)' on a bromide batch); a mixed or alternative
    part ('81Br37Cl(pair)', '81Br/37Cl(pair)') is not."""
    if not satellite:
        return False
    kind, v = parse_label_part(part)
    return kind == "set" and set(v) == {satellite}


def most_probable_heavy(n: int, iso: str) -> int:
    """k_c(n): how many `iso` atoms the most probable line of an ion carrying
    n atoms of its element holds (per-atom ISOTOPE_RATIO) -- the line a scorer
    commits: 81Br 0, 1, 1, 2 for Br1-Br4; 37Cl 0, 0, 0, 1 for Cl1-Cl4."""
    n = max(int(n), 0)
    r = ISOTOPE_RATIO[iso]
    return max(range(n + 1), key=lambda k: math.comb(n, k) * r ** k)


def full_count_line(v: dict, satellite: str | None, own: int) -> bool:
    """D4's last sub-point, read as a POSITION rule (2026-09-30): a kept,
    IN-BAND line (`v` a judge_source verdict) that only the ion's full count
    of the reagent halogen makes -- evidence of the NEUTRAL's halogen. The ion
    carries n atoms of it, the adduct s = n - `own` (the neutral's atoms); the
    line's heavy index relative to the ACTUAL committed line, j = k - k_P (k
    its heavy atoms, k_P the committed configuration's: `committed` on the
    verdict, read from the parent m/z), lies outside [-k_c(s), s - k_c(s)]
    (k_c `most_probable_heavy`) -- the lines an s-atom ion committed on its
    most probable line makes. A line that changes no atom of the halogen
    (j = 0: a 13C or 34S line of any parent) never counts. A line where an
    s-atom ion puts one stays the reagent's whatever its height; an 'M+n' or
    unreadable line has no index; an adduct that supplies none of the halogen
    (s <= 0) leaves nothing to tell apart, nor does a neutral that carries
    none (`own` <= 0: every line of its ion is the reagent's)."""
    if not satellite or not v["ok"] or v.get("heavy") is None:
        return False
    n = int(v["counts"].get(ISOTOPE_ELEMENT[satellite], 0))
    s = n - int(own)
    if s <= 0 or int(own) <= 0:
        return False
    j = int(v["heavy"].get(satellite, 0)) - int(v["committed"].get(satellite, 0))
    kc = most_probable_heavy(s, satellite)
    return not (-kc <= j <= s - kc)


class PositionSigma(NamedTuple):
    """sigma(h)^2 = a^2 + b^2 / h (ppm of the parent m/z; h = the child's
    height), fitted on a source's own '13C' children. `floored`: the
    intercept sits on SIGMA_A_FLOOR_PPM (the fit found no height-independent
    scatter at all)."""
    a: float
    b: float
    floored: bool
    n: int


def fit_position_sigma(residual_ppm, height) -> PositionSigma | None:
    """The height-aware sigma of a source's '13C' children (see the constants
    above); None below SIGMA_MIN_CHILDREN children (no position test there)."""
    r, h = np.asarray(residual_ppm, float), np.asarray(height, float)
    ok = np.isfinite(r) & np.isfinite(h) & (h > 0)
    r, h = r[ok], h[ok]
    if len(r) < SIGMA_MIN_CHILDREN:
        return None
    s0 = MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))
    k = np.abs(r) <= max(SIGMA_CLIP_PPM, SIGMA_CLIP_K * s0)
    r, h = r[k], h[k]
    q = np.quantile(h, np.linspace(0, 1, SIGMA_BINS + 1))
    xs, ys, ws = [], [], []
    for i in range(SIGMA_BINS):
        m = (h >= q[i]) & (h <= q[i + 1])
        if m.sum() < SIGMA_MIN_PER_BIN:
            continue
        s = MAD_TO_SIGMA * np.median(np.abs(r[m] - np.median(r[m])))
        xs.append(1 / np.median(h[m]))
        ys.append(s * s)
        ws.append(m.sum())
    X, Y, W = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ws, float)
    A = np.vstack([np.ones_like(X), X]).T * np.sqrt(W)[:, None]
    coef, *_ = np.linalg.lstsq(A, Y * np.sqrt(W), rcond=None)
    floor = SIGMA_A_FLOOR_PPM ** 2
    return PositionSigma(a=float(np.sqrt(max(coef[0], floor))), b=float(np.sqrt(max(coef[1], 0.0))),
                         floored=bool(coef[0] <= floor), n=int(len(r)))


def position_window_ppm(height, fit: PositionSigma) -> float:
    """max(POSITION_MIN_PPM, POSITION_K x sigma(h)) at a child of `height`."""
    h = float(height)
    return float(max(POSITION_MIN_PPM, POSITION_K * math.sqrt(fit.a ** 2 + fit.b ** 2 / h))) \
        if np.isfinite(h) and h > 0 else float("nan")


#: the height-ratio band a line must fall in to count (evidence.RATIO_LO / RATIO_HI)
RATIO_BAND = (0.5, 2.0)


def _in_band(ratio: float, expected: float) -> bool:
    if not (expected and expected > 0 and np.isfinite(ratio)):
        return False
    rel = ratio / expected
    return bool(np.isfinite(rel) and RATIO_BAND[0] <= rel <= RATIO_BAND[1])


def _placed(c_mz: float, bases: list, readings, w_da: float) -> bool:
    return any(np.isfinite(c_mz - (b + e)) and abs(c_mz - (b + e)) <= w_da for b in bases for e in readings)


def _pulled(c_mz: float, c_h: float, p_mz: float, readings, own: str, parent: str, rows) -> bool:
    """N1: a row of the file >= NEIGHBOUR_RATIO x the child's height, within
    NEIGHBOUR_REACH_PPM of the exact position, on the side the child is
    displaced toward, the residual <= NEIGHBOUR_FRACTION of the distance to it
    (not the child itself, not its parent)."""
    if rows is None:
        return False
    fm, fh, fp = rows
    for e in readings:
        x = p_mz + e
        r = c_mz - x
        if not np.isfinite(r):
            continue
        sel = ((fm >= x * (1 - NEIGHBOUR_REACH_PPM * 1e-6)) & (fm <= x * (1 + NEIGHBOUR_REACH_PPM * 1e-6))
               & (fh >= NEIGHBOUR_RATIO * c_h) & (fp != own) & (fp != parent))
        dj = fm[sel] - x
        if np.any((np.sign(dj) == np.sign(r)) & (abs(r) <= NEIGHBOUR_FRACTION * np.abs(dj))):
            return True
    return False


def judge_source(children, parents: dict, lists, rows: dict, *, klass: str | None = None, fwhm=None,
                 guard: bool = False) -> dict:
    """The isotope children of ONE source (a file, or a batch's pooled files)
    judged against their committed parents: position (the self-fitted window,
    pcal, N1; 'M+n' exempt), count-aware expectation and band.

      parents   {(file, peak_id): {'mz', 'height', 'pcal', 'counts', 'sign'}} --
                every M0 row (the first per (file, peak_id)); `counts` the ION's
                element counts, `sign` its charge sign
      children  [{'file', 'peak_id', 'parent', 'label', 'mz', 'height'}] -- the
                iso_child rows joined to a parent
      lists     [(file, parent peak_id, [entry dict, ...])] -- the M0 rows'
                isotopologues lists (the scorer's record of the lines it scored)
      rows      {file: (mz, height, peak_id) arrays} -- the M0 + iso_child rows
                of each file: N1's neighbours and the list entries' lookup
      klass     'orbitrap' / 'tof' / None (the width model's class): the
                committed line's tolerance (COMMITTED_TOL_PPM)
      fwhm      the width model's FWHM(m/z) in Da, or None: an 'M+n' child's
                half-width
      guard     per file only: a TOF-class source whose fit sits on the
                intercept floor is not tested (its '13C' scatter collapsed)

    Returns {'fit': PositionSigma | None, 'tested': bool, 'children': [one
    verdict per child: keep (placed), ok (in band), expected, ratio, elements,
    parts, kind, heavy (the line's configuration), committed (the parent's
    committed configuration), counts (the ion's)],
    'lists': [one bool per list row: an entry that is not this
    parent's dropped child, names an M0 / iso row of the file, is placed at its
    own height (pcal allowed, no N1) and is in band]}."""
    tol = committed_tolerance_ppm(klass)
    hp_of: dict = {}
    for key, p in parents.items():
        mono = mono_mz(p["counts"], p["sign"])
        pmz = float(p["mz"]) if p["mz"] is not None else float("nan")
        hp_of[key] = committed_configuration(p["counts"], pmz - mono, pmz, tol)

    def half_width(mz: float) -> float:
        if fwhm is None or not np.isfinite(mz):
            return GENERIC_HALF_WIDTH_DA
        return max(GENERIC_HALF_WIDTH_DA, float(fwhm(mz)) / 2)

    # the self-fit on the source's own '13C' children
    s13 = ISOTOPE_SPACING["13C"]
    res13, h13 = [], []
    for c in children:
        if str(c["label"]).strip() == "13C":
            p = parents[(c["file"], c["parent"])]
            pmz = float(p["mz"])
            res13.append((float(c["mz"]) - pmz - s13) / pmz * 1e6)
            h13.append(float(c["height"]))
    fit = fit_position_sigma(res13, h13)
    tested = fit is not None and not (guard and klass == "tof" and fit.floored)

    verdicts = []
    dropped: dict = {}
    for c in children:
        key = (c["file"], c["parent"])
        p = parents[key]
        hp = hp_of[key]
        pmz, ph = float(p["mz"]), float(p["height"])
        cmz, ch = float(c["mz"]), float(c["height"])
        r = resolve_child(c["label"], p["counts"], hp, cmz - pmz, generic_half_width=half_width(cmz))
        keep = True
        if tested and r["kind"] != "gen":
            w = position_window_ppm(ch, fit) * pmz / 1e6
            pcal = float(p["pcal"]) if p["pcal"] is not None else float("nan")
            bases = [pmz] + ([pmz * (1 - pcal / 1e6)] if np.isfinite(pcal) else [])
            keep = (_placed(cmz, bases, r["readings"], w)
                    or _pulled(cmz, ch, pmz, r["readings"], str(c["peak_id"]), str(c["parent"]),
                               rows.get(c["file"])))
        ratio = ch / ph if (np.isfinite(ph) and ph > 0) else float("nan")
        verdicts.append(dict(keep=bool(keep), ok=_in_band(ratio, r["expected"]), expected=r["expected"],
                             ratio=ratio, elements=r["elements"], parts=r["parts"], kind=r["kind"],
                             heavy=r["heavy"], committed=hp, counts=p["counts"]))
        if not keep:
            dropped.setdefault(key, set()).add(str(c["peak_id"]))

    look: dict = {}
    for f, (fm, fh, fp) in rows.items():
        for mz, h, pid in zip(fm, fh, fp):
            look.setdefault((f, str(pid)), (float(mz), float(h)))
    list_ok = []
    for f, parent, entries in lists:
        key = (f, parent)
        p = parents.get(key)
        passed = False
        if p is not None:
            hp = hp_of[key]
            pmz, ph = float(p["mz"]), float(p["height"])
            pcal = float(p["pcal"]) if p["pcal"] is not None else float("nan")
            bases = [pmz] + ([pmz * (1 - pcal / 1e6)] if np.isfinite(pcal) else [])
            gone = dropped.get(key, set())
            for e in entries:
                if not isinstance(e, dict) or str(e.get("peak_id")) in gone:
                    continue
                hit = look.get((f, str(e.get("peak_id"))))
                if hit is None or not np.isfinite(hit[0]):
                    continue
                emz, eh = hit
                r = resolve_child(e.get("label"), p["counts"], hp, emz - pmz, generic_half_width=half_width(emz))
                placed = True
                if tested and r["kind"] != "gen":
                    placed = _placed(emz, bases, r["readings"], position_window_ppm(eh, fit) * pmz / 1e6)
                ratio = eh / ph if (np.isfinite(ph) and ph > 0) else float("nan")
                if placed and _in_band(ratio, r["expected"]):
                    passed = True
                    break
        list_ok.append(passed)
    return dict(fit=fit, tested=tested, children=verdicts, lists=list_ok)


def _reagent_line(v: dict, satellite: str) -> bool:
    if v["parts"]:
        return all(reagent_part(pt, satellite) for pt in v["parts"])
    return v["elements"] == [ISOTOPE_ELEMENT[satellite]]


def line_facts(verdicts, satellite: str | None, own: int = 0) -> dict:
    """The facts one pair's KEPT lines give (`verdicts` = its placed children):
    `iso` an in-band line; `lined` the elements of its in-band lines (C17's
    multiline reads those the neutral supplies); `carbon` a line that adds
    13C; `labels` the lines' labels without 'M0' parts; `reagent_only` every
    line naming a heavy atom names only the reagent halogen's (by its label
    parts; a pure 'M0' line by what it differs in from the committed line, so
    the lighter line of a heavy-committed halogen pattern names the halogen
    and the 'M0' child of a mono-committed parent names nothing), none adds
    13C, and none is a line only the ion's full halogen count makes
    (`full_count_line`; `own` = the neutral's atoms of the halogen)."""
    lines = list(verdicts)
    ok = [v for v in lines if v["ok"]]
    naming = [v for v in lines if v["parts"] or (v.get("kind") == "mono" and v["elements"])]
    carbon = any("C" in v["elements"] for v in lines)
    return dict(
        iso=bool(ok),
        lined=set().union(*(set(v["elements"]) for v in ok)) if ok else set(),
        carbon=carbon,
        labels={"+".join(v["parts"]) for v in lines if v["parts"]},
        reagent_only=bool(satellite) and bool(naming) and not carbon
        and all(_reagent_line(v, satellite) for v in naming)
        and not any(full_count_line(v, satellite, own) for v in lines),
    )


#: the lead setters a halogen lock answers (rule H, C11+b; the engine's
#: evidence.LIFTABLE_LEADS): the three speculative-residual reasons, the
#: reference-list dim rescue and -- where the locked halogen is its only
#: violation -- the element budget; never the radical anion or the reagent-N re-read
LIFTABLE_LEADS = {"spec_n3", "spec_gapfill", "spec_minor", "reflist_dim", "off_budget"}


def lead_setters(value) -> set:
    """The setter codes a `lead_by` cell names; empty for "", NaN or None."""
    if not isinstance(value, str):
        return set()
    return {s for s in value.split("|") if s.strip() and s.strip().lower() != "nan"}


def is_ion_only(frame: pd.DataFrame) -> pd.Series:
    """Rows the ion-only stage wrote (see ION_ONLY_ADDUCTS)."""
    adduct = column(frame, "adduct", "").fillna("").astype(str).isin(ION_ONLY_ADDUCTS)
    method = column(frame, "method", "").fillna("").astype(str).str.startswith(ION_ONLY_METHOD_PREFIX)
    mask = adduct & method
    if "ion_only_of" in frame.columns:
        mask = mask | (adduct & frame["ion_only_of"].notna())
    return mask


def column(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """The column if the ledger has it, a constant column if it does not."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def resolve_source(path: str) -> tuple[str, list[str]]:
    """(label, ledger csv paths) for a run dir, a run dir's parent, or a csv."""
    path = os.path.expanduser(path.rstrip("/"))
    if os.path.isfile(path):
        return os.path.basename(path)[:-4] if path.endswith(".csv") else path, [path]
    if not os.path.isdir(path):
        raise SystemExit(f"no such source: {path}")
    label = os.path.basename(path)
    per_file = sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv")))
    if per_file:
        return label, per_file
    merged = os.path.join(path, "merged_ledger.csv")
    if os.path.isfile(merged):
        return label, [merged]
    # the --out-dir that holds exactly one run dir
    inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)]
    runs = [
        d
        for d in inner
        if os.path.isdir(os.path.join(d, "per_file"))
        or os.path.isfile(os.path.join(d, "merged_ledger.csv"))
    ]
    if len(runs) == 1:
        return resolve_source(runs[0])
    raise SystemExit(f"no ledger under {path}")


def source_resolution(path: str) -> dict | None:
    """A run dir's width model as its batch_summary.json records it (the
    `resolution` dict), for a run dir or an out-dir holding one run; None for a
    ledger CSV or a run that recorded none (a class-less source) -- the engine's
    evidence.source_resolution."""
    path = os.path.expanduser(str(path).rstrip("/"))
    if not os.path.isdir(path):
        return None
    run = path
    if not (os.path.isdir(os.path.join(path, "per_file")) or os.path.isfile(os.path.join(path, "merged_ledger.csv"))):
        inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)
                 and (os.path.isdir(os.path.join(d, "per_file")) or os.path.isfile(os.path.join(d, "merged_ledger.csv")))]
        if len(inner) != 1:
            return None
        run = inner[0]
    summary = os.path.join(run, "batch_summary.json")
    if not os.path.isfile(summary):
        return None
    try:
        res = json.load(open(summary)).get("resolution")
    except (OSError, ValueError):
        return None
    return res if isinstance(res, dict) and res.get("coef") is not None else None


#: an Orbitrap-class width model resolves at least this at m/z 200 (the engine's
#: evidence.ORBITRAP_R200 / batch/iso_checks.ORBITRAP_R200)
ORBITRAP_R200 = 50_000.0


def instrument(resolution) -> tuple:
    """(class, FWHM) of a run's `resolution` dict (batch_summary.json):
    ('orbitrap' | 'tof', FWHM(m/z) in Da); (None, None) without one -- the
    engine's evidence.instrument on the recorded model."""
    if not isinstance(resolution, dict) or resolution.get("coef") is None:
        return None, None
    coef, expo = float(resolution["coef"]), float(resolution.get("exponent", 1.0))
    off = float(resolution.get("offset", 0.0) or 0.0)

    def fwhm(mz: float) -> float:
        return coef * float(mz) ** expo + off
    r = 200.0 / fwhm(200.0)
    return ("orbitrap" if np.isfinite(r) and r >= ORBITRAP_R200 else "tof"), fwhm


def load_source(path: str) -> tuple[str, pd.DataFrame]:
    """Every ledger row of one source, stamped with the file it came from."""
    label, files = resolve_source(path)
    frames = []
    for f in files:
        frame = pd.read_csv(f, low_memory=False)
        frame["__file"] = os.path.basename(f)[:16]
        frames.append(frame)
    return label, pd.concat(frames, ignore_index=True)


def detect_reagent_halogen(m0: pd.DataFrame) -> str | None:
    """The halogen of the channel's commonest CLUSTER adduct, or None.

    A bromide channel clusters on Br; a nitrate channel does not, even when a
    stray `[M+Br]-` row is present. Counting only cluster adducts separates the
    two without naming any instrument.
    """
    adducts = column(m0, "adduct").dropna().astype(str)
    clusters = adducts[adducts.isin(CLUSTER_ADDUCTS)]
    if clusters.empty:
        return None
    top = clusters.value_counts().idxmax()
    for halogen in ("Br", "Cl", "I"):
        if count_element(top, halogen):
            return halogen
    return None


def measure_source(
    label: str, ledger: pd.DataFrame, halogen: str | None, resolution: dict | None = None,
    per_file: bool = False,
) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed.
    `resolution`: the source's width model (`source_resolution`; None =
    class-less); `per_file`: the source is one file of a run."""
    role = column(ledger, "role").astype(str)
    m0 = ledger[role == "M0"].copy()
    iso = ledger[role == "iso_child"]
    if m0.empty:
        return pd.DataFrame()

    m0["neutral_formula"] = column(m0, "neutral_formula").fillna("").astype(str)
    m0["adduct"] = column(m0, "adduct").fillna("").astype(str)
    m0["ion_only"] = is_ion_only(m0).to_numpy()
    regular = m0[~m0["ion_only"]]

    # the isotope children and the isotopologues lists, judged against their
    # committed parents (C11+c; `judge_source` above -- the engine's
    # evidence._judge_children): a child counts where it sits at its label's
    # exact spacing from the committed line (the source's self-fitted window,
    # pcal, N1; 'M+n' exempt), in band under its count-aware expectation; a
    # dropped child is dropped for every fact
    satellite = HALOGEN_SATELLITE.get(halogen) if halogen else None
    for name in ("peak_id", "ion_formula", "mz", "height", "ppm_error_cal", "isotopologues"):
        if name not in m0.columns:
            m0[name] = np.nan
    klass, fwhm = instrument(resolution)
    counts_of: dict = {}
    parents: dict = {}
    pair_of: dict = {}
    for f, pid, n, a, ion, mz, h, pc in zip(m0["__file"].astype(str), m0["peak_id"].astype(str),
                                            m0["neutral_formula"], m0["adduct"], m0["ion_formula"], m0["mz"],
                                            m0["height"], m0["ppm_error_cal"]):
        key = (f, pid)
        if key in parents:
            continue
        ck = (n, a, ion if isinstance(ion, str) else "")
        if ck not in counts_of:
            counts_of[ck] = ({e: v for e, v in ion_counts(n, a, ion).items() if v}, ion_sign(ion, a))
        counts, sign = counts_of[ck]
        parents[key] = dict(mz=to_float(mz), height=to_float(h), pcal=to_float(pc), counts=counts, sign=sign)
        pair_of[key] = (n, a)
    children = []
    for f, pid, par, lab, mz, h in zip(iso["__file"].astype(str), column(iso, "peak_id").astype(str),
                                       column(iso, "parent_peak_id"), column(iso, "iso_label"), column(iso, "mz"),
                                       column(iso, "height")):
        if not isinstance(par, str) and pd.isna(par):
            continue
        key = (f, str(par))
        if key not in parents:
            continue
        lab = str(lab).strip() if pd.notna(lab) else ""
        if not lab.split("+")[0].strip():
            continue
        children.append(dict(file=f, peak_id=pid, parent=str(par), label=lab, mz=to_float(mz), height=to_float(h)))
    lists = [(f, pid, as_list(v)) for f, pid, v in zip(m0["__file"].astype(str), m0["peak_id"].astype(str),
                                                        m0["isotopologues"])]
    readable = ledger[role.isin(["M0", "iso_child"])]
    rows_by_file = {str(f): (pd.to_numeric(column(g, "mz"), errors="coerce").to_numpy(float),
                             pd.to_numeric(column(g, "height"), errors="coerce").to_numpy(float),
                             column(g, "peak_id").astype(str).to_numpy(object))
                    for f, g in readable.groupby(readable["__file"].astype(str), sort=False)}
    judged = judge_source(children, parents, [x for x in lists if x[2]], rows_by_file, klass=klass, fwhm=fwhm,
                          guard=per_file)
    kept: dict = {}
    for c, verdict in zip(children, judged["children"]):
        if verdict["keep"]:
            kept.setdefault(pair_of[(c["file"], c["parent"])], []).append(verdict)
    # the neutral's atoms of the reagent halogen: what the adduct supplies is the rest (D4's full-count line)
    x = ISOTOPE_ELEMENT[satellite] if satellite else None
    facts_of = {k: line_facts(v, satellite, composition(str(k[0])).get(x, 0) if x else 0) for k, v in kept.items()}
    no_lines = line_facts([], None)
    listed = iter(judged["lists"])
    m0["__list_ok"] = [bool(next(listed)) if entries else False for _f, _p, entries in lists]

    # per-neutral facts, over the source's REGULAR M0 rows (an ion-only row is
    # its parent's composition on another adduct, not a second channel for it)
    channels = regular.groupby("neutral_formula")["adduct"].nunique()
    adduct_sets = regular.groupby("neutral_formula")["adduct"].agg(
        lambda s: set(s.dropna().astype(str))
    )

    for name in (
        "method",
        "confidence",
        "degeneracy_note",
        "resolvability",
        "series_unit",
        "anchor_peak_id",
        "isotopologues",
        "tied",
        "below_assignability",
        # C19(c): the lead half of the old flag; a ledger written before the
        # split carries none, reads False, and keeps its leads in below
        "tentative_lead",
        # C11+b: the lead's setter (ledger.LEAD_SETTERS); missing = any setter
        "lead_by",
        "degeneracy_density",
        "ion_formula",
        "tier",
        "height",
        "mz",
        "ppm_error_cal",
        "occurrence_y",
    ):
        if name not in m0.columns:
            m0[name] = np.nan

    rows = []
    for (neutral, adduct), group in m0.groupby(["neutral_formula", "adduct"]):
        key = (neutral, adduct)
        lines = facts_of.get(key) or no_lines
        methods = group["method"].astype(str)
        known = methods[methods.str.startswith("known:")]
        degeneracy = pd.to_numeric(group["degeneracy_density"], errors="coerce")
        seen = {
            str(v).strip()
            for v in group["resolvability"].dropna()
            if str(v).strip() and str(v).strip().lower() != "nan"
        }
        carbon_ev = lines["carbon"]
        ion_only = bool(group["ion_only"].any())
        below_row = group["below_assignability"].map(truthy).astype(bool)
        lead_row = group["tentative_lead"].map(truthy).astype(bool)
        clean_row = ~group["ion_only"].astype(bool) & ~below_row & ~lead_row
        lead_codes = [lead_setters(v) for v in group.loc[lead_row, "lead_by"]]
        own = lines["lined"] & neutral_elements(
            neutral, str(group["ion_formula"].iloc[0]))
        aset = set() if ion_only else adduct_sets.get(neutral, set())
        rows.append(
            dict(
                source=label,
                neutral=neutral,
                adduct=adduct,
                ion_only=ion_only,
                ion=str(group["ion_formula"].iloc[0]),
                mz=float(pd.to_numeric(group["mz"], errors="coerce").median()),
                tier="Assigned" if (group["tier"] == "Assigned").any() else "Candidate",
                known_fam=known.iloc[0][6:] if len(known) else "",
                # an in-band kept line, or a list entry that answers the same question
                iso=lines["iso"] or bool(group["__list_ok"].any()),
                multiline=len(own) >= 2,
                multiline_elements="|".join(sorted(own)),
                carbon_ev=carbon_ev,
                chan2=(not ion_only) and int(channels.get(neutral, 0)) >= 2,
                anchor=(not ion_only) and bool(
                    group["anchor_peak_id"].notna().any()
                    or group["series_unit"].notna().any()
                ),
                branch=bool(aset & BARE_ADDUCTS) and bool(aset & CLUSTER_ADDUCTS),
                # every kept line naming a heavy atom names only the reagent
                # halogen's, none adds 13C, none is a line only the ion's full
                # halogen count makes (D4) -- and the reagent put it there: the
                # ION carries more of the reagent halogen than the neutral
                # (C11+a; the Br-free hold released by C11+c)
                reagent_only_iso=(not ion_only)
                and lines["reagent_only"]
                and carries_reagent(neutral, adduct, group["ion_formula"].iloc[0], halogen),
                iso_labels="|".join(sorted(lines["labels"])),
                tied=bool(group["tied"].map(truthy).all()),
                below=bool(below_row.any()),
                lead=bool(lead_row.any()),
                # rule H (C11+b): the setters behind the pair's lead rows (a lead
                # row naming none -- a ledger written before `lead_by` -- may be
                # the element budget's), whether it has an unflagged regular row,
                # and the anchor its unflagged rows give on their own
                lead_by="|".join(sorted(set().union(*lead_codes))) if lead_codes else "",
                lead_unknown=any(not c for c in lead_codes),
                clean_row=bool(clean_row.any()),
                # a pair with a regular (not ion-only) row gives its adduct to its
                # neutral's pools, row by row like the engine (relabel_pools)
                has_regular=bool((~group["ion_only"].astype(bool)).any()),
                anchor_clean=(not ion_only) and bool(
                    (group["anchor_peak_id"].notna() | group["series_unit"].notna())[~below_row & ~lead_row].any()),
                lowconf=bool(
                    group["confidence"].map(first_word).isin(LOW_CONFIDENCE).all()
                ),
                degeneracy=float(degeneracy.median())
                if degeneracy.notna().any()
                else np.nan,
                saturated=bool(
                    group["degeneracy_note"]
                    .astype(str)
                    .str.contains("MASS-SATURATED")
                    .any()
                ),
                # resolvability is measured on the trace-first path only; a
                # source that never measured it is not penalised for it.
                res_ok=(not seen) or bool(seen & RESOLVED),
                resolvability="|".join(sorted(seen)),
                n_files=int(group["__file"].nunique()),
                height=float(pd.to_numeric(group["height"], errors="coerce").median()),
                ppm=float(
                    pd.to_numeric(group["ppm_error_cal"], errors="coerce").median()
                ),
                occurrence=float(
                    pd.to_numeric(group["occurrence_y"], errors="coerce").median()
                ),
            )
        )
    return pd.DataFrame(rows)


def series_level_of(row) -> str:
    """The decision table. Order matters: the first predicate that holds wins."""
    # a tentative lead (C19(c)) is hard like below assignability; a halogen lock
    # (rule H, C11+b) that lifts it has already set `lead` False (lift_leads)
    hard = (bool(row.tied) or bool(row.below) or bool(getattr(row, "lead", False))
            or bool(row.lowconf) or bool(getattr(row, "label_veto", False))
            or bool(getattr(row, "iso_veto", False)))
    degenerate = bool(row.saturated) or (
        pd.notna(row.degeneracy) and row.degeneracy >= 3
    )
    unique = pd.notna(row.degeneracy) and row.degeneracy <= 1
    if hard:
        return "5b"
    if bool(getattr(row, "ion_only", False)):
        return "4d" if row.iso else "5a"
    if degenerate and row.n_axes == 0:
        return "5b"
    if row.known_fam:
        if (KNOWN_FAMILY_SCOPE.get(row.known_fam, "class") == "compound"
                and plausible_structures(row.neutral) == 1):
            return "2b"
        return "3a"
    if row.branch:
        return "3b"
    if row.n_axes == 0:
        return "4c" if (unique and row.res_ok) else "5a"
    if (not row.neutral_backed) and row.reagent_only_iso:
        return "4d"
    if row.n_axes >= 2 and row.cross:
        return "4a"
    # 9' (rule U): the profile's neutral pair establishes the neutral; the
    # formula needs its own support -- an isotope, or one plausible ion on a
    # resolved peak
    if bool(getattr(row, "upair", False)) and (row.iso or (unique and row.res_ok)):
        return "4a"
    return "4b"


#: rule K's one-channel fold (the 14N and 15N nitrate clusters of one neutral)
LABEL_FOLD = {"[M+NO3]-": "[M+^NO3]-", "[M+15NO3]-": "[M+^NO3]-"}


def relabel_pools(df: pd.DataFrame, alien: set, fold: bool = True) -> pd.DataFrame:
    """Rule K / C11+ on the per-neutral facts: chan2 and branch recomputed over
    the regular rows minus the `alien` pairs (rule K's 14N lines, the pairs an
    isotope check refutes), with `fold` the two nitrate clusters of one neutral
    counted as one channel (rule K only); an alien row takes neither. Row by
    row, like the engine: a pair holding an ion-only row beside a regular one
    still gives its adduct to its siblings (`has_regular`; a measured frame
    without the column reads the pair-level ion-only flag) and takes neither
    itself."""
    df = df.copy()
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    alien_row = pd.Series([k in alien for k in keys], index=df.index, dtype=bool)
    out = alien_row | df["ion_only"].astype(bool)
    has_regular = (df["has_regular"].astype(bool) if "has_regular" in df.columns
                   else ~df["ion_only"].astype(bool))
    reg = df[~alien_row & has_regular]
    chans = reg.assign(ch=reg["adduct"].map(lambda a: LABEL_FOLD.get(a, a) if fold else a)) \
        .groupby("neutral")["ch"].nunique()
    adds = reg.groupby("neutral")["adduct"].agg(lambda s: set(s.astype(str)))
    df["chan2"] = [(not o) and int(chans.get(n, 0)) >= 2 for n, o in zip(df["neutral"], out)]
    df["branch"] = [(not o) and bool(adds.get(n, set()) & BARE_ADDUCTS) and bool(adds.get(n, set()) & CLUSTER_ADDUCTS)
                    for n, o in zip(df["neutral"], out)]
    return df


def lift_leads(df: pd.DataFrame, lock: dict, alien: set, fold: bool) -> pd.DataFrame:
    """Rule H (C11+b): a locked pair whose lead rows were all set by a setter
    the lock answers (the element budget's only where `budget_ok`), none below
    assignability, not alien / ion-only / refuted, is lifted: `lead` False, the
    lock its isotope axis, no reagent-only flag, and its second channel / acid
    branch / anchor read over unflagged rows alone -- its own adduct plus the
    adducts its neutral commits on an unflagged regular row of a pair that is
    not alien (`fold`: the two nitrate clusters count as one channel). Row by
    row, like the engine: a pair holding an ion-only row beside such a regular
    row still gives its adduct (`clean_row` already leaves the ion-only rows out)."""
    df = df.copy()
    df["lead_lift"] = False
    df["lock_note"] = ""
    if not lock:
        return df
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    alien_row = pd.Series([k in alien for k in keys], index=df.index, dtype=bool)
    out = alien_row | df["ion_only"].astype(bool)
    clean = df[df["clean_row"].astype(bool) & ~alien_row].groupby("neutral")["adduct"].agg(
        lambda s: set(s.astype(str)))
    for idx, k in zip(df.index, keys):
        spec = lock.get(k)
        if spec is None or out[idx] or bool(df.at[idx, "below"]) or not bool(df.at[idx, "lead"]):
            continue
        if bool(df.at[idx, "label_veto"]) or bool(df.at[idx, "iso_veto"]):
            continue
        setters = lead_setters(df.at[idx, "lead_by"])
        budget_ok = bool(spec.get("budget_ok", False))
        if not (setters <= LIFTABLE_LEADS and ("off_budget" not in setters or budget_ok)
                and (not bool(df.at[idx, "lead_unknown"]) or budget_ok)):
            continue
        own = {k[1]} | set(clean.get(k[0], set()))
        df.at[idx, "chan2"] = len({LABEL_FOLD.get(a, a) if fold else a for a in own}) >= 2
        df.at[idx, "branch"] = bool(own & BARE_ADDUCTS) and bool(own & CLUSTER_ADDUCTS)
        df.at[idx, "anchor"] = bool(df.at[idx, "anchor_clean"])
        df.at[idx, "iso"] = True
        df.at[idx, "reagent_only_iso"] = False
        df.at[idx, "lead"] = False
        df.at[idx, "lead_lift"] = True
        df.at[idx, "lock_note"] = str(spec.get("note", "") or "")
    return df


def assign_levels(df: pd.DataFrame, corroborating: set[str], upair: set[str] | None = None,
                  label: dict | None = None, iso: dict | None = None) -> pd.DataFrame:
    """Add the axes, the derived flags and the level to measured rows. `upair`
    is the batch's neutral-pair set (rule U); `label` the labelled-nitrate twin
    facts (rule K: {'untie', 'veto', 'alien'}, `label_twin_facts`): the 14N lines
    it could not tie to their neutral's 15N cluster leave the neutral's pools,
    the two clusters count as one channel, a tie the 15N sibling breaks is
    cleared and a refuted reading is hard; `iso` the isotope checks' vetoes and
    locks (C11+: {'veto': {(n, a): note}, 'lock': {(n, a): fact}},
    `iso_check_facts`): a refuted pair is hard and leaves its neutral's pools
    like an alien line; a locked pair's lead the lock answers is lifted
    (`lift_leads`, rule H). Facts, never axes."""
    df = df.copy()
    df["known_fam"] = df["known_fam"].fillna("")
    if "ion_only" not in df.columns:
        df["ion_only"] = False
    df["corroborated"] = df["neutral"].isin(corroborating) & ~df["ion_only"].astype(bool)
    df["upair"] = df["neutral"].isin(upair or set()) & ~df["ion_only"].astype(bool)
    iso_veto = {(str(n), str(a)): str(v or "") for (n, a), v in ((iso or {}).get("veto") or {}).items()}
    if label or iso_veto:
        alien = {(str(n), str(a)) for n, a in ((label or {}).get("alien") or set())} | set(iso_veto)
        df = relabel_pools(df, alien, fold=bool(label))
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    untie = {(str(n), str(a)) for n, a in ((label or {}).get("untie") or set())}
    veto = {(str(n), str(a)) for n, a in ((label or {}).get("veto") or {})}
    df["label_untie"] = (pd.Series([k in untie for k in keys], index=df.index, dtype=bool)
                         & df["tied"].astype(bool) & ~df["ion_only"].astype(bool))
    df.loc[df["label_untie"], "tied"] = False
    df["label_veto"] = pd.Series([k in veto for k in keys], index=df.index, dtype=bool) & ~df["ion_only"].astype(bool)
    # C11+: an isotope check refutes the formula (an ion-only pair too)
    df["iso_veto"] = pd.Series([k in iso_veto for k in keys], index=df.index, dtype=bool)
    df["iso_note"] = [iso_veto.get(k, "") if v else "" for k, v in zip(keys, df["iso_veto"])]
    # rule H (C11+b): the halogen locks lift the leads they answer
    lock = {(str(n), str(a)): dict(v or {}) for (n, a), v in ((iso or {}).get("lock") or {}).items()}
    for col, default in (("lead_by", ""), ("lead_unknown", True), ("clean_row", False), ("anchor_clean", False)):
        if col not in df.columns:
            df[col] = default
    df = lift_leads(df, lock, {(str(n), str(a)) for n, a in ((label or {}).get("alien") or set())} | set(iso_veto),
                    fold=bool(label))
    df["n_axes"] = df[["iso", "chan2", "anchor", "corroborated"]].sum(axis=1)
    df["cross"] = df.corroborated | df.multiline | df.known_fam.ne("")
    df["neutral_backed"] = (
        df.corroborated | df.chan2 | df.anchor | df.known_fam.ne("") | df.carbon_ev
    )
    df["level"] = df.apply(series_level_of, axis=1)
    return df


#: a source corroborates the neutrals it holds at one of these levels by its own
#: evidence (4b or better; 1 and 2a never fire)
CORROBORATING_LEVELS = {"1", "2a", "2b", "3a", "3b", "4a", "4b"}
OWN_AXES = {"iso", "chan2", "anchor"}


def own_good_neutrals(frame: pd.DataFrame) -> set[str]:
    """The neutrals a measured source holds at 4b or better when it is levelled
    with NO corroboration — its own evidence only. Ion-only pairs never count."""
    own = assign_levels(frame, set())
    ok = own["level"].isin(CORROBORATING_LEVELS) & ~own["ion_only"].astype(bool)
    return set(own.loc[ok, "neutral"].astype(str)) - {""}


def stored_good_neutrals(ledger: pd.DataFrame, label: str) -> set[str]:
    """A merged ledger (no predicate columns): the rows its stored level puts at
    4b or better that still hold an axis of their own once `corroborated` is
    taken away (a level the axis alone produced does not count)."""
    if "evidence_level" not in ledger.columns:
        raise SystemExit(f"{label}: neither per-file predicate columns nor an evidence_level column")
    ledger = ledger[~is_ion_only(ledger)]
    keep = []
    for level, axes in zip(ledger["evidence_level"], column(ledger, "evidence_axes", "")):
        parts = set(str(axes).split("|")) if pd.notna(axes) else set()
        good = str(level) in CORROBORATING_LEVELS
        if good and "corroborated" in parts:
            own = parts & OWN_AXES
            good = bool(own) and not (own == {"iso"} and "reagent_only_iso" in parts)
        keep.append(good)
    return set(column(ledger, "neutral_formula")[np.array(keep, dtype=bool)].dropna().astype(str)) - {""}


def source_good_neutrals(path: str) -> tuple[str, pd.DataFrame | None, set[str]]:
    """(label, measured frame or None, the neutrals it corroborates) for one source."""
    label, ledger = load_source(path)
    if "role" not in ledger.columns:
        return label, None, stored_good_neutrals(ledger, label)
    halogen = detect_reagent_halogen(ledger[column(ledger, "role").astype(str) == "M0"])
    frame = measure_source(label, ledger, halogen, source_resolution(path))
    if frame.empty:
        return label, None, set()
    frame["reagent_halogen"] = halogen or ""
    return label, frame, own_good_neutrals(frame)


def upair_neutrals(path: str) -> set[str]:
    """The neutrals whose neutral pair holds, read from a batch's
    tables/neutral_pairs.csv (a run dir, or an --out-dir holding one run) or
    from that CSV itself. A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "neutral_pairs.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "neutral_pairs.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --upair: no neutral_pairs.csv for {path}; rule U does not fire there", file=sys.stderr)
        return set()
    frame = pd.read_csv(table)
    if "upair" not in frame.columns:
        return set()
    held = frame["upair"].map(truthy)
    return set(frame.loc[held, "neutral_formula"].astype(str))


def label_twin_facts(path: str) -> dict | None:
    """{'untie', 'veto', 'alien'} -- the labelled-nitrate twin facts (rule K)
    read from a batch's tables/label_twins.csv (a run dir, or an --out-dir holding
    one run) or from that CSV itself; None when there is no table or it is empty
    (out of scope). A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "label_twins.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "label_twins.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --label-twins: no label_twins.csv for {path}; rule K does not fire there", file=sys.stderr)
        return None
    frame = pd.read_csv(table)
    if frame.empty:
        return None
    out = {}
    for col in ("untie", "veto", "alien"):
        if col not in frame.columns:
            out[col] = set()
            continue
        held = frame[col].map(truthy)
        out[col] = set(zip(frame.loc[held, "neutral_formula"].astype(str), frame.loc[held, "adduct"].astype(str)))
    return out


#: the order the isotope checks' notes join in, and each check's name there
ISO_CHECKS = ("C", "REQ", "HIGH", "H")
ISO_CHECK_NAME = {"C": "rule C", "REQ": "REQ", "HIGH": "HIGH", "H": "rule H"}


def iso_check_facts(path: str) -> dict | None:
    """{'veto': {(neutral, adduct): note}, 'lock': {(neutral, adduct): {'element',
    'n', 'budget_ok', 'note'}}} -- the isotope checks (C11+) read from a batch's
    tables/iso_checks.csv (a run dir, or an --out-dir holding one run) or from
    that CSV itself: one note per pair joined over the checks that refute it
    ('rule C: ...; REQ: ...'), and rule H's locks (C11+b; a table written before
    rule H has none); None when there is no table or it is empty (no time
    series). A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "iso_checks.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "iso_checks.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --iso-checks: no iso_checks.csv for {path}; the isotope checks do not fire there", file=sys.stderr)
        return None
    frame = pd.read_csv(table)
    if frame.empty:
        return None
    veto: dict = {}
    if "veto" in frame.columns:
        held = frame[frame["veto"].map(truthy)].copy()
        order = {c: i for i, c in enumerate(ISO_CHECKS)}
        held["__o"] = held["check"].map(lambda c: order.get(str(c), len(order))) if "check" in held.columns else 0
        held = held.sort_values("__o", kind="mergesort")
        for _, r in held.iterrows():
            key = (str(r["neutral_formula"]), str(r["adduct"]))
            name = ISO_CHECK_NAME.get(str(r.get("check", "")), str(r.get("check", "")))
            note = r.get("note")
            piece = f"{name}: {note}" if isinstance(note, str) and note else name
            veto[key] = f"{veto[key]}; {piece}" if key in veto else piece
    lock: dict = {}
    if "lock" in frame.columns and "check" in frame.columns:
        for _, r in frame[(frame["check"].astype(str) == "H") & frame["lock"].map(truthy)].iterrows():
            n = pd.to_numeric(r.get("n_halogen"), errors="coerce")
            el, note = r.get("element"), r.get("note")
            lock[(str(r["neutral_formula"]), str(r["adduct"]))] = {
                "element": el if isinstance(el, str) else "", "n": int(n) if pd.notna(n) else 0,
                "budget_ok": truthy(r.get("budget_ok")), "note": note if isinstance(note, str) else ""}
    return {"veto": veto, "lock": lock}


def series_run(sources: list[str], corroborate: list[str], upair: str | None = None,
        twins: str | None = None, iso: str | None = None) -> pd.DataFrame:
    """Level every source, each corroborated by the others plus --corroborate —
    by the neutrals each of them holds at 4b or better on its own evidence.
    `upair`: 'auto' reads each run-dir source's own tables/neutral_pairs.csv;
    a CSV path applies that table to every levelled source; None = rule U off.
    `twins`: the same for tables/label_twins.csv (rule K). `iso`: the same for
    tables/iso_checks.csv (C11+, its vetoes and rule H's locks); a named table
    applies only to a source that holds every pair it vetoes or locks -- the
    checks were measured on one batch's time series and name its pairs, so a
    source missing one of them is another batch and the table does not fire
    there."""
    measured = {}
    neutrals = {}
    for path in sources:
        label, frame, good = source_good_neutrals(path)
        if frame is None:
            print(f"  {label}: no M0 rows, skipped", file=sys.stderr)
            continue
        measured[path] = (label, frame)
        neutrals[path] = good
    for path in corroborate:
        neutrals[f"--corroborate:{path}"] = source_good_neutrals(path)[2]
    out = []
    for path, (label_, frame) in measured.items():
        others: set[str] = set()
        for other, values in neutrals.items():
            if other != path:
                others |= values
        held = set()
        if upair == "auto":
            held = upair_neutrals(path)
        elif upair:
            held = upair_neutrals(upair)
        label = None
        if twins == "auto":
            label = label_twin_facts(path)
        elif twins:
            # one named table applies only to the source it was measured on: every
            # pair it names (untie, veto, alien) must be one of the source's pairs
            facts = label_twin_facts(twins)
            named = set().union(*(set(facts[k]) for k in ("untie", "veto", "alien"))) if facts else set()
            pairs = set(zip(frame["neutral"].astype(str), frame["adduct"].astype(str)))
            if facts is not None and named <= pairs:
                label = facts
            elif facts is not None:
                print(f"  --label-twins: {label_} does not hold the pairs the table names; rule K does not fire there",
                      file=sys.stderr)
        checks = None
        if iso == "auto":
            checks = iso_check_facts(path)
        elif iso:
            checks = iso_check_facts(iso)
            pairs = set(zip(frame["neutral"].astype(str), frame["adduct"].astype(str)))
            missing = (set((checks or {}).get("veto") or {}) | set((checks or {}).get("lock") or {})) - pairs
            if missing:
                print(f"  --iso-checks: {label_} does not hold {len(missing)} pair(s) the table vetoes or locks "
                      f"(another batch's table); the isotope checks do not fire there", file=sys.stderr)
                checks = None
        out.append(assign_levels(frame, others, held, label, checks))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def series_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CIMS-adapted Schymanski evidence levels over a peaky ledger."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="run dirs or ledger CSVs; two or more corroborate each other",
    )
    parser.add_argument(
        "--corroborate",
        action="append",
        default=[],
        help="a source that corroborates but is not itself levelled",
    )
    parser.add_argument(
        "--upair",
        nargs="?",
        const="auto",
        default=None,
        help="rule U: read each batch source's tables/neutral_pairs.csv (no value), "
        "or apply this neutral-pair CSV to every levelled source",
    )
    parser.add_argument(
        "--label-twins",
        nargs="?",
        const="auto",
        default=None,
        dest="twins",
        help="rule K: read each batch source's tables/label_twins.csv (no value), "
        "or apply this label-twin CSV to every levelled source",
    )
    parser.add_argument(
        "--iso-checks",
        nargs="?",
        const="auto",
        default=None,
        dest="iso",
        help="C11+: read each batch source's tables/iso_checks.csv (no value), "
        "or apply this isotope-check CSV to the levelled sources that hold every pair it vetoes or locks "
        "(its vetoes refute; rule H's locks lift the tentative leads they answer)",
    )
    parser.add_argument("--out", help="write the levelled rows here as CSV")
    args = parser.parse_args(argv)

    df = series_run(args.sources, args.corroborate, args.upair, args.twins, args.iso)
    if df.empty:
        print("nothing to level", file=sys.stderr)
        return 1
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)

    order = [k for k in SERIES_LEVEL_ORDER if k not in ("1", "2a")]
    for label, group in df.groupby("source", sort=False):
        halogen = group["reagent_halogen"].iloc[0] or "none"
        counts = group.level.value_counts()
        vector = "/".join(str(int(counts.get(k, 0))) for k in order)
        print(f"{label}  n={len(group)}  reagent halogen {halogen}")
        print(f"  {'/'.join(order)}")
        print(f"  {vector}")
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
