"""Step 2 of the evidence scale: is the neutral / adduct split of the ion pinned?

The committed ion is decomposed over the run's adducts (the profile's, every
non-ion-only adduct the run committed, [M+NH4]+ beside [M+H]+, and the pair's
own adduct) minus the LOCKED side channels; each decomposition's neutral is
tested for plausibility (`space.Space.plausible_neutral`). The split is pinned
when the ion carries the labelled reagent atom on the labelled channel (the 15N
label), or the committed reading is the only plausible decomposition; it is
open otherwise. Rule order (the first that decides wins):

  no ion composition -> open | ion-only channel -> open | the label -> PINNED |
  ^N on an unlabelled adduct -> open | the amine gate (positive runs with
  [M+NH4]+ in the grid) | only decomposition -> PINNED | halogen count -> PINNED
  (as coded it is subsumed by "only decomposition") | committed neutral not
  plausible -> open | k decompositions -> open.

THE AMINE GATE. On a run where [M+NH4]+ is the reagent's own channel it stays in
the grid and the engine's amine gate decides each NH4 reading (Si -> kept;
protected identity -> kept; tracks its own [M+H]+ / urea parent -> kept, a
positive fact; else re-read as the amine unless that amine is
valence-impossible). The NH4 ADMISSIBILITY RULE: for a committed [M+H]+, an NH4
reading Y [M+NH4]+ whose Y has no uronium adduct ION in the run (by composition,
under any reading, in any ledger) while one of them lies in the scan leaves the
set before the gate.

THE CONTEXT WINDOW. A reading excluded only by the context window still counts
(the split OPENS) when it is the X+reagent isobar of the committed cluster; any
other window-only exclusion keeps a pin and is disclosed.

SIDE CHANNELS are locked by default (`evidence.SIDE_CHANNELS_LOCKED`,
`evidence.UNLOCKED`): formate, acetate, CO3-, O2-, O3-, NH4+ (except the
uronium run's own channel), Na+ and chloride never enter the grid; where one
would open a pinned split the split text says so.

Each split returns its gate facts (`gi`) with it: no global state.
"""
from __future__ import annotations

import numpy as np

from peaky.assignment.levels import space as SP
from peaky.chem import chemistry as C

# the as-built side channels (the unlocked split's extra grid; the as-built note's names)
SIDE = {"negative": ["[M+CHO2]-", "[M+C2H3O2]-", "[M+CO3]-", "[M+O2]-", "[M+NO3]-"],
        "positive": ["[M+NH4]+", "[M+Na]+"]}
SIDE_NAME = {"[M+CHO2]-": "formate", "[M+C2H3O2]-": "acetate", "[M+CO3]-": "CO3-", "[M+O2]-": "O2-",
             "[M+NO3]-": "14N nitrate", "[M+NH4]+": "NH4+", "[M+Na]+": "Na+"}
# every side channel the scale locks (adduct -> channel name); [M+O3]- has no adduct composition in the code base
SIDE_CHANNEL_NAMES = {"[M+CHO2]-": "formate", "[M+C2H3O2]-": "acetate", "[M+CO3]-": "CO3-", "[M+O2]-": "O2-",
                      "[M+O3]-": "O3-", "[M+NH4]+": "NH4+", "[M+^NH4]+": "NH4+", "[M+Na]+": "Na+"}
# [M+NH4]+ on a positive run is the reagent's own channel there (decided by the amine gate), not a side channel.
# Keyed by POLARITY, not by the reagent profile (as built; see BACKLOG).
OWN_REAGENT_CHANNEL = {"positive": {"[M+NH4]+"}}
# chloride: a side channel no run scored; locked like formate, named where it would open a pinned split (a re-run
# of the split with it). Its lock only blocks an ADDITION: a grid that already holds [M+Cl]- keeps it (as built).
V1_SIDE_EXTRA = {"negative": {"[M+Cl]-": "chloride"}}
# the X+reagent isobar's reagent molecules (keyed by polarity, as built)
REAGENT_MOLS = {"positive": {"urea": {"C": 1, "H": 4, "N": 2, "O": 1}},
                "negative": {"HNO3": {"H": 1, "N": 1, "O": 3}, "H^NO3": {"H": 1, "^N": 1, "O": 3},
                             "HBr": {"H": 1, "Br": 1}}}
URONIUM_PARENTS = ("[M+H]+", "[M+(CH4N2O)H]+")
AMINE_R_MIN = 0.6            # assign_batch.run(amine_r_min=0.6): the run's own value when it recorded one
AMINE_R_REJECT = 0.2         # cleanup.prefer_amine_over_ammonium's default
AMINE_MIN_OVERLAP = 12       # cleanup.prefer_amine_over_ammonium's default (2-h bins)
GATE_WHY = {"reject": "independent time trace", "ambiguous": "only weak tracking",
            "presence-cap": "parent flat, adduct unconfirmable", "presence-reread": "parent channels absent",
            "presence-keep": "no time series to confirm"}
NO_TS_WHY = "no time series: nothing can be confirmed"
_PARENT_LABEL = {"[M+H]+": "[M+H]+", "[M+(CH4N2O)H]+": "urea cluster"}


def is_locked(channel_name: str) -> bool:
    """A side channel is locked unless the single switch is off or it is in evidence.UNLOCKED."""
    from peaky.assignment import evidence as EV
    return bool(EV.SIDE_CHANNELS_LOCKED) and channel_name not in EV.UNLOCKED


def locked_adducts() -> set:
    return {a for a, nm in SIDE_CHANNEL_NAMES.items() if is_locked(nm)}


def _b(v) -> bool:
    from peaky.assignment import evidence as EV
    return bool(EV.truthy(v)) if not isinstance(v, (bool, np.bool_)) else bool(v)


# ---------------------------------------------------------------------------
# the amine gate
# ---------------------------------------------------------------------------
def _ion_key(ic):
    return tuple(sorted((k, int(v)) for k, v in (ic or {}).items() if v))


def _canon(f):
    try:
        return C.format_formula({k: v for k, v in C.parse_formula(str(f)).items() if v})
    except Exception:  # noqa: BLE001
        return str(f)


def _mz(Y, ad):
    try:
        return float(C.ion_mz(Y, ad))
    except Exception:  # noqa: BLE001
        return float("nan")


def _where(reading, srcs):
    """'C9H20N2O [M+H]+ (merged, 1 per-file)' for the NH4 admissibility rule's ion index."""
    nm = "merged" in srcs
    npf = len([x for x in srcs if x != "merged"])
    w = ", ".join(x for x in (("merged" if nm else ""), (f"{npf} per-file" if npf else "")) if x)
    return f"{reading} ({w})"


class AmineGate:
    """The engine's amine gate (`cleanup.prefer_amine_over_ammonium`'s order)
    as a decision on one neutral X read as [M+NH4]+, plus the NH4 admissibility
    rule's ion index. ``ledger`` = the merged ledger (a source's pair table on a
    one-file source); ``ts_peaks`` = the batch time series (None: the presence
    path); ``protected`` = the neutrals the gate never re-reads; ``scan`` =
    (first, last) m/z of the run's peaks; ``per_file`` = {sid: ledger} (their
    M0 rows join the ion index)."""

    def __init__(self, ledger, ts_peaks, protected, scan=None, per_file=None, *, r_min: float = AMINE_R_MIN):
        from peaky.assignment import cleanup as CU
        self.verdict = CU._covariation_verdict(ledger, ts_peaks, r_min, AMINE_R_REJECT, AMINE_MIN_OVERLAP)
        self.protected = set(protected)
        self.has_ts = ts_peaks is not None
        self.memo: dict = {}
        self.parents = {ad: {_canon(z) for z in ledger.loc[ledger["adduct"] == ad, "neutral_formula"].dropna().astype(str)}
                        for ad in URONIUM_PARENTS}
        self.ions: dict = {}
        for src, led in [("merged", ledger)] + sorted((per_file or {}).items()):
            L = led
            if "role" in L.columns and src != "merged":
                L = L[L["role"].astype(str) == "M0"]
            sub = L[["neutral_formula", "adduct"]].dropna().astype(str).drop_duplicates()
            for nf, ad in zip(sub["neutral_formula"], sub["adduct"]):
                ic = SP.ion_counts_of(nf, ad)
                if ic:
                    self.ions.setdefault(_ion_key(ic), {}).setdefault(f"{nf} {ad}", set()).add(src)
        self.scan = scan
        self.adm_memo: dict = {}

    def nh4_admissibility(self, Y) -> dict:
        """The NH4 admissibility rule: dict(ok, why, present, testable, own_name).
        ok False = no uronium adduct ion of Y in the run although one lies in the scan."""
        if Y in self.adm_memo:
            return self.adm_memo[Y]
        present, testable, offscan = [], [], []
        for ad in URONIUM_PARENTS:
            hits = self.ions.get(_ion_key(SP.ion_counts_of(Y, ad)), {})
            if hits:
                present.append((ad, [_where(rd, srcs) for rd, srcs in sorted(hits.items())]))
            mz = _mz(Y, ad)
            inside = self.scan is None or (np.isfinite(mz) and self.scan[0] <= mz <= self.scan[1])
            (testable if inside else offscan).append((ad, mz))
        lab = _PARENT_LABEL
        if present:
            g = dict(ok=True, why="uronium adduct ion present: " + "; ".join(
                f"{lab[a]} m/z {_mz(Y, a):.4f} as {' / '.join(h)}" for a, h in present))
        elif not testable:
            g = dict(ok=True, why="both uronium adducts of Y lie outside the scan (" + ", ".join(
                f"{lab[a]} m/z {m:.2f}" for a, m in offscan) + "): alias stays admissible")
        else:
            g = dict(ok=False, why="no uronium adduct ion of Y present (merged + per-file ledgers): " + ", ".join(
                f"{lab[a]} m/z {m:.2f} in scan, absent" for a, m in testable) + "".join(
                f", {lab[a]} m/z {m:.2f} outside the scan (not counted)" for a, m in offscan))
        g.update(present=[a for a, _h in present], testable=[a for a, _ in testable],
                 own_name=any(_canon(Y) in self.parents[a] for a, _h in present))
        self.adm_memo[Y] = g
        return g

    def __call__(self, X) -> dict:
        if X in self.memo:
            return self.memo[X]
        cnt = C.parse_formula(X)
        if cnt.get("Si"):
            g = dict(kept=True, track=False, verdict="Si", r=np.nan, ov=0, how="Si (siloxane NH4 adducts are real)")
        elif X in self.protected:
            g = dict(kept=True, track=False, verdict="protected", r=np.nan, ov=0,
                     how="protected identity (reflist-rescue / known / certified)")
        else:
            v, r, ov = self.verdict(X)
            if v == "keep":
                g = dict(kept=True, track=True, verdict=v, r=r, ov=ov,
                         how=f"tracks its own [M+H]+/urea parent, r {r:.2f} over {ov} 2-h bins")
            else:
                am = dict(cnt)
                am["N"] = am.get("N", 0) + 1
                am["H"] = am.get("H", 0) + 3
                ok, _ = C.dbe_ok(am)
                why = GATE_WHY.get(v, "unconfirmed") + (
                    f", r {r:.2f}" if v in ("reject", "ambiguous") and np.isfinite(r) else "")
                if not self.has_ts and v.startswith("presence"):
                    why = NO_TS_WHY
                if not ok:
                    g = dict(kept=True, track=False, verdict="amine-impossible", r=r, ov=ov,
                             how=f"amine {C.format_formula(am)} valence-impossible (kept unconfirmed; {why})")
                else:
                    g = dict(kept=False, track=False, verdict=v, r=r, ov=ov, how=why, amine=C.format_formula(am))
        self.memo[X] = g
        return g


def protected_neutrals(per_file: dict) -> set:
    """The union of `assign_batch._protected_neutrals` over the per-file ledgers."""
    from peaky.batch import assign_batch as AB
    out = set()
    for led in per_file.values():
        out |= AB._protected_neutrals(led)
    return out


def scan_range(per_file: dict):
    """(first, last) m/z over every per-file row with height > 0 (any role)."""
    import pandas as pd
    lo, hi = [], []
    for led in per_file.values():
        mz = pd.to_numeric(led["mz"], errors="coerce")[pd.to_numeric(led["height"], errors="coerce") > 0]
        lo.append(float(mz.min()))
        hi.append(float(mz.max()))
    return (min(lo), max(hi)) if lo else None


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------
def _dtxt(ds, k=5):
    return "; ".join(f"{d['neutral']} {d['adduct']}" for d in ds[:k]) + (f" (+{len(ds) - k})" if len(ds) > k else "")


def side_name(ad):
    return SIDE_CHANNEL_NAMES.get(ad) or next((m[ad] for m in V1_SIDE_EXTRA.values() if ad in m), ad)


def _window_only(decs, n, a, nh4_out):
    """Grid readings that fail ONLY the run's context window (the first space
    profile's `context filter` reason), not the committed reading, and not an
    NH4 reading the gate drops or the admissibility rule removes."""
    out = []
    for d in decs:
        if d["ok"] or not d["neutral"] or not str(d["why"]).startswith("context filter"):
            continue
        if d["adduct"] == a and SP.same_formula(d["neutral"], n):
            continue
        if d["adduct"] == "[M+NH4]+" and nh4_out is not None and nh4_out(d["neutral"]):
            continue
        out.append(d)
    return out


def reagent_isobar(pol, a, d) -> str:
    """The reagent molecule R when the window-excluded reading d is the
    'X plus reagent' isobar of the committed reading (a = d's adduct + R); else ''."""
    da, dd = SP.adduct_delta(a), SP.adduct_delta(d["adduct"])
    if da is None or dd is None:
        return ""
    diff = {k: da.get(k, 0) - dd.get(k, 0) for k in set(da) | set(dd)}
    diff = {k: v for k, v in diff.items() if v}
    for name, R in REAGENT_MOLS.get(pol, {}).items():
        if diff == R:
            return name
    return ""


def _wtxt(ws):
    return "; ".join(f"{d['neutral']} {d['adduct']} ({str(d['why']).replace('context filter: ', '')})" for d in ws)


def new_gate_info() -> dict:
    return dict(kind="", nh4=[], note="", track=False, window="", inadm=[], isobar="")


def as_built_split(S, r):
    """The split over the decomposition adducts + EVERY side channel, no gate,
    no admissibility or window rule: (pinned, plausible). Only the locked
    channels' 'would open it' note reads it."""
    ctx = S.ctx
    n, a = str(r.neutral_formula), str(r.adduct)
    counts = SP.ion_counts_of(n, a, getattr(r, "ion", None))
    if not counts:
        return False, []
    ads = list(dict.fromkeys(list(ctx.decomp_adducts) + [x for x in SIDE[S.pol] if SP.adduct_delta(x) is not None]))
    decs = ctx.decompositions(counts, n, a, adducts=ads)
    plaus = [d for d in decs if d["ok"] and d["neutral"]]
    self_ok = any(d["adduct"] == a and SP.same_formula(d["neutral"], n) for d in plaus)
    ion_only = _b(r.ion_only) or a in SP.ION_ONLY
    ion_label = ctx.labelled and counts.get("^N", 0) > 0
    if ion_only:
        return False, plaus
    if ion_label and "^" in a and not _b(r.label_veto) and (n, a) not in S.alien:
        return True, plaus
    if ion_label and "^" not in a:
        return False, plaus
    if len(plaus) == 1 and self_ok:
        return True, plaus
    matched = {x for x in str(r.committed_matched).split(",") if x and x != "nan"}
    for el in ("Br", "Cl"):
        nx = counts.get(el, 0)
        if nx and el in matched and nx > SP.reagent_supply(ads, el):
            keep = [d for d in plaus if d["counts"].get(el, 0) > 0]
            if len(keep) == 1 and keep[0]["adduct"] == a:
                return True, plaus
    return False, plaus


def split_core(S, r, ads):
    """The split over the grid ``ads``: (pinned, how, plausible, label_pin, gate info)."""
    ctx = S.ctx
    n, a = str(r.neutral_formula), str(r.adduct)
    gi = new_gate_info()
    counts = SP.ion_counts_of(n, a, getattr(r, "ion", None))
    if not counts:
        return False, "no ion composition", [], False, gi
    decs = ctx.decompositions(counts, n, a, adducts=ads)
    plaus = [d for d in decs if d["ok"] and d["neutral"]]
    ion_only = _b(r.ion_only) or a in SP.ION_ONLY
    ion_label = ctx.labelled and counts.get("^N", 0) > 0
    if ion_only:
        return False, "ion-only channel (process open)", plaus, False, gi
    if ion_label and "^" in a and not _b(r.label_veto) and (n, a) not in S.alien:
        return True, "label (the ion carries the reagent's ^N)", plaus, True, gi
    if ion_label and "^" not in a:
        lab = [d for d in plaus if "^" in d["adduct"]]
        return False, ("label points to the cluster reading: " + _dtxt(lab, 3) if lab
                       else "the ion carries ^N but no plausible cluster reading"), plaus, False, gi
    # ---- the amine gate (a positive run with [M+NH4]+ in the grid) ----
    gate = S.gate
    note = ""
    inadm_txt = ""
    adm_sfx = ""
    use_gate = gate is not None and "[M+NH4]+" in ads
    adm_on = use_gate and a == "[M+H]+"      # the NH4 admissibility rule: committed [M+H]+ (amine) readings only

    def adm_refused(Y):
        return adm_on and not gate.nh4_admissibility(Y)["ok"]

    if use_gate:
        if a == "[M+NH4]+":
            # a committed NH4 reading is decided by the gate alone: other decompositions are not looked at and the
            # admissibility rule does not apply (as built; see BACKLOG)
            g = gate(n)
            gi.update(kind="committed NH4", nh4=[(n, g)], track=g["track"])
            if g["kept"]:
                how = ("NH4 adduct tracks its parent (" + g["how"] + ")") if g["track"] else \
                    ("NH4 adduct kept by the amine gate: " + g["how"])
                return True, how, plaus, False, gi
            gi["kind"] = "amine default"
            return False, (f"amine default (NH4 reading unconfirmed: {g['how']}; the engine's gate reads this ion as "
                           f"{g.get('amine', '?')} [M+H]+)"), plaus, False, gi
        nh4_all = [d for d in plaus if d["adduct"] == "[M+NH4]+"]
        bad = [d for d in nh4_all if adm_refused(d["neutral"])]
        nh4 = [d for d in nh4_all if not any(d is x for x in bad)]
        if bad:
            gi["inadm"] = [(d["neutral"], gate.nh4_admissibility(d["neutral"])["why"], gate(d["neutral"])) for d in bad]
            inadm_txt = "; ".join(f"{d['neutral']} [M+NH4]+ -- {gate.nh4_admissibility(d['neutral'])['why']}" for d in bad)
            plaus = [d for d in plaus if not any(d is x for x in bad)]
        res = [(d, gate(d["neutral"])) for d in nh4]
        if adm_on:
            gi["adm"] = [(d["neutral"], gate.nh4_admissibility(d["neutral"])) for d in nh4]
            other = [(Y, ga) for Y, ga in gi["adm"] if ga["present"] and not ga["own_name"]]
            if other:
                adm_sfx = (" [NH4 admissibility rule: Y's uronium adduct ion is present under another reading: "
                           + "; ".join(f"{Y} -- {ga['why'].replace('uronium adduct ion present: ', '')}"
                                       for Y, ga in other) + "]")
        gi["nh4"] = [(d["neutral"], g) for d, g in res] + [(Y, g) for Y, _w, g in gi["inadm"]]
        kept = [(d, g) for d, g in res if g["kept"]]
        dropped = [(d, g) for d, g in res if not g["kept"]]
        plaus = [d for d in plaus if not (d["adduct"] == "[M+NH4]+" and any(d is x for x, _ in dropped))]
        others = [d for d in plaus if not (d["adduct"] == a and SP.same_formula(d["neutral"], n))
                  and d["adduct"] != "[M+NH4]+"]
        if a == "[M+H]+" and res:
            # an [M+H]+ with an admissible NH4 reading is always OPEN (before the window rules)
            if kept:
                gi["kind"] = "NH4 kept"
                return False, ("NH4 reading kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ ({g['how']})" for d, g in kept)
                    + " -- the committed [M+H]+ (amine) reading is contested"
                    + (f"; also {len(others)} other decomposition(s): {_dtxt(others, 3)}" if others else "")
                    + adm_sfx), plaus, False, gi
            gi["kind"] = "amine default"
            return False, ("amine default (NH4 reading unconfirmed: " + "; ".join(
                f"{d['neutral']} [M+NH4]+ -- {g['how']}" for d, g in dropped) + ")"
                + (f"; also {len(others)} other decomposition(s): {_dtxt(others, 3)}" if others else "")
                + adm_sfx), plaus, False, gi
        if a == "[M+H]+" and bad:
            gi["kind"] = "NH4 inadmissible"
        elif res or bad:
            parts = []
            if dropped:
                gi["kind"] = "NH4 dropped"
                parts.append("NH4 reading not kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ -- {g['how']}" for d, g in dropped))
            if kept:
                gi["kind"] = "NH4 kept" if not dropped else "NH4 kept+dropped"
                parts.append("NH4 reading kept by the amine gate: " + "; ".join(
                    f"{d['neutral']} [M+NH4]+ ({g['how']})" for d, g in kept))
            if bad:
                gi["kind"] = gi["kind"] or "NH4 inadmissible"
                parts.append("NH4 reading inadmissible: " + inadm_txt)
            note = "; ".join(parts)
    gi["note"] = note
    sfx = f" [{note}]" if note else ""
    if use_gate and a != "[M+NH4]+":
        sfx += adm_sfx
    # the inadmissible-NH4 prefix (rows 'only decomposition' .. 'k decompositions'; not 'committed not plausible')
    pre = (f"NH4 reading inadmissible: no uronium adduct of Y ({inadm_txt}); ") if (a == "[M+H]+" and inadm_txt) else ""
    # ---- the context window; the X+reagent isobar is not excluded by it ----
    nh4_out = None
    if use_gate:
        nh4_out = (lambda Y: adm_refused(Y) or not gate(Y)["kept"])
    win = _window_only(decs, n, a, nh4_out)
    iso = [(d, reagent_isobar(S.pol, a, d)) for d in win]
    iso = [(d, R) for d, R in iso if R]
    if iso:
        gi["isobar"] = "; ".join(f"{d['neutral']} {d['adduct']} (X+{R}; window "
                                 f"{str(d['why']).replace('context filter: ', '')})" for d, R in iso)
        plaus = plaus + [d for d, _R in iso]
        win = [d for d in win if not any(d is x for x, _ in iso)]
        sfx += f" [X+reagent isobar admitted over the context window: {gi['isobar']}]"
    self_ok = any(d["adduct"] == a and SP.same_formula(d["neutral"], n) for d in plaus)
    if len(plaus) == 1 and self_ok:
        if win:      # only the context window removes the other reading(s): disclosed
            gi["window"] = _wtxt(win)
            sfx += f" [pinned only by the context window: {gi['window']}]"
        return True, pre + "only decomposition" + sfx, plaus, False, gi
    # rule (iii), the halogen count: as coded it never fires -- n_el above every grid adduct's supply means every
    # decomposition carries the element, so keep == plaus and "only decomposition" above already decided it
    matched = {x for x in str(r.committed_matched).split(",") if x and x not in ("nan", "<NA>")}
    for el in ("Br", "Cl"):
        nx = counts.get(el, 0)
        if nx and el in matched and nx > SP.reagent_supply(ads, el):
            keep = [d for d in plaus if d["counts"].get(el, 0) > 0]
            if len(keep) == 1 and keep[0]["adduct"] == a:
                wk = [d for d in win if d["counts"].get(el, 0) > 0]
                if wk:
                    gi["window"] = _wtxt(wk)
                    sfx += f" [pinned only by the context window: {gi['window']}]"
                return True, pre + "halogen count" + sfx, plaus, False, gi
    if not self_ok:
        why = next((d["why"] for d in decs if d["adduct"] == a), "")
        return False, (f"committed neutral not plausible ({why}); {len(plaus)} plausible: " + _dtxt(plaus, 4) + sfx), \
            plaus, False, gi
    return False, pre + f"{len(plaus)} decompositions: " + _dtxt(plaus) + sfx, plaus, False, gi


def split(S, r) -> dict:
    """The scale's split of pair ``r`` on source ``S`` (ctx, pol, alien, gate):
    dict(pinned, how, plaus, flags, label_pin, gi). The grid = the decomposition
    adducts and the unlocked side channels; where a LOCKED channel would open a
    pinned split (the unlocked as-built split is open; chloride: a re-run with
    it) the text says so."""
    a = str(r.adduct)
    lk = locked_adducts() - OWN_REAGENT_CHANNEL.get(S.pol, set())
    grid = [x for x in S.ctx.decomp_adducts if x not in lk]
    grid += [x for x in SIDE[S.pol] if x not in lk and x not in grid and SP.adduct_delta(x) is not None]
    xside = V1_SIDE_EXTRA.get(S.pol, {})
    x_locked = [ad for ad, nm in xside.items() if is_locked(nm) and ad not in grid]
    grid += [ad for ad, nm in xside.items() if not is_locked(nm) and ad not in grid]   # the side-channel sweep hook
    pinned, how, plaus, lab, gi = split_core(S, r, grid)
    pu, plu = as_built_split(S, r)
    allmine = {(d["neutral"], d["adduct"]) for d in plaus}
    extra = [d for d in plu if (d["neutral"], d["adduct"]) not in allmine and d["adduct"] != a and d["adduct"] in lk]
    x_open = []
    if pinned and x_locked:
        pc, _hc, plc, _lc, _gc = split_core(S, r, grid + x_locked)
        if not pc:
            x_open = [d for d in plc if d["adduct"] in x_locked and d["adduct"] != a]
    gi["xside_open"] = "; ".join(f"{d['neutral']} {d['adduct']}" for d in x_open)
    opener = (extra if (pinned and not pu) else []) + x_open
    extra_txt = "; ".join(f"{side_name(d['adduct'])}: {d['neutral']} {d['adduct']}" for d in extra + x_open)
    if pinned and opener:
        names = sorted({side_name(d["adduct"]) for d in opener})
        otxt = "; ".join(f"{side_name(d['adduct'])}: {d['neutral']} {d['adduct']}" for d in opener)
        how = f"{how}; side channels locked ({', '.join(names)} would open it: {otxt})"
    flags = ("locked side channels: " + extra_txt) if (extra or x_open) else ""
    return dict(pinned=bool(pinned), how=how, plaus=plaus, flags=flags, label_pin=bool(lab), gi=gi)
