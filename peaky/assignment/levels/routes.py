"""Routes, homologue ladders and other-source partners.

None of these lifts a level on the evidence scale: "two routes" (the same
neutral seen in both of the reagent's channels in the same files), a CH2 / CF2
ladder and an other-source partner are TAGS, printed with the run's measured
base rate beside them. They still matter to step 1: the in-pass decision (the
scale's own internal pass, `decide.inpass`) reads a pair with qualifying routes
as a route-anchored pair, and an anchored CH2 / CF2 series excludes a left
competitor whose own homologues are absent at the anchors' spacings (series
exclusion).

Pure: frames in, dicts out. The formula helpers (`fcounts` & co.) are shared
with the decision layer.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from peaky.assignment.levels import space as SP
from peaky.chem import chemistry as C

# ---------------------------------------------------------------------------
# formula helpers (count dicts; zero counts dropped)
# ---------------------------------------------------------------------------
_FC: dict = {}


def fcounts(f):
    """A formula's element counts (zero counts dropped); None for an empty,
    NaN or unreadable formula. Memoised (pure); callers never mutate the dict."""
    if not isinstance(f, str) or not f or f == "nan":
        return None
    hit = _FC.get(f)
    if hit is None:
        try:
            hit = {k: int(v) for k, v in C.parse_formula(f).items() if v}
        except Exception:  # noqa: BLE001 -- an unreadable formula has no counts
            hit = {}
        _FC[f] = hit
    return hit or None


def fkey(counts):
    """A hashable key of a count dict (sorted items); None for no counts."""
    return tuple(sorted(counts.items())) if counts else None


def fadd(counts, unit, k):
    """counts + k x unit; None when a count goes negative."""
    out = dict(counts)
    for el, v in unit.items():
        out[el] = out.get(el, 0) + k * v
    if any(v < 0 for v in out.values()):
        return None
    return {e: v for e, v in out.items() if v}


def fsub(a, b):
    """a - b as counts; None unless every count is >= 0 and the difference is non-empty."""
    if not a or not b:
        return None
    out = dict(a)
    for e, v in b.items():
        out[e] = out.get(e, 0) - v
    if any(v < 0 for v in out.values()):
        return None
    out = {e: v for e, v in out.items() if v}
    return out or None


def ion_key(n, a, ion=None):
    """The ion's count key (`space.ion_counts_of`, ^N kept apart); None without counts."""
    c = SP.ion_counts_of(n, a, ion)
    return fkey(c) if c else None


def _b(v) -> bool:
    from peaky.assignment import evidence as EV
    return bool(EV.truthy(v)) if not isinstance(v, (bool, np.bool_)) else bool(v)


# ---------------------------------------------------------------------------
# route classes
# ---------------------------------------------------------------------------
ROUTE_CLASS = {
    "[M-H]-": "deprotonation",
    "[M+NO3]-": "nitrate cluster", "[M+15NO3]-": "nitrate cluster", "[M+^NO3]-": "nitrate cluster",
    "[M+Br]-": "bromide cluster", "[M+HBr+Br]-": "bromide cluster",
    "[M+Cl]-": "chloride cluster", "[M+HCl+Cl]-": "chloride cluster",
    "[M+CO3]-": "carbonate",
    "[M+H]+": "protonation",
    "[M+(CH4N2O)H]+": "urea cluster",
    "[M+NH4]+": "ammonium", "[M+^NH4]+": "ammonium",
    "[M+Na]+": "sodium",
    "[M]+.": "NO+-type (charge transfer)", "[M-H]+": "NO+-type (hydride abstraction)",
    "[M-CH3]+": "NO+-type (methyl loss)", "[M+H-H2O]+": "NO+-type (dehydration)",
}
# the route classes of side-channel adducts (never a route class while their channel is locked)
SIDE_CLASSES = {"carbonate": "CO3-", "ammonium": "NH4+", "sodium": "Na+"}
ROUTE_COFILES = 2            # two routes: both route ions seen in the SAME file in >= 2 files (1 on a one-file source)
LADDER_MIN_FILES = 2         # a ladder (in-pass level and tag) needs the pair seen in >= 2 files
CROSS_MIN_FILES = 2          # a pair receiving an other-source partner must itself be seen in >= 2 files


def route_class(adduct: str):
    """The adduct's route (ionization chemistry) class; None for an ion-only
    channel; an adduct not in ROUTE_CLASS is its own class."""
    if adduct in SP.ION_ONLY:
        return None
    return ROUTE_CLASS.get(adduct, adduct)


def route_label(A, B, tag=""):
    s = {A, B}
    if "deprotonation" in s and s & {"nitrate cluster", "bromide cluster"}:
        return "acidic H" + tag
    if s == {"protonation", "urea cluster"}:
        return "protonated + urea cluster" + tag
    return f"two routes ({' + '.join(sorted(s))})" + tag


def routes_table(per_file: dict, run_classes: set, excl: set, cofiles_min: int, skip_m0=frozenset()):
    """Per neutral: the per-file route classes of its committed M0 rows (not
    ion-only, not excluded, not explained as another committed ion's isotope
    line there, on one of the run's own route classes) and the class pairs
    co-seen in the same file, counted over files. Also byfile[sid][class] =
    {formula key} (the route ions; the base rate's null reads it)."""
    from peaky.assignment import evidence as EV
    per = defaultdict(lambda: defaultdict(set))
    ads = defaultdict(lambda: defaultdict(set))
    byfile = defaultdict(lambda: defaultdict(set))
    for sid, led in per_file.items():
        m0 = led[led["role"] == "M0"]
        io = EV.is_ion_only(m0).to_numpy()
        keys = list(zip(m0["neutral_formula"].fillna("").astype(str), m0["adduct"].fillna("").astype(str)))
        for (n, a), isio in zip(keys, io):
            if not n or not a or isio or (n, a) in excl or (sid, n, a) in skip_m0:
                continue
            cl = route_class(a)
            if cl is None or cl not in run_classes:
                continue          # only the reagent's own channels are route classes (side channels never)
            per[n][sid].add(cl)
            ads[n][cl].add(a)
            fk = fkey(fcounts(n))
            if fk:
                byfile[sid][cl].add(fk)
    out = {}
    for n, by in per.items():
        co = Counter()
        for cls in by.values():
            cl = sorted(cls)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    co[(cl[i], cl[j])] += 1
        out[n] = dict(classes=set().union(*by.values()), co=co, adducts=ads[n])
    return out, byfile


# ---------------------------------------------------------------------------
# the route base rates (reported beside a route / partner tag; never a gate)
# ---------------------------------------------------------------------------
# The null asks the route question of formula NEIGHBOURS X + delta: if X is seen in both channels about as often as
# X + delta, "two routes" carries no formula-specific information.
BR_DELTAS = {"+CH2": {"C": 1, "H": 2}, "-CH2": {"C": -1, "H": -2}, "+O": {"O": 1}, "-O": {"O": -1},
             "+H2": {"H": 2}, "-H2": {"H": -2}, "+C2H4": {"C": 2, "H": 4}, "-C2H4": {"C": -2, "H": -4},
             "+CO": {"C": 1, "O": 1}, "-CO": {"C": -1, "O": -1}, "+H2O": {"H": 2, "O": 1}, "-H2O": {"H": -2, "O": -1},
             "+O2": {"O": 2}, "-O2": {"O": -2}, "+CH2O": {"C": 1, "H": 2, "O": 1}}
BR_ALIAS = {"negative": {"formate Y = X - CH2O2": {"C": -1, "H": -2, "O": -2},
                         "acetate Y = X - C2H4O2": {"C": -2, "H": -4, "O": -2}},
            "positive": {"NH4+ Y = X - NH3": {"N": -1, "H": -3}}}
BR_ROUTE = {"negative": ("deprotonation", "nitrate cluster"), "positive": ("protonation", "urea cluster")}


def _cho(c):
    return bool(c) and set(c) <= {"C", "H", "O"}


def _rates(X, hit_of, pol):
    def rate(d):
        if not X:
            return np.nan
        h = 0
        for x in X:
            y = fadd(dict(x), d, 1) if d else dict(x)
            if y and hit_of(x, fkey(y)):
                h += 1
        return h / len(X)
    r0 = rate({})
    nulls = {k: rate(d) for k, d in BR_DELTAS.items()}
    al = {k: rate(d) for k, d in BR_ALIAS[pol].items()}
    nv = np.array([v for v in nulls.values() if np.isfinite(v)])
    med = float(np.median(nv)) if len(nv) else np.nan
    return dict(n=len(X), rate0=r0, null_median=med, null_lo=float(nv.min()) if len(nv) else np.nan,
                null_hi=float(nv.max()) if len(nv) else np.nan, lr=(r0 / med) if med and np.isfinite(med) else np.nan,
                nulls=nulls, aliases=al)


def base_rate_own(pol: str, run_classes: set, byfile, keys, heights, cof: int):
    """Own two-route base rate: CHO neutrals X seen on the base route
    (deprotonation / protonation) in some file; hit = X + delta seen on the
    cluster route in the SAME file in >= ``cof`` files."""
    if pol not in BR_ROUTE:
        return None
    A, B = BR_ROUTE[pol]
    if A not in run_classes or B not in run_classes:
        return None
    hx = {}
    for (nn, aa), h in zip(keys, heights):
        if route_class(aa) == A:
            c = fcounts(nn)
            if _cho(c):
                hx[fkey(c)] = max(hx.get(fkey(c), 0.0), float(h) if np.isfinite(h) else 0.0)
    X = {x for x in hx if any(x in byfile[sid][A] for sid in byfile)}

    def hit(x, y):
        return sum(1 for sid in byfile if x in byfile[sid][A] and y in byfile[sid][B]) >= cof
    out = dict(route=f"{A} + {B}", all=_rates(X, hit, pol))
    if X:
        thr = float(np.median([hx[x] for x in X]))
        out["bright"] = _rates({x for x in X if hx[x] >= thr}, hit, pol)
    return out


def base_rate_cross(pol: str, run_classes: set, partners, keys, eligible):
    """Other-source base rate: CHO neutrals X of the pairs that may receive a
    partner (``eligible``); hit = X + delta has a partner through a class this
    source lacks."""
    if not partners:
        return None
    part = set()
    for nn, cc in partners.items():
        if any(cl not in run_classes for cl in cc):
            fk = fkey(fcounts(nn))
            if fk:
                part.add(fk)
    X = set()
    for i, (nn, _aa) in enumerate(keys):
        c = fcounts(nn)
        if eligible[i] and _cho(c):
            X.add(fkey(c))
    return dict(all=_rates(X, lambda x, y: y in part, pol))


#: a base rate with no likelihood ratio: no formula neighbour X±delta was a hit (a tiny batch), so the null
#: rate is 0 and the ratio undefined -- said so, never printed as "LR nan"
BR_TOO_FEW = "base rate n/a (too few CHO neutrals)"


def br_text(b, what="formula seen in both channels"):
    if not b or not np.isfinite(b.get("rate0", np.nan)):
        return ""
    if not np.isfinite(b.get("lr", np.nan)):
        return BR_TOO_FEW
    return (f"{what}: base rate {100 * b['rate0']:.0f} % vs {100 * b['null_median']:.0f} % for formula neighbours "
            f"X±delta (LR {b['lr']:.2f}, {b['n']} CHO neutrals)")


# ---------------------------------------------------------------------------
# route co-variation (printed in the route tag: r of log heights over the spectra both ions share)
# ---------------------------------------------------------------------------
def ts_groups(ts: pd.DataFrame | None):
    """{(neutral, adduct): DataFrame(height, index = sample_item_id)} of the
    batch time series' committed M0 rows (the tallest row per spectrum); None
    without a time series."""
    if ts is None:
        return None
    t = ts[["sample_item_id", "role", "neutral_formula", "adduct", "height"]]
    t = t[(t["role"] == "M0") & t["neutral_formula"].notna() & t["adduct"].notna()]
    t = t.sort_values("height", ascending=False).drop_duplicates(["neutral_formula", "adduct", "sample_item_id"])
    g = {}
    for (n, a), x in t.groupby(["neutral_formula", "adduct"], sort=False):
        g[(n, a)] = x.set_index("sample_item_id")[["height"]]
    return g


def covary(groups, k1, k2):
    """(r of log heights over the spectra both readings share, n shared); (NaN, n) below 3 or flat."""
    if groups is None:
        return np.nan, 0
    g1, g2 = groups.get(k1), groups.get(k2)
    if g1 is None or g2 is None:
        return np.nan, 0
    j = g1.join(g2, how="inner", lsuffix="_1", rsuffix="_2")
    j = j[(j.iloc[:, 0] > 0) & (j.iloc[:, 1] > 0)]
    n = len(j)
    if n < 3:
        return np.nan, n
    x, y = np.log(j.iloc[:, 0].to_numpy(float)), np.log(j.iloc[:, 1].to_numpy(float))
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


# ---------------------------------------------------------------------------
# homologue ladders and series exclusion
# ---------------------------------------------------------------------------
UNITS = {"CH2": {"C": 1, "H": 2}, "C2H4": {"C": 2, "H": 4}, "CF2": {"C": 1, "F": 2}, "O": {"O": 1}, "2O": {"O": 2}}
LADDER_UNITS = {u: UNITS[u] for u in ("CH2", "CF2")}   # the scale's homologue units (ladders AND series exclusion)
MONO = {"C": 12.0, "H": 1.00782503207, "O": 15.99491461956, "F": 18.99840322}
HOMO_MIN_ANCHORS = 2         # anchored = >= 2 route-anchored chain members at the right spacing


def unit_mass(unit):
    return sum(MONO[e] * v for e, v in unit.items())


def kfmt(k, uname):
    return f"{'+' if k > 0 else '-'}{abs(k) if abs(k) > 1 else ''}{uname}" if uname != "2O" else \
        f"{'+' if k > 0 else '-'}{abs(k) if abs(k) > 1 else ''}(2O)"


def homologue_facts(D: pd.DataFrame, member, anchor, *, tol_ppm: float, units=None) -> dict:
    """Per pair index i: the best unit's chain and anchors. A chain = the
    contiguous run of MEMBER pairs (same adduct, neutral +/- k x unit, k = 1,
    2, ... until a position has no member) through i; anchors = chain members
    with ``anchor`` set whose observed spacing (the pooled m/z of D) agrees
    with k x unit within tol_ppm x sqrt(2). The best unit: most ok anchors,
    then anchors, then chain length, then UNITS order. {i: dict(unit,
    anchored, anchors, chain, n_ok, units)}."""
    n_all = D["neutral_formula"].astype(str).to_numpy()
    a_all = D["adduct"].astype(str).to_numpy()
    mz_all = pd.to_numeric(D["mz"], errors="coerce").to_numpy(float)
    lv_all = D["_lv"].astype(str).to_numpy() if "_lv" in D.columns else np.array([""] * len(D))
    by_ad = defaultdict(dict)
    for i in np.flatnonzero(member):
        c = fcounts(n_all[i])
        if c and a_all[i] not in SP.ION_ONLY:
            by_ad[a_all[i]][fkey(c)] = i
    out = {}
    for i in range(len(D)):
        c0 = fcounts(n_all[i])
        if not c0 or a_all[i] in SP.ION_ONLY:
            continue
        ulist = []
        for uname, u in (units or UNITS).items():
            chain = []
            for sgn in (1, -1):
                k = 1
                while True:
                    c2 = fadd(c0, u, sgn * k)
                    j = by_ad[a_all[i]].get(fkey(c2)) if c2 else None
                    if j is None or j == i:
                        break
                    chain.append((sgn * k, j))
                    k += 1
            if not chain:
                continue
            anc = []
            for k, j in chain:
                if not anchor[j]:
                    continue
                exp = k * unit_mass(u)
                obs = mz_all[j] - mz_all[i]
                tol = tol_ppm * np.sqrt(2.0) * 1e-6 * max(mz_all[j], mz_all[i])
                sp_ok = bool(np.isfinite(obs) and abs(obs - exp) <= tol)
                anc.append(dict(k=k, j=j, n=n_all[j], lv=lv_all[j], sp_ok=sp_ok, ok=sp_ok))
            n_ok = sum(1 for x in anc if x["ok"])
            ulist.append(dict(unit=uname, chain=chain, anchors=anc, n_ok=n_ok, n_anc=len(anc)))
        if not ulist:
            continue
        best = max(ulist, key=lambda x: (x["n_ok"], x["n_anc"], len(x["chain"]), -list(UNITS).index(x["unit"])))
        out[i] = dict(unit=best["unit"], anchored=best["n_ok"] >= HOMO_MIN_ANCHORS, anchors=best["anchors"],
                      chain=best["chain"], n_ok=best["n_ok"], units=ulist)
    return out


def series_exclusions(D, left_of: dict, homo: dict, member_ions: set) -> dict:
    """{i: {competitor name: reason}}: a competitor left on pair i is excluded
    when i sits in an ANCHORED series (unit u, anchors at spacings k) and the
    competitor's formula + k u, on the competitor's own adduct, is no member ion
    at ANY of the anchors' spacings. Isotope-line competitors and competitors
    without a neutral are never series-testable."""
    out = {}
    for i, comps in left_of.items():
        h = homo.get(i)
        if not h or not h["anchored"] or not comps:
            continue
        u = UNITS[h["unit"]]
        ks = sorted({x["k"] for x in h["anchors"] if x["ok"]})
        ex = {}
        for c in comps:
            if c["kind"] == "isoline" or not c.get("neutral"):
                continue
            cn = fcounts(c["neutral"])
            if not cn:
                continue
            member = None
            for k in ks:
                c2 = fadd(cn, u, k)
                if c2 is None:
                    continue
                ik = ion_key(C.format_formula(c2), c["adduct"])
                if ik is not None and ik in member_ions:
                    member = (k, C.format_formula(c2))
                    break
            if member is None:
                ex[c["name"]] = (f"series: {h['unit']} anchors at {','.join(kfmt(k, h['unit']) for k in ks)}; "
                                 f"{c['neutral']} {c['adduct']} has no member at those spacings")
        if ex:
            out[i] = ex
    return out


# ---------------------------------------------------------------------------
# other-source partners
# ---------------------------------------------------------------------------
PARTNER_LEVELS = frozenset({"3a", "3b", "3c", "3d"})   # an in-pass route / ladder / list level makes a partner


def partner_how(level, why) -> str:
    """What made a partner (its in-pass why): 'two routes', 'ladder' or
    'listed' -- never the in-pass level name."""
    w = str(why or "")
    if w == "routes":
        return "two routes"
    if w in ("ladder", "anchored homologue"):
        return "ladder"
    if w.startswith("split pinned"):
        return "listed"
    return {"3b": "two routes", "3d": "ladder", "3c": "listed"}.get(str(level), "listed")


ANCHOR_NONE = "none"


def anchor_kind(level, why) -> str:
    """What anchored a pair in the internal pass, in the words a partner prints
    (`partner_how`): 'two routes', 'ladder', 'listed', or 'none' (the in-pass
    level is not a route / ladder / list anchor)."""
    return partner_how(level, why) if str(level) in PARTNER_LEVELS else ANCHOR_NONE


def anchor_why(level, why) -> str:
    """The internal pass's why without its level token: an anchor's kind (+
    'listed' when a context list also names it: 'two routes + listed'), the
    list anchor's 'split pinned + listed', else the pass's own reason ('split
    open', 'competitors left (2)', 'rejected: ...')."""
    lv, w = str(level), str(why or "")
    if lv not in PARTNER_LEVELS:
        return w
    kind = partner_how(lv, w)
    if kind == "listed":
        return w or kind
    return kind + (" + listed" if lv == "3a" else "")


def partners_from(levels: pd.DataFrame, label: str, *, level_col: str = "inpass_level", why_col: str = "inpass_why",
                  ion_only_col: str = "ion_only_reading", locked: bool = True) -> dict:
    """{neutral: {route class: ['<label> <neutral> <adduct> <how>']}} of a
    source's pairs whose in-pass level is a route / ladder / list level (not
    ion-only, on a route class). ``locked``: drop the side-channel route classes
    (carbonate / ammonium / sodium) while their channel is locked."""
    from peaky.assignment.levels import split as SPL
    out = defaultdict(lambda: defaultdict(list))
    io = levels[ion_only_col].map(_b) if ion_only_col in levels.columns else pd.Series(False, index=levels.index)
    whys = levels[why_col] if why_col in levels.columns else pd.Series("", index=levels.index)
    for r, isio, why in zip(levels.itertuples(index=False), io, whys):
        lvv = str(getattr(r, level_col))
        if lvv not in PARTNER_LEVELS or isio or str(r.adduct) in SP.ION_ONLY:
            continue
        cl = route_class(str(r.adduct))
        if not cl:
            continue
        if locked and cl in SIDE_CLASSES and SPL.is_locked(SIDE_CLASSES[cl]):
            continue
        out[str(r.neutral_formula)][cl].append(f"{label} {r.neutral_formula} {r.adduct} {partner_how(lvv, why)}")
    return out


def merge_partners(*ds) -> dict:
    """Merge partner dicts in order (a neutral's partner texts concatenated)."""
    out = defaultdict(lambda: defaultdict(list))
    for d in ds:
        for nn, cc in (d or {}).items():
            for cl, v in cc.items():
                out[nn][cl].extend(v)
    return out
