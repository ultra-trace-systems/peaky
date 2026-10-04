"""The isotope-line model of the evidence scale (Orbitrap class) and the
per-file evaluation of a candidate's lines.

`cand_lines` predicts a candidate ion's observable isotope lines (count-aware
fine structure merged within the width model's FWHM); `eval_candidate` probes
each testable line in every file the pair was seen in (detectability against the
file's height gate, the width-model position window, the ratio band, occupied /
shadowed / off-scan files). The verdicts built on these records -- tests (a),
(b), (c), (k), the matched elements, isoline competitors -- sit on top of them.

Only the Orbitrap branches exist: a TOF-class source is not assessed.
"""
from __future__ import annotations

import numpy as np

from peaky.batch import iso_checks as IC
from peaky.chem import isotopes as ISO

DET_X = 3.0                   # detectable: predicted height >= 3 x the file's height gate
DET_X_MINOR = 5.0             # composite lines (>= 2 substitutions) and the light 15N / 18O lines need 5 x the gate
DET_SINGLE_ISO = frozenset({"13C", "37Cl", "81Br", "34S", "29Si", "30Si"})   # single substitutions that keep DET_X
BAND = (0.5, 2.0)             # a line is in band at 0.5 - 2 x its predicted height
SPAN_SHARE = 0.1              # a merged line spans its components carrying >= 10 % of it
SHADOW_FWHM = 1.5             # an absent line with a peak >= SHADOW_FRAC x its expected height within
SHADOW_FRAC = 0.5             # SHADOW_FWHM x FWHM is "shadowed" (untestable), not absent
MAJOR_RATIO = 0.03            # a line >= 0.03 x the anchor may test by its height; a minor line by its absence only
ALLOWED_ISO = frozenset({"13C", "15N", "18O", "34S", "37Cl", "81Br", "29Si", "30Si"})   # testable added isotopes
HEAVY_ISO = frozenset({"37Cl", "81Br", "34S", "29Si", "30Si"})
ISO_EL = {"13C": "C", "2H": "H", "15N": "N", "17O": "O", "18O": "O", "33S": "S", "34S": "S", "37Cl": "Cl",
          "81Br": "Br", "29Si": "Si", "30Si": "Si"}
MIN_LINE_RATIO = 1e-5         # a predicted line below 1e-5 x the anchor is not a line
MAX_NOMINAL = 6               # a testable line lies 1 - 6 nominal Da from the anchor
MAX_D13C = 2                  # ... and adds at most two 13C
CL37_FIX = True               # a 37Cl-bearing line is tested at its exact position (probe_exact_37cl)
CL37_NEIGHBOUR_FWHM = 1.0     # a peak within 1 FWHM of a 37Cl line is not resolved from it
REAGENT14N_LABEL = "14N (reagent impurity)"

_LINES: dict = {}
_LINES_MAX = 400000


def tag_text(tags: dict) -> str:
    """'13C', '2x13C', '13C+18O' ...; 'all-light' for none."""
    if not tags:
        return "all-light"
    return "+".join(f"{k}" if v == 1 else f"{v}x{k}" for k, v in sorted(tags.items()))


def cand_lines(counts: dict, fwhm: float, anchor_shift: float, purity: float) -> list[dict]:
    """The candidate ion's observable isotope lines relative to the line the
    observed peak represents (its mono line for a competitor; for the committed
    reading the line at ``anchor_shift`` Da from the mono line). One dict per
    line: d (Da from the anchor centroid), dpure (its main component), span (the
    components >= SPAN_SHARE of the line), ratio (merged, to the anchor),
    ratio_main (main component), label, elements, testable, mode ('full' |
    'reagent14N'), nominal, heavy, det_x. An ion with ^N adds the labelled
    reagent's 14N impurity line at n15 (1 - purity) / purity. Memoised."""
    ck = tuple(sorted((k, int(v)) for k, v in counts.items() if v))
    key = (ck, round(float(fwhm), 6), round(float(anchor_shift), 2), float(purity))
    hit = _LINES.get(key)
    if hit is not None:
        return hit
    cc = {k: int(v) for k, v in counts.items() if k in IC._ISO and v > 0}
    fs = IC.fine_structure(cc)
    fs, cen, rel = IC._lines(fs, fwhm)
    an = int(np.argmin(np.abs(cen - anchor_shift)))
    a_rows = fs[fs["line"] == an]
    a_tags = dict(a_rows.loc[a_rows["rel"].idxmax(), "tags"])
    has_m2_owner = any(counts.get(e, 0) for e in ("Br", "Cl"))
    out = []
    for li in range(len(cen)):
        if li == an:
            continue
        g = fs[fs["line"] == li]
        main = g.loc[g["rel"].idxmax()]
        tags = dict(main["tags"])
        ratio = float(rel[li] / rel[an])
        if ratio < MIN_LINE_RATIO:
            continue
        d = float(cen[li] - cen[an])
        diff = {k: tags.get(k, 0) - a_tags.get(k, 0) for k in set(tags) | set(a_tags)}
        diff = {k: v for k, v in diff.items() if v}
        isos = set(diff)
        els = sorted({ISO_EL[i] for i in isos if i in ISO_EL})
        heavy = float(g.loc[[any(t in HEAVY_ISO for t in tg) for tg in g["tags"]], "rel"].sum() / rel[li])
        nominal = int(round(d))
        testable = bool(isos) and isos <= ALLOWED_ISO and abs(diff.get("13C", 0)) <= MAX_D13C \
            and not ("18O" in isos and has_m2_owner) and 0 < abs(nominal) <= MAX_NOMINAL
        sig = g[g["rel"] >= SPAN_SHARE * rel[li]]
        span = (float(sig["shift"].min() - cen[an]), float(sig["shift"].max() - cen[an]))
        ratio_main = float(main["rel"] / rel[an])
        single = sum(abs(v) for v in diff.values()) == 1 and isos <= DET_SINGLE_ISO
        out.append(dict(d=d, dpure=float(main["shift"] - cen[an]), span=span, ratio_main=ratio_main,
                        ratio=ratio, label=tag_text(diff) if diff else tag_text(tags),
                        elements=els, testable=bool(testable), mode="full", nominal=nominal, heavy=heavy,
                        det_x=DET_X if single else DET_X_MINOR))
    n15 = int(counts.get("^N", 0))
    if n15 > 0:
        p = float(purity)
        d14 = float(ISO.ISOTOPE_SPACING["14N"])
        r14 = n15 * (1 - p) / p
        out.append(dict(d=d14, dpure=d14, span=(d14, d14), ratio_main=r14, ratio=r14, label=REAGENT14N_LABEL,
                        elements=["^N"], testable=True, mode="reagent14N", nominal=-1, heavy=0.0, det_x=DET_X))
    _LINES[key] = out
    if len(_LINES) > _LINES_MAX:
        _LINES.clear()
    return out


def labelled_line_at(ctx, sid: str, j: int):
    """The m/z of the LABELLED-REAGENT line peak ``j`` of file ``sid`` sits on,
    else None: the peak is the M0 of a committed, unrefuted reading whose ion
    carries ^N, or it lies within the position window (by the peak's height) of
    a predicted isotope line of such a reading."""
    fa = ctx.files[sid]
    if not ctx.labelled:
        return None
    if fa.role[j] == "M0" and "^" in fa.pk[j] and fa.pk[j] not in fa.refuted:
        return float(fa.mz[j])
    ix = ctx.isolines.get(sid) or {}
    cs = ix.get("c", np.array([]))
    if not len(cs):
        return None
    x, h = float(fa.mz[j]), float(fa.h[j])
    tol = ctx.tol_da(x, h)
    a = np.searchsorted(cs, x - tol - 0.01, "left")
    b = np.searchsorted(cs, x + tol + 0.01, "right")
    for i in range(a, b):
        pk = ix["pk"][i]
        if "^" not in pk:
            continue
        pmz = ix["pos"][pk][0]
        if pmz + ix["lo"][i] - tol <= x <= pmz + ix["hi"][i] + tol:
            return float(cs[i])
    return None


def probe_exact_37cl(ctx, sid: str, span, tol: float, anchor_pid: str, own=(), h_exp: float = 0.0):
    """A 37Cl-bearing line tested at its exact position, the resolution taken
    into account: the tallest peak within the position window ``tol`` of the
    span (no 25 ppm neighbour reach); a peak at a labelled-reagent position more
    than CL37_NEIGHBOUR_FWHM from the span is resolved from it and ignored;
    nothing in the window but a peak >= SHADOW_FRAC x the expected height within
    1 FWHM -> 'shadowed'. Returns FileArr.probe's (status, height, index)."""
    fa = ctx.files[sid]
    fw = CL37_NEIGHBOUR_FWHM * ctx.fwhm(0.5 * (span[0] + span[1]))
    reach = max(tol, fw)
    lo = np.searchsorted(fa.mz, span[0] - reach, "left")
    hi = np.searchsorted(fa.mz, span[1] + reach, "right")

    def dist(x):
        return span[0] - x if x < span[0] else (x - span[1] if x > span[1] else 0.0)

    keep = []
    blend = False
    for j in range(lo, hi):
        pos = labelled_line_at(ctx, sid, j)
        if pos is not None and dist(pos) > fw:
            ctx.cl37_flips += 1          # a resolved labelled-reagent line: not the 37Cl line
            continue
        if dist(float(fa.mz[j])) <= tol:
            keep.append(j)
        elif float(fa.h[j]) >= SHADOW_FRAC * h_exp:
            blend = True                 # an unresolved neighbour within 1 FWHM
    if not keep:
        if blend:
            ctx.cl37_blend += 1
            return "shadowed", 0.0, -1
        return "absent", 0.0, -1
    best = max(keep, key=lambda j: fa.h[j])
    r = fa.role[best]
    if r == "M0" and (fa.pk[best] in fa.refuted or fa.pk[best] in own):
        return "free", float(fa.h[best]), best
    if r in ("M0", "reagent", "artifact") or (r == "iso_child" and fa.parent[best] != anchor_pid):
        return "occupied", float(fa.h[best]), best
    return "free", float(fa.h[best]), best


def eval_candidate(ctx, cand: dict, obs: list, pairkey: str, collect=False, pair=None) -> list[dict]:
    """Every testable line of one candidate over the pair's files (``obs``: per
    file dicts sid, mz, h, area, pid). One record per line: L, n_det, n_test,
    n_ok, n_bad, n_abs, n_hi, n_lo, n_occ and ``per`` = one (status, h_obs,
    exp_h, detectable, ok, peak index) per file.

    A line is detectable at L['det_x'] x the gate; an OCCUPIED line whose
    occupant is in band counts as present (tested, neither bad nor matched); an
    occupant above the band leaves the file untestable, one below it is too low
    (a major line only); a too-high free line counts only on an iso child of
    the anchor; a minor line tests by its absence only. ``pairkey`` and
    ``collect`` are kept for the reference's signature (unused)."""
    recs = []
    pair = pair or {}
    lines = cand["lines"]
    for L in lines:
        if not L["testable"]:
            continue
        n_det = n_test = n_ok = n_hi = n_lo = n_abs = n_occ = 0
        per = []
        det_x = L.get("det_x", DET_X)
        for o in obs:
            fa = ctx.files[o["sid"]]
            eff = min([ctx.eff.get(e, 1.0) for e in L["elements"]] or [1.0])
            exp_h = o["h"] * L["ratio"] * eff
            det = exp_h >= det_x * ctx.gates[o["sid"]]
            tol = ctx.tol_da(o["mz"], exp_h)
            span = (o["mz"] + min(L["span"][0], L["d"]), o["mz"] + max(L["span"][1], L["d"]))
            if not fa.in_scan(span[0] - tol, span[1] + tol):
                per.append(("offscan", 0.0, exp_h, False, False, -1))    # outside the scan: untestable
                continue
            if CL37_FIX and "37Cl" in L["label"]:
                st, hobs, j = probe_exact_37cl(ctx, o["sid"], span, tol, o["pid"], pair.get("own_pk", ()), exp_h)
            else:
                st, hobs, j = fa.probe(span, tol, o["pid"], SHADOW_FWHM * ctx.fwhm(o["mz"]), exp_h,
                                       pair.get("own_pk", ()))
            # in band against the merged line OR its main component alone (a partly resolved blend)
            exp_lo = o["h"] * L["ratio_main"] * eff
            if L["mode"] == "reagent14N":
                # the labelled reagent's 14N line: at least half the impurity level; a 14N reagent ion in the
                # source adds to it (no upper bound); only the engine's own twin / child counts
                ok = st == "free" and exp_h > 0 and hobs >= BAND[0] * exp_lo and fa.role[j] in ("iso_child", "M0")
                occ_ok = st == "occupied" and exp_h > 0 and hobs >= BAND[0] * exp_lo
                per.append((st, hobs, exp_h, det, ok, j))
                if det and st != "shadowed":
                    n_det += 1
                    n_test += 1
                    if ok:
                        n_ok += 1
                    elif occ_ok:
                        n_occ += 1
                    elif st == "absent":
                        n_abs += 1
                    else:
                        n_lo += 1
                continue
            ok = (st == "free" and exp_h > 0 and BAND[0] * exp_lo <= hobs <= BAND[1] * exp_h)
            # a too-high line counts against the reading only where the engine itself hung that peak on this
            # parent (an iso child of the anchor); an unexplained peak may carry a coincident compound
            major = L["ratio"] >= MAJOR_RATIO
            high_ok = st == "free" and exp_h > 0 and hobs / exp_h > BAND[1] and fa.role[j] == "iso_child" \
                and L["mode"] == "full" and major
            per.append((st, hobs, exp_h, det, ok, j))
            if not det:
                continue
            n_det += 1
            if st == "shadowed":
                continue
            if st == "occupied":
                if hobs > BAND[1] * exp_h:
                    continue      # an occupant above the band: another reading's peak can only ADD height
                if hobs >= BAND[0] * exp_lo:
                    n_test += 1   # an occupant in band: the line is there, not bad, not matched
                    n_occ += 1
                    continue
                if not major:
                    continue      # a minor line tests by its absence only
                n_test += 1
                n_lo += 1         # a too-small occupant: the line is too low
                continue
            if st == "free" and not ok and hobs > BAND[1] * exp_h and not high_ok:
                continue          # too high on an unexplained peak (or a minor line): untestable
            if st == "free" and not ok and not major:
                continue          # a minor line tests by its absence only
            n_test += 1
            if ok:
                n_ok += 1
            elif st == "absent":
                n_abs += 1
            elif hobs > BAND[1] * exp_h:
                n_hi += 1
            else:
                n_lo += 1
        n_bad = n_abs + n_lo + n_hi
        recs.append(dict(L=L, n_det=n_det, n_test=n_test, n_ok=n_ok, n_bad=n_bad, n_abs=n_abs, n_hi=n_hi, n_lo=n_lo,
                         n_occ=n_occ, per=per))
    return recs
