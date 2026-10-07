"""The isotope-line model of the evidence scale (Orbitrap class) and the
per-file evaluation of a candidate's lines.

`cand_lines` predicts a candidate ion's observable isotope lines (count-aware
fine structure merged within the width model's FWHM); `eval_candidate` probes
each testable line in every file the pair was seen in (detectability against the
file's height gate, the width-model position window, the ratio band, occupied /
shadowed / off-scan files). On those records: test (a) (`contradiction_a`: a
line absent / off its band), the matched elements (`matched_elements`: the
positive fact), test (c) (`carbon_test`: the observed carbon count), test (b)
(`q1_isotopes`: another candidate's observed line this one cannot explain) and
test (k) (`twin_test`: the labelled run's 15N twin), and the isotope-line
competitors (`isoline_competitors`: the peak is another reading's line).

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


# ---------------------------------------------------------------------------
# the verdicts on a candidate's line records: (a), matched elements, (c), (b), (k)
# ---------------------------------------------------------------------------
ABSENT_FRAC = 0.8             # (a)/(b)/(k): >= 80 % of the testable files
MATCH_FRAC = 0.5              # a line "matched in band": >= 50 % of its testable files
C_TOL_ABS, C_TOL_REL, C_TOL_SE = 1.5, 0.25, 3.0   # (c): |C_obs - nC| > max(1.5, 0.25 nC, 3 se) contradicts
B_FACTOR = 4.0                # (b): "nothing comparable" = the observed line > 4 x what J predicts there
B_MIN_RATIO = 0.04            # (b) probes only K's lines >= 0.04 x the anchor (the 14N reagent line excepted)
B_SHOULDER_FWHM, B_SHOULDER_X = 2.0, 3.0   # (b): a line within 2 FWHM of a peak > 3 x taller is a shoulder
B_LINE_GATE_X = 1.0           # (b): an observed line counts only at >= 1 x the file's height gate
B_SAME_LINE_FWHM = 1.0        # (b): J's lines within max(position window, 1 FWHM) of the observed line are "there"
D15N_TWIN = 0.99703           # (k): a 14N [M+NO3]- reading implies its 15N twin at +0.99703 Da
ISOLINE_LIST_X = 2.0          # a committed reading P's isotope line is a competitor of an M0 when, in >= 1 file,
                              # it predicts >= 1/2 of the observed height there
ISOLINE_REACH_FWHM = 3.0      # ... searched within max(position window, 3 FWHM) + ISOLINE_REACH_PAD of the peak
ISOLINE_REACH_PAD = 0.05


def contradiction_a(recs: list, nmin: int) -> list[str]:
    """(a): a line bad (absent / too high / too low) in >= ABSENT_FRAC of its
    >= ``nmin`` testable files. One text per refuting line."""
    out = []
    for r in recs:
        if r["n_test"] >= nmin and r["n_bad"] / r["n_test"] >= ABSENT_FRAC:
            L = r["L"]
            what = ("absent" if r["n_abs"] >= max(r["n_hi"], r["n_lo"]) else
                    "too high" if r["n_hi"] >= r["n_lo"] else "too low")
            out.append(f"{L['label']} ({L['ratio']:.3g}x) {what} in {r['n_bad']}/{r['n_test']} files")
    return out


def matched_elements(recs: list, nmin: int) -> tuple[set, list]:
    """(elements, labels) of the lines in band in >= MATCH_FRAC of their >=
    ``nmin`` testable files. The 14N reagent line is a route marker, not an
    element's isotope line: its label counts, its element does not."""
    els = set()
    labs = []
    for r in recs:
        if r["n_test"] >= nmin and r["n_ok"] / r["n_test"] >= MATCH_FRAC:
            if r["L"]["mode"] != "reagent14N":
                els.update(r["L"]["elements"])
            labs.append(r["L"]["label"])
    return els, labs


def carbon_test(ctx, pair_key: tuple, cand_counts: dict) -> tuple:
    """(c): the observed PEAK's carbon count (``ctx.c13[pair_key]``) against the
    candidate's nC: ('contradicts' | 'agrees' | None, text). Skipped when the
    candidate's +1 line is < IC.C_MIN_13C_SHARE 13C (a carbon-free candidate IS
    tested)."""
    cobs = ctx.c13.get(pair_key)
    if not cobs or not np.isfinite(cobs.get("c", np.nan)):
        return None, ""
    share, _ = IC.c13_share(cand_counts)
    nC = int(cand_counts.get("C", 0))
    if nC > 0 and share < IC.C_MIN_13C_SHARE:
        return None, ""
    tol = max(C_TOL_ABS, C_TOL_REL * nC, C_TOL_SE * (cobs.get("se") if np.isfinite(cobs.get("se", np.nan)) else 0.0))
    if abs(cobs["c"] - nC) > tol:
        return "contradicts", f"13C reads {cobs['c']:.1f} C for {nC} (tol {tol:.1f}, {cobs['src']})"
    return "agrees", f"13C reads {cobs['c']:.1f} C for {nC}"


def _b_uninformative(ctx, o: dict, p: tuple) -> bool:
    """A file that says nothing about K's line either way: K's parent below the
    file's height gate, the observed (in-band) line below B_LINE_GATE_X x the
    gate, or the line peak a shoulder (within B_SHOULDER_FWHM FWHM of a peak >
    B_SHOULDER_X x taller)."""
    if o["h"] < ctx.gates[o["sid"]]:
        return True
    j = p[5]
    if p[4] and p[1] < B_LINE_GATE_X * ctx.gates[o["sid"]]:
        return True
    if p[4] and j >= 0:
        fa = ctx.files[o["sid"]]
        if fa.shoulder(j, B_SHOULDER_FWHM * ctx.fwhm(fa.mz[j]), B_SHOULDER_X):
            return True
    return False


def _line_off(fa, o: dict, p: tuple, L: dict) -> float:
    """Signed offset (Da) of the observed line peak from the nearest point of the predicted span (0 inside)."""
    x = fa.mz[p[5]] - o["mz"]
    lo, hi = min(L["span"][0], L["d"]), max(L["span"][1], L["d"])
    return 0.0 if lo <= x <= hi else (x - lo if x < lo else x - hi)


def _line_is_neighbour(ctx, okfiles: list, pair: dict, L: dict) -> str:
    """The in-band line is a neighbouring compound, not K's line, when its
    POOLED position (median over the files) is closer to another reading's
    committed, unrefuted M0 than to the prediction, within the position window.
    Returns the reason or ''. (The reference's TOF offset branch is not ported:
    a TOF-class source is not assessed.)"""
    if not okfiles:
        return ""
    xs, preds = [], []
    for o, p in okfiles:
        fa = ctx.files[o["sid"]]
        off = _line_off(fa, o, p, L)
        xs.append(float(fa.mz[p[5]]))
        preds.append(float(fa.mz[p[5]]) - off)
    if not len(ctx.m0_mz):
        return ""
    x = float(np.median(xs))
    d_line = abs(x - float(np.median(preds)))
    tol = ctx.tol_da(x, float(np.median([p[1] for _o, p in okfiles])))
    skip = {pair.get("pairkey")} | set(pair.get("own_pk", ()))
    lo = np.searchsorted(ctx.m0_mz, x - tol, "left")
    hi = np.searchsorted(ctx.m0_mz, x + tol, "right")
    for i in range(lo, hi):
        if ctx.m0_pk[i] not in skip and abs(ctx.m0_mz[i] - x) < d_line:
            return f"another reading's M0 ({ctx.m0_pk[i]}) sits closer"
    return ""


def twin_test(ctx, pair: dict, cand: dict, nmin: int) -> str:
    """(k), labelled-nitrate run, competitors only: a reading that can only be
    a 14N [M+NO3]- cluster (every plausible decomposition over the run's
    decomposition adducts is [M+NO3]-) implies its 15N twin (+D15N_TWIN Da) at
    >= the file's twin ratio x its height (detectable at DET_X x the gate; an
    occupant counts as present). Refuted when absent or too low in >=
    ABSENT_FRAC of >= ``nmin`` files. Returns the text or ''."""
    if not ctx.labelled or not ctx.twin_q or not cand.get("counts"):
        return ""
    if cand["adduct"] != "[M+NO3]-":
        return ""
    for d in ctx.decompositions(cand["counts"], cand["neutral"], cand["adduct"]):
        if d["ok"] and d["neutral"] and d["adduct"] != "[M+NO3]-":
            return ""
    n_test = n_bad = 0
    for o in pair["obs"]:
        q = ctx.twin_q.get(o["sid"])
        if not q:
            continue
        fa = ctx.files[o["sid"]]
        exp_h = o["h"] * q
        if exp_h < DET_X * ctx.gates[o["sid"]]:
            continue
        t = o["mz"] + D15N_TWIN
        if not fa.in_scan(t, t):
            continue
        st, hobs, _j = fa.probe((t, t), ctx.tol_da(o["mz"], exp_h), o["pid"], SHADOW_FWHM * ctx.fwhm(o["mz"]), exp_h)
        if st == "shadowed":
            continue
        n_test += 1
        if not (st in ("free", "occupied") and hobs >= BAND[0] * exp_h):
            n_bad += 1
    if n_test >= nmin and n_bad / n_test >= ABSENT_FRAC:
        return f"15N twin (+0.997, >= q_lo x) absent or too low in {n_bad}/{n_test} files"
    return ""


def q1_isotopes(ctx, pair: dict, cands: list, nmin: int) -> list:
    """The isotope tests of the committed reading (``cands[0]``) and every
    competitor, in place: ``recs`` (eval_candidate), ``a``, ``c`` / ``c_agree``,
    ``matched_els`` / ``matched_labels``, ``b`` (a candidate K's observed
    in-band line where J predicts nothing comparable; K must not be refuted by
    its own (a)/(c); the committed reading too can be refuted through a
    competitor's line), ``k`` (competitors only) and ``contradicted`` = a or b
    or c or k."""
    obs = pair["obs"]
    pk = pair["pairkey"]
    for c in cands:
        c["recs"] = eval_candidate(ctx, c, obs, pk, pair=pair)
        c["a"] = contradiction_a(c["recs"], nmin)
        c["b"] = []
        cv, ctext = carbon_test(ctx, pair["key"], c["counts"])
        c["c"] = ctext if cv == "contradicts" else ""
        c["c_agree"] = ctext if cv == "agrees" else ""
        c["matched_els"], c["matched_labels"] = matched_elements(c["recs"], nmin)
    for K in cands:
        if K["a"] or K["c"]:
            continue              # a reading its own (a)/(c) refutes does not vouch for a line
        for r in K["recs"]:
            L = r["L"]
            if L["ratio"] < B_MIN_RATIO and L["mode"] != "reagent14N":
                continue
            # an OBSERVED in-band line counts wherever it is seen, also below K's predicted detection
            consid = [(o, p) for o, p in zip(obs, r["per"]) if (p[3] and p[0] not in ("shadowed", "offscan")) or p[4]]
            if L["mode"] != "reagent14N":
                consid = [(o, p) for o, p in consid if not _b_uninformative(ctx, o, p)]
                okfiles = [(o, p) for o, p in consid if p[4]]
                if _line_is_neighbour(ctx, okfiles, pair, L):
                    continue
            else:
                okfiles = [(o, p) for o, p in consid if p[4]]
            if len(okfiles) < nmin or len(okfiles) < ABSENT_FRAC * len(consid):
                continue
            eff = min([ctx.eff.get(e, 1.0) for e in L["elements"]] or [1.0])
            for J in cands:
                if J is K:
                    continue
                if L["mode"] == "reagent14N" and any(z["mode"] == "reagent14N" for z in J["lines"]):
                    continue      # J carries the labelled atom too: it makes the 14N line (no upper bound)
                n_over = 0
                rJs = []
                for o, p in okfiles:
                    x = ctx.files[o["sid"]].mz[p[5]] - o["mz"]          # the observed line, from the anchor
                    tol = max(ctx.tol_da(o["mz"], p[1]), B_SAME_LINE_FWHM * ctx.fwhm(o["mz"]))
                    rJ = sum(z["ratio"] for z in J["lines"]
                             if min(z["span"][0], z["d"]) - tol <= x <= max(z["span"][1], z["d"]) + tol)
                    rJs.append(rJ)
                    if p[1] > B_FACTOR * o["h"] * rJ * eff:
                        n_over += 1
                if n_over / len(okfiles) >= ABSENT_FRAC and n_over >= nmin:
                    J["b"].append(f"{L['label']} line at {np.median([p[1] / o['h'] for o, p in okfiles]):.3g}x seen in "
                                  f"{n_over}/{len(okfiles)} files (in band for {K['name']}), predicts {np.median(rJs):.2g}x")
    for c in cands:
        c["k"] = twin_test(ctx, pair, c, nmin) if c is not cands[0] else ""
    for c in cands:
        c["contradicted"] = bool(c["a"] or c["b"] or c["c"] or c["k"])
    return cands


# ---------------------------------------------------------------------------
# isotope-line competitors
# ---------------------------------------------------------------------------
def isoline_competitors(ctx, pair: dict, nmin: int) -> tuple[list, set]:
    """'This peak is line L of committed reading P': listed when, in >= 1 file
    of the pair, a line of a committed, unrefuted reading P of ANOTHER neutral
    lands on the observed peak (P's position window) and predicts >= 1 /
    ISOLINE_LIST_X of its height. Tested in every file by P's predicted line
    there (P's own M0 in that file, else the tallest peak at P's position;
    none -> 0): EXCLUDED when the peak is > BAND[1] x the predicted line in >=
    ABSENT_FRAC of >= ``nmin`` files. Returns (competitors, the sids where some
    P line explains >= half the peak)."""
    n, _a = pair["key"]
    obs = pair["obs"]
    listed = {}
    for o in obs:
        ix = ctx.isolines.get(o["sid"])
        if not ix or not len(ix["c"]):
            continue
        reach = max(ctx.tol_da(o["mz"], 0.0), ISOLINE_REACH_FWHM * ctx.fwhm(o["mz"])) + ISOLINE_REACH_PAD
        lo = np.searchsorted(ix["c"], o["mz"] - reach, "left")
        hi = np.searchsorted(ix["c"], o["mz"] + reach, "right")
        for i in range(lo, hi):
            pk = ix["pk"][i]
            if pk == pair["pairkey"] or pk in pair.get("own_pk", ()) or ix["neutral"][i] == n:
                continue
            pmz, ph = ix["pos"][pk]
            exp_h = ph * ix["r"][i]
            tol = ctx.tol_da(o["mz"], exp_h)
            if not (pmz + ix["lo"][i] - tol <= o["mz"] <= pmz + ix["hi"][i] + tol):
                continue
            if exp_h * ISOLINE_LIST_X < o["h"]:
                continue
            key = (pk, ix["label"][i])
            if key not in listed:
                listed[key] = dict(pk=pk, label=ix["label"][i], ratio=float(ix["r"][i]), lo=float(ix["lo"][i]),
                                   hi=float(ix["hi"][i]), pmz=pmz, ppm=(o["mz"] - float(ix["c"][i])) / o["mz"] * 1e6)
    comps, explained = [], set()
    for (pk, lab), v in listed.items():
        n_ex = n_exp = 0
        ratios = []
        for o in obs:
            fa = ctx.files[o["sid"]]
            ix = ctx.isolines.get(o["sid"]) or {}
            pos = (ix.get("pos") or {}).get(pk)
            if pos is None:
                lo_, hi_ = fa.window(v["pmz"], ctx.tol_da(v["pmz"], 0.0))
                if hi_ > lo_:
                    jj = lo_ + int(np.argmax(fa.h[lo_:hi_]))
                    pos = (float(fa.mz[jj]), float(fa.h[jj]))
            exp_h = 0.0
            if pos is not None:
                e = pos[1] * v["ratio"]
                tol = ctx.tol_da(o["mz"], e)
                if pos[0] + v["lo"] - tol <= o["mz"] <= pos[0] + v["hi"] + tol:
                    exp_h = e
            ratios.append(exp_h / o["h"] if o["h"] > 0 else 0.0)
            if o["h"] > BAND[1] * exp_h:
                n_ex += 1
            else:
                n_exp += 1
                explained.add(o["sid"])
        excl = len(obs) >= nmin and n_ex >= nmin and n_ex / len(obs) >= ABSENT_FRAC
        pn, pa = pk.split("|", 1)
        comps.append(dict(name=f"{lab} line of {pn} {pa}", neutral=pn, adduct=pa, kind="isoline", ppm=v["ppm"],
                          excluded=excl, explained_files=n_exp, n_files=len(obs),
                          why=(f"isotope height: the peak is > {BAND[1]:g}x {pn} {pa}'s predicted {lab} line in "
                               f"{n_ex}/{len(obs)} files" if excl else
                               f"{lab} line of {pn} {pa} (pred/obs median {np.median(ratios):.2f}) explains >= half the "
                               f"peak in {n_exp}/{len(obs)} files; > {BAND[1]:g}x in {n_ex}")))
    return comps, explained
