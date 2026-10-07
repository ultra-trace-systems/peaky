"""Step 1 of the evidence scale (the ion formula): every pair's competitors
and their isotope evidence (pass A), and the per-pair fact columns and
competitor rows the decision reads (pass B).

Pass A (`q1_pass`), per pair:

- `build_candidates`: the committed reading and its competitors -- every ion
  the run's space proposes inside the calibrated window, probed at the pooled
  m/z in the run window AND at each file's m/z in that file's window (the
  union; a competitor's ppm is reported at the pooled m/z), plus the engine's
  alternatives (a TIED alternative always; an untied one inside the run's or
  its file's window). The committed ion itself and any reading with the same
  element counts (the same ion read another way) are not competitors. A
  committed neutral the space cannot produce through any channel is
  enumerated in the space widened to admit its class (D5); when no widening
  admits it, nothing can be enumerated (``enum_ok`` False).
- `lines.q1_isotopes` on the committed reading and every competitor (tests
  (a), (b), (c), (k); the matched elements), `lines.isoline_competitors`.

Pass B (`pass_b`): the competitor rows (excluded by isotopes, or left; the
isotope-line competitors appended) and the fact columns (committed reading
contradicted / matched, no_comp_info, the space notes, the committed reading's
plausibility as its own decomposition, its per-file window z ...).

The reference's joint (route) exclusion of competitors is NOT ported: the
scale counts a route-excluded competitor as left, so it never changed a level
(``n_excl_routes`` is always 0).
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd

from peaky.assignment.levels import context as CX
from peaky.assignment.levels import lines as LN
from peaky.assignment.levels import space as SP
from peaky.assignment.levels.context import FileArr, pairs_from_files, parse_alts  # noqa: F401 -- re-exported
from peaky.chem import chemistry as C

K_SIGMA = CX.K_SIGMA          # the calibrated half-window, in sigma
ANCHOR_SHIFT_MIN_DA = 0.3     # the committed peak is not the mono line when it sits >= 0.3 Da from it
NO_KEY_ORDER = 9              # competitors without a finite ppm sort after |ppm - mu| (the reference's literal)
STAY_PARTS = 3                # "why it stays": the first three line records
OFFWIN_Z = 3.0                # the committed reading's own median per-file |z| > 3: "off its own window"
OFFWIN_E_MAX_PPM = 50.0       # a per-file error beyond 50 ppm from the mono ion is read from the row's ppm_error
NO_ENUM_NOTE = "committed neutral outside the enumerable space (no widening admits it)"
NO_ION_OUTSIDE = "its adduct has no ion composition"


def _ev():
    from peaky.assignment import evidence as EV
    return EV


# ---------------------------------------------------------------------------
# pass A
# ---------------------------------------------------------------------------
def build_candidates(ctx, pair: dict, want_alts: bool = True):
    """The committed reading (``cands[0]``) and its competitors, each with its
    predicted lines. Returns (cands, n engine alternatives added, the committed
    neutral's space reason (``outside``), n ties with the same ion)."""
    n, a = pair["key"]
    sp = ctx.space
    mz = pair["mz"]
    counts0 = SP.ion_counts_of(n, a, pair.get("ion"))
    try:
        mono = C.ion_mz(n, a)
        ashift = mz - mono
    except Exception:  # noqa: BLE001
        ashift = 0.0
    if abs(ashift) < ANCHOR_SHIFT_MIN_DA:
        ashift = 0.0
    fw = ctx.fwhm(mz)
    ion_key0 = sp.canon(n, a)
    if ion_key0 is None and counts0:
        # an adduct the canonical key cannot read (a decoy's wrong adduct): the key from the ion counts, ^N folded
        fold = Counter()
        for el, v in counts0.items():
            fold["N" if el == "^N" else el] += v
        ion_key0 = "".join(f"{el}{fold[el]}" for el in sorted(fold))
    committed = dict(name=f"{n} {a}", neutral=n, adduct=a, counts=counts0, kind="committed",
                     ion_key=ion_key0, ppm=pair.get("ppm", np.nan),
                     lines=ctx.lines(counts0, fw, ashift) if counts0 else [])
    mu, sg = ctx.run_window
    lo, hi = mu - K_SIGMA * sg, mu + K_SIGMA * sg
    chans = list(ctx.channels) + ([a] if a in C.ADDUCT_SHIFTS and a not in ctx.channels else [])
    # D5: a committed ion the space cannot produce through ANY channel would read 'unique by mass' vacuously (the
    # filter that dropped it also drops its same-class competitors): enumerate in the space widened to admit it
    relax_note = ""
    reachable = (a in chans and a in C.ADDUCT_SHIFTS and sp.space_reason(n) is None) or any(
        d["ok"] for d in ctx.decompositions(counts0, n, a, adducts=[x for x in chans if x not in SP.ION_ONLY] + [a]))
    if ion_key0 is not None and counts0 and not reachable:
        rs = sp.relaxed(n)
        if rs is not None:
            sp_enum, relax_note = rs, rs.relaxed_note
        else:
            sp_enum, relax_note = None, NO_ENUM_NOTE
    else:
        sp_enum = sp
    # the pooled m/z in the run window, plus every file's m/z in that file's window; the competitors are the union
    found = {}
    win_src = {}
    if sp_enum is not None:
        probes = [(mz, lo, hi, "run")] if np.isfinite(mz) else []
        for o in pair.get("obs", []):
            mu_f, sg_f = ctx.windows.get(o["sid"], ctx.run_window)
            probes.append((o["mz"], mu_f - K_SIGMA * sg_f, mu_f + K_SIGMA * sg_f, o["sid"]))
        for pmz, plo, phi, src in probes:
            for ion, v in sp_enum.enumerate(pmz, plo, phi, chans).items():
                win_src.setdefault(ion, []).append(src)
                if ion not in found:
                    try:          # the competitor's error at the POOLED m/z (comparable across files)
                        th = C.ion_mz(v[0], v[1])
                        found[ion] = (v[0], v[1], (mz - th) / th * 1e6)
                    except Exception:  # noqa: BLE001
                        found[ion] = v
    comps = {}
    for ion, (cn, ca, ppm) in found.items():
        if ion == committed["ion_key"]:
            continue
        w = win_src.get(ion, [])
        comps[ion] = dict(neutral=cn, adduct=ca, ppm=ppm, src="mass", win=("run" if "run" in w else "") + (
            f"{'+' if 'run' in w else ''}{sum(1 for x in w if x != 'run')} file windows"
            if any(x != "run" for x in w) else ""))
    n_alt_added = 0
    n_tie_same = 0
    if want_alts:
        for item in pair["alts"]:
            alt, tied = item[0], item[1]
            asid = item[2] if len(item) > 2 else None
            ion = sp.canon(alt["formula"], alt["adduct"])
            if ion is None or ion == committed["ion_key"]:
                if tied and ion is not None:
                    n_tie_same += 1   # a tie with the SAME ion is a split tie, not a competitor
                continue
            ppm = float(alt.get("ppm", np.nan))
            mu_f, sg_f = ctx.windows.get(asid, ctx.run_window) if asid else ctx.run_window
            inwin = np.isfinite(ppm) and ((lo <= ppm <= hi) or (mu_f - K_SIGMA * sg_f <= ppm <= mu_f + K_SIGMA * sg_f))
            if ion in comps:
                if tied:
                    comps[ion]["src"] += "+tie"
                continue
            if tied or inwin:
                comps[ion] = dict(neutral=str(alt["formula"]), adduct=str(alt["adduct"]), ppm=ppm,
                                  src="tie" if tied else "alt")
                n_alt_added += 1
    cands = [committed]
    for ion, v in sorted(comps.items(),
                         key=lambda kv: abs(kv[1]["ppm"] - mu) if np.isfinite(kv[1]["ppm"]) else NO_KEY_ORDER):
        cnt = SP.ion_counts_of(v["neutral"], v["adduct"])
        if cnt and cnt == counts0:
            # the SAME ion read another way (a labelled atom the canonical key folds): a split question
            if "tie" in v["src"]:
                n_tie_same += 1
            continue
        cands.append(dict(name=f"{v['neutral']} {v['adduct']}", neutral=v["neutral"], adduct=v["adduct"], counts=cnt,
                          kind=v["src"], ion_key=ion, ppm=v["ppm"], win=v.get("win", ""),
                          lines=ctx.lines(cnt, fw, 0.0) if cnt else []))
    outside = sp.space_reason(n) if committed["ion_key"] is not None else NO_ION_OUTSIDE
    committed["relax_note"] = relax_note
    committed["enum_ok"] = sp_enum is not None
    return cands, n_alt_added, outside, n_tie_same


def slim_q1(c: dict, nmin: int) -> dict:
    """A candidate's pass-A facts without its line / per-file records:
    ``has_testable`` and ``stay`` (why it stays: the first STAY_PARTS lines'
    in-band counts, or why they are untestable)."""
    out = {k: v for k, v in c.items() if k not in ("lines", "recs")}
    out["has_testable"] = bool(c.get("lines")) and any(L["testable"] for L in c["lines"])
    parts = []
    for r in c.get("recs", []):
        L = r["L"]
        if r["n_test"] < nmin:
            parts.append(f"{L['label']} {L['ratio']:.2g}x testable in {r['n_test']} files")
        else:
            parts.append(f"{L['label']} {L['ratio']:.2g}x ok in {r['n_ok']}+{r.get('n_occ', 0)}occ/{r['n_test']}")
    out["stay"] = "; ".join(parts[:STAY_PARTS]) if out["has_testable"] else "no testable isotope line"
    return out


def q1_pass(ctx, pairs: dict, nmin: int, *, progress=None) -> dict:
    """Pass A over every pair: {(n, a): dict(cands = [slim candidates,
    committed first], n_alt, outside, n_tie_same, isolines, iso_explained =
    sorted sids)}. A pair seen in no file gets no isotope tests and no isoline
    competitors. ``progress(i, n, seconds)`` is called every 500 pairs."""
    t0 = time.time()
    res = {}
    for i, (k, pr) in enumerate(pairs.items()):
        cands, n_alt, outside, n_tie_same = build_candidates(ctx, pr)
        if pr["obs"]:
            LN.q1_isotopes(ctx, pr, cands, nmin)
        else:
            for c in cands:
                c.update(recs=[], a=[], b=[], c="", c_agree="", k="", matched_els=set(), matched_labels=[],
                         contradicted=False)
        isol, iso_expl = LN.isoline_competitors(ctx, pr, nmin) if pr["obs"] else ([], set())
        res[k] = dict(cands=[slim_q1(c, nmin) for c in cands], n_alt=n_alt, outside=outside, n_tie_same=n_tie_same,
                      isolines=isol, iso_explained=sorted(iso_expl))
        if progress is not None and (i + 1) % 500 == 0:
            progress(i + 1, len(pairs), time.time() - t0)
    return res


def skip_m0_of(q1: dict) -> frozenset:
    """{(sid, neutral, adduct)}: the M0s some committed reading's isotope line explains (no route from them)."""
    return frozenset((sid, k[0], k[1]) for k, v in q1.items() for sid in v.get("iso_explained", ()))


# ---------------------------------------------------------------------------
# pass B
# ---------------------------------------------------------------------------
COMP_COLUMNS = ("neutral_formula", "adduct", "competitor", "kind", "ppm", "status", "how", "why", "window",
                "comp_neutral", "comp_adduct")
#: pass B's per-pair columns (in order; a source with no pair gives an empty frame with them)
PASS_B_COLUMNS = ("neutral_formula", "adduct", "n_files_obs", "committed_contradicted", "committed_reasons",
                  "committed_matched", "committed_matched_lines", "committed_c13", "strong_isotope", "n_competitors",
                  "n_comp_mass", "n_comp_mass_engine", "n_comp_alt", "n_excl_iso", "n_excl_routes", "n_left",
                  "competitors_left", "competitors_excluded", "n_decomp_plausible", "committed_plausible",
                  "plaus_decomps", "outside_space", "space_note", "n_tie_same", "committed_z_med",
                  "committed_files_off_window", "off_own_window", "n_comp_isoline", "n_isoline_left",
                  "iso_explained_files", "n_comp_file_window_only", "no_comp_info", "ion_only_reading", "lead_fact")


def _committed_z(ctx, pr: dict, n: str, a: str) -> list:
    """The committed reading's per-file error (vs its mono ion; the row's
    ppm_error when that is not finite or >= OFFWIN_E_MAX_PPM) in units of the
    file's window: (e - mu_f) / sigma_f."""
    zs = []
    try:
        th = C.ion_mz(n, a)
    except Exception:  # noqa: BLE001
        th = np.nan
    for o in pr["obs"]:
        e = (o["mz"] - th) / th * 1e6 if np.isfinite(th) else np.nan
        if not (np.isfinite(e) and abs(e) < OFFWIN_E_MAX_PPM):
            e = o.get("ppm", np.nan)
        mu_f, sg_f = ctx.windows.get(o["sid"], ctx.run_window)
        if np.isfinite(e) and sg_f > 0:
            zs.append((e - mu_f) / sg_f)
    return zs


def pass_b(ctx, facts: pd.DataFrame, pairs: dict, q1: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-pair step-1 facts and the competitor rows, from pass A.

    ``facts`` = the pair table (neutral_formula, adduct + the B-series facts
    ``ion_only`` and ``lead``); ``pairs`` = `pairs_from_files`; ``q1`` =
    `q1_pass`. Returns (one row per pair, in ``facts`` order; the competitor
    rows: COMP_COLUMNS, status 'excluded' (how 'isotopes') | 'left')."""
    truthy = _ev().truthy
    F = {(str(r.neutral_formula), str(r.adduct)): r
         for r in facts[["neutral_formula", "adduct", "ion_only", "lead"]].itertuples(index=False)}
    engine_ch = set(ctx.engine_channels)
    rows = []
    comp_out = []
    for k, pr in pairs.items():
        f = F[(str(k[0]), str(k[1]))]
        n, a = k
        v = q1[k]
        cands = v["cands"]
        com = cands[0]
        ion_only = bool(truthy(f.ion_only)) or a in SP.ION_ONLY
        comp_rows = []
        for c in cands[1:]:
            why = []
            if c["a"]:
                why.append("iso(a): " + "; ".join(c["a"]))
            if c["b"]:
                why.append("iso(b): " + "; ".join(c["b"]))
            if c["c"]:
                why.append("iso(c): " + c["c"])
            if c.get("k"):
                why.append("label twin: " + c["k"])
            comp_rows.append(dict(name=c["name"], kind=c["kind"], ppm=c["ppm"], adduct=c["adduct"],
                                  excluded=bool(why), how="isotopes" if why else "",
                                  why=" | ".join(why) if why else c.get("stay", ""),
                                  win=c.get("win", ""), neutral=c.get("neutral", "")))
        for c in v.get("isolines", []):
            # 'this peak is line L of committed reading P': no neutral of its own (never series-testable)
            comp_rows.append(dict(name=c["name"], kind="isoline", ppm=c["ppm"], adduct=c["adduct"],
                                  excluded=c["excluded"], how="isotopes" if c["excluded"] else "", why=c["why"],
                                  win="", neutral=""))
        left = [c for c in comp_rows if not c["excluded"]]
        n_iso = sum(1 for c in comp_rows if c["excluded"])
        committed_bad = com["contradicted"]
        enum_ok = com.get("enum_ok", True)
        no_comp_info = com["ion_key"] is None or not com["counts"] or not enum_ok
        decs = ctx.decompositions(com["counts"], n, a) if com["counts"] else []
        plaus = [d for d in decs if d["ok"]]
        self_ok = any(d["adduct"] == a and d["neutral"] and SP.same_formula(d["neutral"], n) and d["ok"]
                      for d in decs)
        zs = _committed_z(ctx, pr, n, a)
        z_med = float(np.median(zs)) if zs else np.nan
        for c in comp_rows:
            comp_out.append(dict(neutral_formula=n, adduct=a, competitor=c["name"], kind=c["kind"], ppm=c["ppm"],
                                 status="excluded" if c["excluded"] else "left", how=c["how"], why=c["why"],
                                 window=c["win"], comp_neutral=c["neutral"], comp_adduct=c["adduct"]))
        rows.append(dict(
            neutral_formula=n, adduct=a,
            n_files_obs=len(pr["obs"]),
            committed_contradicted=committed_bad,
            committed_reasons=" | ".join(com["a"] + com["b"] + ([com["c"]] if com["c"] else [])),
            committed_matched=",".join(sorted(com["matched_els"])),
            committed_matched_lines="|".join(com["matched_labels"]),
            committed_c13=com["c_agree"] or com["c"],
            strong_isotope=len(com["matched_els"]) >= 2,
            n_competitors=len(comp_rows),
            n_comp_mass=sum(1 for c in cands[1:] if c["kind"].startswith("mass")),
            n_comp_mass_engine=sum(1 for c in cands[1:] if c["kind"].startswith("mass")
                                   and (c["adduct"] in engine_ch or c["adduct"] == a)),
            n_comp_alt=v["n_alt"], n_excl_iso=n_iso, n_excl_routes=0, n_left=len(left),
            competitors_left="; ".join(c["name"] for c in left),
            competitors_excluded=" || ".join(f"{c['name']}: {c['why']}" for c in comp_rows if c["excluded"]),
            n_decomp_plausible=len(plaus), committed_plausible=self_ok,
            plaus_decomps="; ".join(f"{d['neutral']}|{d['adduct']}" for d in plaus if d["neutral"]),
            outside_space=v["outside"] or "", space_note=com.get("relax_note", ""),
            n_tie_same=v.get("n_tie_same", 0),
            committed_z_med=z_med, committed_files_off_window=int(sum(1 for z in zs if abs(z) > OFFWIN_Z)),
            off_own_window=bool(np.isfinite(z_med) and abs(z_med) > OFFWIN_Z),
            n_comp_isoline=sum(1 for c in comp_rows if c["kind"] == "isoline"),
            n_isoline_left=sum(1 for c in comp_rows if c["kind"] == "isoline" and not c["excluded"]),
            iso_explained_files="|".join(v.get("iso_explained", [])),
            n_comp_file_window_only=sum(1 for c in cands[1:] if c.get("win") and "run" not in c.get("win", "")),
            no_comp_info=no_comp_info, ion_only_reading=ion_only, lead_fact=bool(truthy(f.lead)),
        ))
    return pd.DataFrame(rows, columns=list(PASS_B_COLUMNS)), pd.DataFrame(comp_out, columns=list(COMP_COLUMNS))
