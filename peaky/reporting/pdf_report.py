"""Standard PDF report for a cover-selected batch assignment run.

Assembles the assignment findings into one PDF per batch: cover + headline,
coverage stats (assigned vs unassigned, by count AND signal), composition / full
Van Krevelen, analyte families, correlated-cluster TS figures, and methods.

ITERABLE BY DESIGN: the report is an ordered list of SECTIONS, each a function
`section(ctx, pdf)` that draws one or more pages. To change the report, edit /
reorder / add a section function and list it in SECTIONS — nothing else couples.
`ctx` is a dict loaded once by `load_context()` from the run's on-disk artifacts
(merged_ledger.csv, per_file/*, the figures, *_summary.json), so a section just
reads what it needs and degrades gracefully when an artifact is missing.

Pure matplotlib (PdfPages) — no new dependencies, no web stack. Build with
`build(out_dir, tag=..., label=..., ts_path=...)`.
"""
from __future__ import annotations

import glob
import json
import os
import textwrap

import numpy as np
import pandas as pd

from peaky import paths as PT
from peaky.assignment import reflists as RL

__version__ = "0.1.0"

A4 = (8.27, 11.69)                       # portrait inches
INK = "#222222"
GREY = "#777777"


# ---------------------------------------------------------------------------
# context: load everything the report needs, once
# ---------------------------------------------------------------------------
def _skill_version() -> str:
    """Identify the skill build: assign version + git short SHA of this repo."""
    import subprocess
    d = os.path.dirname(os.path.abspath(__file__))
    sha = "?"
    try:
        sha = subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    try:
        from peaky.assignment import assign
        av = assign.__version__
    except Exception:
        av = "?"
    return f"peaky (assign v{av}) · git {sha}"


def load_context(out_dir: str, *, tag: str, label: str, ts_path: str | None = None,
                 generated: str = "", batch_name: str | None = None,
                 dataset: str | None = None,
                 run_id: str | None = None) -> dict:
    from peaky.reporting import analyte_viz as V
    from peaky.chem import chemistry as C
    out_dir = os.path.expanduser(out_dir)
    RP = PT.run_paths(out_dir)
    FIG, TAB = RP.figures, RP.tables       # MUST mirror the writers (clustering/analyte_viz)
    ctx: dict = {"out_dir": out_dir, "fig_dir": FIG, "tag": tag, "label": label,
                 "fig": {}, "generated": generated, "version": _skill_version(),
                 "batch_name": batch_name, "dataset": dataset, "run_id": run_id}

    merged = pd.read_csv(f"{out_dir}/merged_ledger.csv")
    ctx["merged"] = merged
    ctx["n_m0"] = len(merged)
    ctx["tiers"] = merged["tier"].value_counts().to_dict()
    u = merged.drop_duplicates("neutral_formula").copy()
    ctx["n_neutrals"] = len(u)
    # composition by CHO/CHON/CHOS backbone (Si/F/halogen folded in)
    ctx["composition"] = u["neutral_formula"].map(V.backbone_class).value_counts().to_dict()
    # heteroatom side-counts (additions to the backbone)
    cnt = u["neutral_formula"].map(lambda f: C.parse_formula(str(f)))
    ctx["hetero"] = {
        "Si-bearing": int(cnt.map(lambda c: c.get("Si", 0) > 0).sum()),
        "F-bearing": int(cnt.map(lambda c: c.get("F", 0) > 0).sum()),
        "Cl/Br in neutral": int(cnt.map(lambda c: c.get("Cl", 0) + c.get("Br", 0) > 0).sum()),
    }
    # match score per tier + adduct-channel breakdown (assignment-quality page)
    if "ion_score" in merged.columns:
        gs = merged.groupby("tier")["ion_score"].mean()
        ctx["score_by_tier"] = {t: float(gs[t]) for t in gs.index}
    ctx["adduct_counts"] = merged["adduct"].value_counts().to_dict()   # actual channels
    if "admitted_by" in merged.columns:   # admission provenance (persistence path)
        ctx["n_admitted_occurrence"] = int((merged["admitted_by"] == "occurrence").sum())
    # the selected sample NAMES (timestamps), not just ids
    ss = f"{TAB}/selected_samples.csv"
    if os.path.exists(ss):
        s = pd.read_csv(ss)
        name = s["sample_item_name"] if "sample_item_name" in s.columns else s["sample_item_id"]
        ctx["samples"] = list(zip(name.astype(str), s.get("role", pd.Series([""] * len(s))).astype(str)))

    # per-file pooled role breakdown (count + signal) + the explained m/z set
    rows, rep_ids = [], []
    for f in sorted(glob.glob(f"{out_dir}/per_file/*_ledger.csv")):
        rep_ids.append(os.path.basename(f).replace("_ledger.csv", ""))
        d = pd.read_csv(f)
        d["h"] = pd.to_numeric(d.get("height"), errors="coerce").fillna(0)
        rows.append(d)
    ctx["n_files"] = len(rows)
    ctx["rep_ids"] = rep_ids
    if rows:
        a = pd.concat(rows, ignore_index=True)
        ctx["role_count"] = a["role"].value_counts().to_dict()
        # the persistence path's ACTIVITY is per file: the merge keeps the highest
        # tier then score, so an occurrence-admitted row almost always loses to a
        # height-admitted one in the same bin, and the merged count (0-3 on real
        # batches) under-reports per-file admissions (14-63) by an order of
        # magnitude -- on one batch it suppressed the cover line entirely.
        if "admitted_by" in a.columns:
            ctx["n_admitted_occurrence_files"] = int((a["admitted_by"] == "occurrence").sum())
        # the BRIGHTEST selected-sample full ledger (max total height) — the
        # mass-defect / mass-error QC figure (qc_massdefect) reads it whole (all
        # roles incl iso_child + unexplained), not the M0-only merged ledger.
        ctx["bright_ledger"] = max(rows, key=lambda r: float(r["h"].sum()))
        ctx["role_signal"] = {k: float(v) for k, v in a.groupby("role")["h"].sum().items()}
        # signal share split into analyte (M0+iso) / reagent / unexplained — for an
        # honest "explained" headline: a Br- spectrum is mostly the reagent ion, so
        # "98% explained" must not read as "98% analyte characterised".
        rs = ctx["role_signal"]; rtot = sum(rs.values()) or 1.0
        ctx["role_signal_frac"] = {
            "analyte": (rs.get("M0", 0) + rs.get("iso_child", 0)) / rtot,
            "reagent": rs.get("reagent", 0) / rtot,
            "unexplained": (rs.get("unexplained", 0) + rs.get("artifact", 0)) / rtot}
        # per-neutral summed M0 signal (drives signal-weighted composition + findings)
        if "neutral_formula" in a.columns:
            m0a = a[a["role"] == "M0"]
            ns = m0a.groupby(m0a["neutral_formula"].astype(str))["h"].sum()
            ctx["neutral_signal"] = {k: float(v) for k, v in ns.items()
                                     if k and k != "nan"}
        ctx["expl_mz"] = np.sort(a.loc[a["role"].astype(str) != "unexplained",
                                       "mz"].dropna().to_numpy())
        ctx["_flag_ev_src"] = a          # kept for scrutiny-evidence enrichment below
        # isotopes confirmed per (neutral, channel) — union of isotopologue labels
        # across the selected files (each M0 row carries an `isotopologues`
        # JSON list of {label, score, peak_id}). Feeds the assignment appendix.
        iso_by_channel: dict = {}
        if "isotopologues" in a.columns:
            m0i = a[a["role"] == "M0"]
            for nf, ad, raw in zip(m0i["neutral_formula"].astype(str),
                                   m0i["adduct"].astype(str), m0i["isotopologues"]):
                if not (isinstance(raw, str) and raw.strip().startswith("[")):
                    continue
                try:
                    labs = [d.get("label") for d in json.loads(raw)]
                except Exception:
                    labs = []
                labs = [str(l) for l in labs if l]
                if labs:
                    iso_by_channel.setdefault((nf, ad), set()).update(labs)
        ctx["iso_by_channel"] = iso_by_channel
        # max observed intensity (cps) per (neutral, channel): the brightest M0
        # height for that channel across the selected files. Heights live in
        # the per-file ledgers (not the m/z-only merged ledger), so pool them here.
        if {"neutral_formula", "adduct"} <= set(a.columns):
            _m0h = a[a["role"] == "M0"]
            if len(_m0h):
                mh = _m0h.groupby([_m0h["neutral_formula"].astype(str),
                                   _m0h["adduct"].astype(str)])["h"].max()
                ctx["max_h_by_channel"] = {(nf, ad): float(v)
                                           for (nf, ad), v in mh.items()}
        # mass error (ppm) by category for the accuracy plot
        if "ppm_error" in a.columns:
            m0 = a[a["role"] == "M0"]
            pbc = {}
            for t in ("Assigned", "Candidate"):
                v = m0[m0.get("tier") == t]["ppm_error"].dropna().tolist()
                if v:
                    pbc[t] = v
            iso = _iso_ppm(rows)
            if iso:
                pbc["isotopologue"] = iso
            ctx["ppm_by_cat"] = pbc
        if "adduct" in a.columns:                     # signal share per ion channel
            asig = a[a["role"] == "M0"].groupby("adduct")["h"].sum()
            tot = float(asig.sum()) or 1.0
            ctx["adduct_signal"] = {k: float(v) / tot for k, v in asig.items()}

        # REFERENCE-LIST prior: unlock literature peaklists from the run's context
        # (batch-name/label metadata), then (1) corroborate Candidate-tier neutrals
        # by formula membership and (2) rescue UNEXPLAINED peaks by mass under the
        # actual reagent adducts. Soft + provenance-tagged; never overrides a tier.
        try:
            tags = RL.resolve_context_tags(ctx.get("batch_name") or "", ctx.get("dataset") or "",
                                           ctx.get("label") or "")
            lists = RL.active_lists(RL.load_catalog(), context_tags=tags)
            if lists:
                cand = merged.loc[merged.get("tier") == "Candidate", "neutral_formula"].dropna()
                corr = RL.match_assigned(cand.unique(), lists)
                adducts = list(ctx.get("adduct_counts", {}).keys()) or ["[M-H]-"]
                un_mz = a.loc[a["role"] == "unexplained", "mz"].dropna()
                un_mz = sorted(set(round(float(x), 5) for x in un_mz))
                rescue = RL.match_by_mass(un_mz, lists, adducts, tol_ppm=4.0)
                by_id = {L.id: L for L in lists}            # attach compound names (contaminants)
                for m in rescue:
                    nm = (by_id.get(m["list"]).meta_of or {}).get(m["formula"], {})
                    m["name"] = nm.get("name", "")
                ctx["reflist"] = {
                    "tags": sorted(tags), "lists": [L.id for L in lists],
                    "cites": [L.cite() for L in lists],
                    "n_candidate": int(cand.nunique()), "corroborated": corr,
                    "n_unexplained": len(un_mz), "rescue": rescue}
        except Exception:
            pass

    # signal-weighted composition + ammonium/amine degeneracy (composition page)
    from peaky.batch import composition as CMP
    nsig = ctx.get("neutral_signal", {})
    ctx["sig_comp_frac"], ctx["sig_comp_abs"] = CMP.signal_by_backbone(merged, nsig)
    ctx["shadow"] = CMP.amine_shadow_stats(merged)
    ctx["comp_asg"], ctx["comp_collapsed"], ctx["n_collapsed"] = CMP.collapsed_composition(merged)
    ctx["top_species"] = CMP.top_species_by_signal(merged, nsig, n=8)
    ctx["oligomers"] = CMP.oligomer_flag(merged)
    # polarity (gates positive-only messaging: the amine re-read, the shadow note)
    # + chemical-plausibility QC of the assignments
    ctx["positive"] = any(str(k).rstrip().endswith("+") for k in ctx.get("adduct_counts", {}))
    from peaky.assignment import plausibility as PL
    pol = "+" if ctx["positive"] else "-"
    ctx["flagged"] = PL.scan(merged, polarity=pol)
    # enrich each flagged neutral with its evidence (ppm / isotopes / margin / sane
    # alternative) so the scrutiny page shows WHY it is suspect and whether a saner
    # formula was available — answering "how did we arrive at these?" on the page.
    ev_src = ctx.pop("_flag_ev_src", None)
    if ev_src is not None and ctx["flagged"]:
        ev = _flag_evidence(ev_src, pol)
        for d in ctx["flagged"]:
            d.update(ev.get(d["neutral_formula"], {}))
    # single source of truth for "formula disagreements": the merged ledger's own
    # formula_agree column (the artifact the report publishes), with a denominator
    # — not the separate jitter pass, which clusters differently and prints a
    # slightly different number (the 65-vs-68 mismatch).
    if "formula_agree" in merged.columns:
        ctx["n_disagree"] = int((~merged["formula_agree"].astype(bool)).sum())
    if "n_files" in merged.columns:
        ctx["n_multifile"] = int((merged["n_files"] >= 2).sum())

    # whole-spectrum coverage (TS bins explained vs unexplained, count + signal)
    if ts_path and os.path.exists(os.path.expanduser(ts_path)):
        from peaky.batch import timeseries as TS
        ts = pd.read_parquet(os.path.expanduser(ts_path))
        if not ctx.get("batch_name") and "sample_batch_name" in ts.columns:
            names = ts["sample_batch_name"].dropna().unique()
            if len(names):
                ctx["batch_name"] = str(names[0])
        mat, bin_mz = TS.build_matrix(ts)
        binsig = mat.sum(axis=0)
        # batch-wide MAX intensity per assigned channel: the brightest this ion gets
        # in ANY sample of the FULL batch (the appendix's `max cps`). The per-file
        # value computed above only saw the cover-selected samples; override it
        # with the whole-batch max by matching each channel's ion m/z to its TS bin.
        _bmax = mat.max(axis=0)
        _bins = list(mat.columns)
        _bmz = np.array([float(bin_mz[b]) for b in _bins], dtype=float)
        _bmx = _bmax.reindex(_bins).to_numpy(dtype=float)
        _o = np.argsort(_bmz); _bmz = _bmz[_o]; _bmx = _bmx[_o]

        def _batch_max(mz, tol=8.0):
            i = int(np.searchsorted(_bmz, mz)); best = None
            for j in (i - 1, i):
                if 0 <= j < len(_bmz):
                    ppm = abs(_bmz[j] - mz) / mz * 1e6
                    if ppm <= tol and (best is None or ppm < best[1]):
                        best = (float(_bmx[j]), ppm)
            return None if best is None else best[0]
        if "neutral_formula" in merged.columns and "adduct" in merged.columns:
            mbc = {}
            for _mz, _nf, _ad in zip(merged["mz"], merged["neutral_formula"].astype(str),
                                     merged["adduct"].astype(str)):
                v = _batch_max(float(_mz))
                if v is not None:
                    mbc[(_nf, _ad)] = v
            if mbc:
                ctx["max_h_by_channel"] = mbc     # batch-wide, replaces the rep-file max
                ctx["max_h_scope"] = "batch"
        expl = ctx.get("expl_mz", np.array([]))

        def matched(mz, tol=8.0):
            i = np.searchsorted(expl, mz)
            return any(0 <= j < len(expl) and abs(expl[j] - mz) / mz * 1e6 <= tol
                       for j in (i - 1, i))
        ex = np.array([matched(float(bin_mz[b])) for b in mat.columns])
        ctx["ts"] = {"nbins": int(len(ex)), "expl_count": int(ex.sum()),
                     "expl_signal": float(binsig.values[ex].sum()),
                     "tot_signal": float(binsig.sum())}
        # event overview: per-sample total signal vs wall-clock over the FULL batch
        # (the reader needs to SEE the burst; the cluster pages only show a
        # normalised 0-1 'hour' axis). Mark where the selected files sit.
        try:
            tt = pd.to_datetime(ts["datetime_utc"], utc=True)
            tstamp = tt.groupby(ts["sample_item_id"]).first()
            order = tstamp.sort_values().index
            t0 = tstamp.min()
            hrs = (tstamp.reindex(order) - t0).dt.total_seconds().to_numpy() / 3600.0
            tot = mat.reindex(order).sum(axis=1).to_numpy()
            rep_h = [float((tstamp[r] - t0).total_seconds() / 3600.0)
                     for r in ctx.get("rep_ids", []) if r in tstamp.index]
            ctx["event"] = {"hours": hrs, "total": tot, "rep_hours": rep_h}
        except Exception:
            pass

    for key, fn in [("jitter", "jitter_summary.json"), ("batch", "batch_summary.json"),
                    ("clusters", "clusters_summary.json")]:
        p = f"{out_dir}/{fn}"
        if os.path.exists(p):
            ctx[key] = json.load(open(p))
    vk = f"{FIG}/van_krevelen_full_{tag}.png"
    if os.path.exists(vk):
        ctx["fig"]["vk"] = vk           # single (scatter)
    # cluster figures are PAGED (clusters_<set>_<tag>_p<i>.png) — collect ALL pages
    for key, stem in [("changing", f"clusters_changing_{tag}"),
                      ("changers", f"clusters_changers_{tag}"),
                      ("flat", f"clusters_flat_{tag}"),
                      ("unassigned", f"clusters_unassigned_{tag}")]:
        paged = sorted(glob.glob(f"{FIG}/{stem}_p*.png"),
                       key=lambda s: int(s.rsplit("_p", 1)[1].split(".")[0]))
        if paged:
            ctx["fig"][key] = paged
        elif os.path.exists(f"{FIG}/{stem}.png"):
            ctx["fig"][key] = [f"{FIG}/{stem}.png"]
    cc = f"{TAB}/clusters_changing_{tag}.csv"
    if os.path.exists(cc):
        ctx["changing_csv"] = pd.read_csv(cc)
    return ctx


# ---------------------------------------------------------------------------
# page helpers
# ---------------------------------------------------------------------------
def _close(pdf, fig):
    import matplotlib.pyplot as plt
    pdf.savefig(fig)
    plt.close(fig)


def _text_lines(fig, lines, *, x=0.08, y0=0.90, dy=0.026, size=10, bottom=0.045):
    """Render (style, text) lines top-down. style: 'h' head / 'b' body / 'm' mono /
    'dim' caption / 'gap'.

    Long 'h'/'b'/'dim' lines WRAP to the printable page width so a sentence never
    runs off the right margin (a leading '• '/indent is kept aligned); 'm' lines are
    pre-aligned tables and are left verbatim. If the wrapped block would run past
    `bottom`, the line spacing — and, only if that is not enough, the font size — is
    scaled down so the block still fits the page (a long caption no longer overflows
    the bottom margin)."""
    usable_in = max(1.0, (1.0 - x - 0.07) * A4[0])      # printable width (right margin ~0.07)
    def _budget(fontsize):                              # ~chars that fit at this size
        return max(24, int(usable_in / (0.54 * fontsize / 72.0)))
    def _fs(style):                                     # font size per style
        return (size + 3 if style == "h" else size - 1 if style == "dim"
                else size - 0.5 if style == "m" else size)

    # Pass 1 — expand to physical pieces, honouring wrapping: ('gap', units) | (style, text)
    pieces = []
    for style, txt in lines:
        if style == "gap":
            pieces.append(("gap", (txt or 1)))
            continue
        fs = _fs(style)
        if style in ("h", "b", "dim") and txt and len(str(txt)) > _budget(fs):
            t = str(txt)
            lead = t[:len(t) - len(t.lstrip())]                  # existing indent
            cont = "  " if t.lstrip().startswith("•") else lead  # align under a bullet
            for seg in (textwrap.wrap(t, width=_budget(fs), initial_indent=lead,
                                      subsequent_indent=cont) or [t]):
                pieces.append((style, seg))
        else:
            pieces.append((style, txt))

    # Pass 2 — fit vertically: scale spacing (then font, only if forced) to stay above `bottom`
    n_lines = sum(1 for k, _ in pieces if k != "gap")
    gap_units = sum(v for k, v in pieces if k == "gap")
    extent = (n_lines + gap_units) * dy
    avail = y0 - bottom
    sdy, fscale = dy, 1.0
    if extent > avail > 0:
        sdy = dy * avail / extent
        line_h = 1.16 * size / 72.0 / A4[1]             # ~line height (figure fraction)
        if sdy < line_h:                                # spacing alone not enough -> shrink font
            fscale = sdy / line_h

    y = y0
    for kind, val in pieces:
        if kind == "gap":
            y -= sdy * val
            continue
        kw = dict(fontsize=_fs(kind) * fscale, color=INK, va="top", family="sans-serif")
        if kind == "h":
            kw.update(weight="bold")
        elif kind == "m":
            kw.update(family="monospace")
        elif kind == "dim":
            kw.update(color=GREY)
        fig.text(x, y, val, **kw)
        y -= sdy
    return y


def _image_page(pdf, png, title, *, landscape=False, dpi=200, native=False, src_dpi=170):
    """Embed a PNG as one page. native=True makes the PAGE the image's own size
    (page = pixels/src_dpi inches) so a tall cluster figure embeds 1:1 and stays
    fully legible (the page is tall; scroll). Otherwise fit to A4 (landscape opt)."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    img = mpimg.imread(png)
    if native:
        ih, iw = img.shape[0] / src_dpi, img.shape[1] / src_dpi
        Th = 0.5 if title else 0.0                  # title strip (inches); the figures
        fig = plt.figure(figsize=(iw, ih + Th), dpi=src_dpi)   # carry their own title
        if title:
            fig.text(0.01, 1 - 0.20 / (ih + Th), title, fontsize=12, weight="bold",
                     color=INK, va="top")
        ax = fig.add_axes([0.005, 0.002, 0.99, ih / (ih + Th)])
        ax.imshow(img); ax.axis("off")
        pdf.savefig(fig); plt.close(fig)
        return
    figsize = (A4[1], A4[0]) if landscape else A4
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if title:
        fig.text(0.04, 0.975, title, fontsize=13, weight="bold", color=INK)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.93] if title else [0.02, 0.02, 0.96, 0.96])
    ax.imshow(img); ax.axis("off")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def _pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# readable descriptor for each ion channel (so the breakdown names the actual adduct)
_ADDUCT_DESC = {
    "[M+H]+": "protonated", "[M-H]-": "deprotonated",
    "[M+Br]-": "Br- cluster", "[M+HBr+Br]-": "di-bromide cluster", "[M+Br2]-": "Br2 cluster",
    "[M+CO3]-": "CO3- cluster", "[M+HBr+CO3]-": "HBr.CO3- cluster", "[M+HSO4]-": "HSO4- cluster",
    "[M+(CH4N2O)H]+": "urea cluster", "[M+Na]+": "Na+ adduct", "[M+NH4]+": "NH4+ adduct",
    "[M+Cl]-": "Cl- cluster", "[M+I]-": "I- cluster",
    "[M+NO3]-": "NO3- cluster", "[M+^NO3]-": "[15N]O3- cluster",
    "[M+^NH4]+": "[15N]H4+ adduct", "[M+^NH4-H2O]+": "[15N]H4+ adduct, dehydrated",
    "[M+H-H2O]+": "protonated, dehydrated",
}


def _adduct_label(a) -> str:
    a = str(a)
    d = _ADDUCT_DESC.get(a)
    return f"{a}  ({d})" if d else a


_ISO_C13 = 1.0033548   # 13C - 12C; used to recover isotopologue mass error


def _flag_evidence(a, polarity):
    """Per-neutral evidence for the scrutiny page: smallest |ppm|, count of CONFIRMED
    isotopologues (0 = no corroboration), eff_margin to the runner-up, and whether a
    chemically PLAUSIBLE alternative existed within the candidate set (parsed from the
    `alternatives` JSON) — i.e. whether the implausible fit was forced or passed over a
    saner one. Returns {neutral_formula: {ppm, iso, margin, sane_alt}}."""
    from peaky.assignment import plausibility as PL
    out = {}
    if "neutral_formula" not in a.columns:
        return out
    m0 = a[a["role"] == "M0"]
    for nf, sub in m0.groupby(m0["neutral_formula"].astype(str)):
        r = sub.iloc[(sub["ppm_error"].abs().argmin()
                      if "ppm_error" in sub.columns and sub["ppm_error"].notna().any() else 0)]
        iso = 0
        raw = r.get("isotopologues")
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                iso = len(json.loads(raw))
            except Exception:
                iso = 0
        sane = None
        alt = r.get("alternatives")
        if isinstance(alt, str) and alt.strip().startswith("["):
            try:
                for d in json.loads(alt):
                    af = d.get("formula")
                    if af and PL.implausible(af, tier="Candidate", polarity=polarity) is None:
                        sane = f"{af} @ {d.get('ppm'):+.2f} ppm" if d.get("ppm") is not None else af
                        break
            except Exception:
                pass
        out[nf] = {"ppm": float(r["ppm_error"]) if pd.notna(r.get("ppm_error")) else None,
                   "iso": int(iso),
                   "margin": float(r["eff_margin"]) if pd.notna(r.get("eff_margin")) else None,
                   "sane_alt": sane}
    return out


def _iso_ppm(per_file_frames):
    """Isotopologue mass error (ppm): per file, match each M0's predicted 13C M+1
    (parent ion m/z + 1.00335) to the nearest iso_child peak. iso_child rows carry
    no ppm_error of their own, so recompute it here."""
    import numpy as np
    out = []
    for d in per_file_frames:
        if "role" not in d.columns or "mz" not in d.columns:
            continue
        iso = np.sort(d[d["role"] == "iso_child"]["mz"].dropna().to_numpy())
        if not len(iso):
            continue
        for mz in d[d["role"] == "M0"]["mz"].dropna():
            tgt = float(mz) + _ISO_C13
            j = np.searchsorted(iso, tgt)
            for k in (j - 1, j):
                if 0 <= k < len(iso):
                    p = (iso[k] - tgt) / tgt * 1e6
                    if abs(p) <= 6:
                        out.append(p); break
    return out


# ---------------------------------------------------------------------------
# sections  (each draws page(s); add/reorder freely)
# ---------------------------------------------------------------------------
def cover(ctx, pdf):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4)
    batch = ctx.get("batch_name") or ctx["label"]
    fig.text(0.08, 0.93, "Peak Assignment Report", fontsize=20, weight="bold", color=INK)
    fig.text(0.08, 0.895, batch, fontsize=14, color=INK)
    _bsel = (ctx.get("batch") or {}).get("selection") or {}
    _pipe = ("single-sample" if ctx.get("n_files", 1) <= 1
             else _bsel.get("method") or "batch")   # the recorded selector, never a hard-coded rule
    fig.text(0.08, 0.872, f"{ctx['label']} · {_pipe} pipeline", fontsize=11, color=GREY)
    meta = ctx.get("version", "")
    if ctx.get("generated"):
        meta = f"{meta}  ·  generated {ctx['generated']}" if meta else f"generated {ctx['generated']}"
    if meta:
        fig.text(0.08, 0.852, meta, fontsize=8.5, color=GREY)
    if ctx.get("run_id"):
        fig.text(0.08, 0.834, f"Report ID:  {ctx['run_id']}", fontsize=8.5, color=GREY)

    tiers = ctx["tiers"]; idn = tiers.get("Assigned", 0); cn = tiers.get("Candidate", 0)
    ts = ctx.get("ts"); sig = (_pct(ts["expl_signal"], ts["tot_signal"]) if ts else None)
    ex_c = (_pct(ts["expl_count"], ts["nbins"]) if ts else None)
    head = [
        ("h", "Summary"),
        ("gap", 0.3),
        ("b", f"Unique analytes assigned (M0):   {ctx['n_m0']}   "
              f"({idn} Assigned / {cn} Candidate)"),
        ("b", f"Distinct neutral compounds:       {ctx['n_neutrals']}"),
    ]
    rc = ctx.get("role_count", {})
    if rc:
        tot = sum(rc.values())
        asg = rc.get("M0", 0) + rc.get("iso_child", 0) + rc.get("reagent", 0)
        scope = "" if ctx.get("n_files", 1) <= 1 else f" (pooled over {ctx['n_files']} files)"
        head += [
            ("b", f"Peaks{scope}:".ljust(34) + f"{tot} total"),
            ("dim", f"   {asg} assigned ({rc.get('M0', 0)} M0 + {rc.get('iso_child', 0)} isotopologues "
                    f"+ {rc.get('reagent', 0)} reagent + {rc.get('artifact', 0)} artifact) · "
                    f"{rc.get('unexplained', 0)} unexplained ({_pct(rc.get('unexplained', 0), tot):.0f}%)")]
    if ts:
        head += [
            ("b", f"Spectral coverage:                {ex_c:.0f}% of m/z bins, "
                  f"{sig:.0f}% of signal explained"),
            ("b", f"Unexplained:                      {ts['nbins']-ts['expl_count']} bins "
                  f"({100-ex_c:.0f}% count, {100-sig:.0f}% signal)"),
        ]
        rf = ctx.get("role_signal_frac", {})
        if rf.get("reagent", 0) >= 0.05:        # reagent-dominated (e.g. Br-): be honest
            head += [("dim", "   of detected signal:  analyte (M0+iso) "
                             f"{rf['analyte']*100:.0f}%,  reagent ion {rf['reagent']*100:.0f}%,  "
                             f"unexplained {rf['unexplained']*100:.0f}%")]
    nf = ctx.get("n_files", 1)
    bsel = (ctx.get("batch") or {}).get("selection") or {}
    if nf <= 1:
        sel_txt = "Single sample assigned (no merge)."
    elif bsel.get("method") == "presence-cover":
        sel_txt = (f"{nf} files: presence set-cover selection — greedy over the "
                   f"{bsel.get('n_bins', '?')} m/z bins present in ≥{bsel.get('min_prevalence', 2)} "
                   f"samples (no height floor); {bsel.get('achieved_coverage', 0):.0%} of "
                   f"them covered, stopped on {bsel.get('stop_reason', '?')}"
                   + (f" (budget k_max={bsel.get('k_max')} hit while still gaining — "
                      "coverage is incomplete)" if bsel.get("stop_reason") == "k_max" else "")
                   + "; merged by m/z.")
    else:                                   # a run folder without a selection record
        sel_txt = (f"{nf} files assigned and merged by m/z (this run's batch_summary.json "
                   "carries no selection record).")
    head += [("gap", 1), ("h", "Samples assigned"), ("gap", 0.3), ("b", sel_txt)]
    for name, role in ctx.get("samples", [])[:8]:
        head.append(("m", f"   {name}   [{role}]"))
    # persistence path of the admission gate: state the RESOLVED threshold (the
    # fraction the run actually compared against) and the knob it came from --
    # `occurrence_min` is the knob ('auto' by default) and cannot be formatted
    # as a percentage; `occurrence_threshold` is None when the path was off.
    _adm = (ctx.get("batch") or {}).get("admission") or {}
    _thr = _adm.get("occurrence_threshold")
    _n_occ_files = ctx.get("n_admitted_occurrence_files")
    if (ctx.get("n_admitted_occurrence") or _n_occ_files) and _thr is not None:
        # per-file rows are the path's real activity; the merged count is what
        # survived the merge's tier-then-score consensus (see load_context)
        _pf = (f"{_n_occ_files} per-file peak rows, " if _n_occ_files is not None else "")
        head.append(("dim", f"   {_pf}{ctx.get('n_admitted_occurrence', 0)} of {ctx.get('n_m0', '?')} "
                            f"merged peaks were eligible by persistence only: a peak within "
                            f"tolerance in ≥{float(_thr):.0%} of spectra (occurrence-min "
                            f"{_adm.get('occurrence_min', 'auto')}), below the height gate"))
    _gate = (ctx.get("batch") or {}).get("gate") or {}
    if _gate.get("x_edge") is not None:
        head.append(("dim", f"   brightness floor derived from the batch: {_gate['x_edge']:g}x the "
                            f"noise edge ({_gate.get('share_at_x', 0):.0%} of the admitted peaks "
                            f"transient, ≤{_gate.get('max_transient_share', 0):.0%} allowed; "
                            f"{(_gate.get('share_at_1') or 0):.0%} at 1x)"))
    _tr = (ctx.get("batch") or {}).get("traces") or {}
    if _tr.get("n_rows"):
        head.append(("dim", f"   traces: {_tr.get('n_recentred', 0)} of {_tr['n_rows']} ledger anchors "
                            f"re-centred on their own trace (median move "
                            f"{_tr.get('median_abs_move_ppm', 0)} ppm), {_tr.get('n_collapsed', 0)} "
                            f"competing labels collapsed onto {_tr.get('n_traces', '?')} traces; "
                            f"stamping window ±{_tr.get('stamp_tol_ppm', '?')} ppm "
                            f"(per-ion scatter {_tr.get('sigma_ppm', 'n/a')} ppm)"))
    j = ctx.get("jitter", {})
    if j:
        nm = ctx.get("n_multifile")
        nd = ctx.get("n_disagree", j.get("formula_disagreements", "?"))
        tu = j.get("tier_unstable", "?")
        dis = (f"{nd} of {nm} multi-file ({_pct(nd, nm):.0f}%)"
               if nm and isinstance(nd, int) else f"{nd}")
        tus = (f"{tu} of {nm} multi-file ({_pct(tu, nm):.0f}%)"
               if nm and isinstance(tu, int) else f"{tu}")
        head += [
            ("gap", 1),
            ("h", "File-to-file reproducibility"),
            ("gap", 0.3),
            ("b", f"Per-file calibration spread:  {j.get('offset_spread_ppm','?')} ppm"),
            ("b", f"Mass jitter (median / p95):   {j.get('mz_jitter_raw_median','?')} / "
                  f"{j.get('mz_jitter_raw_p95','?')} ppm  (≈ genuine peak noise)"),
            ("b", f"Formula disagreements:        {dis}  (same m/z, different formula)"),
            ("b", f"Tier-unstable assignments:    {tus}  (flip Assigned <-> Candidate)"),
            ("dim", "On a disagreement/flip the merge keeps the highest tier, then the "
                    "highest match score."),
        ]
    _text_lines(fig, head, y0=0.80, dy=0.029)
    _close(pdf, fig)


def findings(ctx, pdf):
    """Plain-language findings page (right after the cover): the event time-trace
    plus data-driven takeaways — top species by signal, signal-weighted
    composition, and the oligomer/HOM fingerprint. Answers 'what changed during
    the run?' before the QC detail. Everything is derived from the data, so it
    works for any batch."""
    import matplotlib.pyplot as plt
    ev = ctx.get("event"); top = ctx.get("top_species", [])
    scf = ctx.get("sig_comp_frac", {}); olig = ctx.get("oligomers", [])
    if not (ev or top):
        return
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.955, "Findings", fontsize=16, weight="bold", color=INK)
    rise_txt = None
    if ev and len(ev.get("total", [])):
        ax = fig.add_axes([0.10, 0.60, 0.82, 0.27])
        h = np.asarray(ev["hours"], float); tt = np.asarray(ev["total"], float)
        ax.plot(h, tt, color="#1D9E75", lw=1.5)
        ax.fill_between(h, tt, color="#1D9E75", alpha=0.12)
        for rh in ev.get("rep_hours", []):
            ax.axvline(rh, color="#888", lw=0.7, ls=":")
        ax.set_xlabel("hour of experiment (UTC)", fontsize=9)
        ax.set_ylabel("total signal (cps)", fontsize=9)
        ax.set_title("Event overview — total signal vs time (dotted = assigned samples)",
                     loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
        ok = np.isfinite(tt)
        if ok.sum() >= 3:
            tail = tt[ok][max(1, int(ok.sum() * 2 / 3)):]
            base = float(np.median(tail)) if len(tail) else float(np.median(tt[ok]))
            peak = float(np.nanmax(tt)); pk_h = float(h[int(np.nanargmax(tt))])
            if base > 0:
                rise_txt = (f"Total signal peaks at hour {pk_h:.1f} — {peak/base:.1f}x the "
                            f"late-run baseline — then decays (a transient event).")
    lines = []
    if rise_txt:
        lines += [("b", "• " + rise_txt)]
    if scf:
        cc = ctx.get("comp_asg", {}); nn = ctx.get("n_neutrals", 1)
        cho = scf.get("CHO", 0) * 100; chon = scf.get("CHON", 0) * 100
        lines += [("b", f"• By signal the assigned chemistry is {cho:.0f}% CHO / {chon:.0f}% CHON"
                        f" — vs {_pct(cc.get('CHON', 0), nn):.0f}% CHON by compound count")]
        if ctx.get("positive"):     # the CHON count inflation is the amine re-read (positive only)
            lines += [("b", "  (the count is inflated by mass-degenerate ammonium/amine re-reads; "
                            "see Composition).")]
        else:
            lines += [("b", "  (a few bright CHO species carry most of the signal).")]
    if top:
        lines += [("gap", 0.6), ("h", "Top species by signal"), ("gap", 0.25),
                  ("m", "   share   class   neutral")]
        for r in top[:8]:
            lines.append(("m", f"   {r['frac']*100:>4.1f}%   {r['klass']:5s}   {r['neutral_formula']}"))
    if olig:
        nsig = ctx.get("neutral_signal", {})
        olig = sorted(olig, key=lambda f: nsig.get(f, 0.0), reverse=True)[:12]
        lines += [("gap", 0.6), ("h", "Accretion / oligomer products (high C & O, by signal)"),
                  ("gap", 0.25)]
        for k in range(0, len(olig), 6):           # wrap ~6 formulas per line (no edge clip)
            lines.append(("m", "   " + ", ".join(olig[k:k + 6])))
        lines += [("dim", "high-carbon high-oxygen neutrals — candidate HOM dimers / oligomers,"),
                  ("dim", "often the most event-specific signal.")]
    _text_lines(fig, lines, y0=0.52, dy=0.027, size=9.5)
    _close(pdf, fig)


def coverage(ctx, pdf):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.965, "Assignment quality", fontsize=15, weight="bold", color=INK)

    # (a) mean match score, Assigned vs Candidate
    sbt = ctx.get("score_by_tier", {})
    tiers = [t for t in ("Assigned", "Candidate") if t in sbt]
    ax = fig.add_axes([0.11, 0.74, 0.33, 0.17])
    ax.bar(tiers, [sbt[t] for t in tiers], color=["#1D9E75", "#E0A93B"][:len(tiers)], width=0.55)
    for i, t in enumerate(tiers):
        ax.text(i, sbt[t], f"{sbt[t]:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("mean match score", fontsize=9)
    ax.set_title("Match score by confidence", loc="left", fontsize=11)

    # (b) mass error (ppm) by category: Assigned / Candidate / isotopologues
    pbc = ctx.get("ppm_by_cat", {})
    ax2 = fig.add_axes([0.58, 0.74, 0.34, 0.17])
    cats = [c for c in ("Assigned", "Candidate", "isotopologue") if c in pbc]
    if cats:
        bp = ax2.boxplot([pbc[c] for c in cats], vert=True, showfliers=False, widths=0.5,
                         patch_artist=True, medianprops=dict(color="#111"))
        for patch, c in zip(bp["boxes"], ["#1D9E75", "#E0A93B", "#9AD1BE"]):
            patch.set_facecolor(c)
        ax2.set_xticklabels([f"{c}\n(n={len(pbc[c])})" for c in cats], fontsize=8)
        ax2.axhline(0, color="0.7", lw=0.7)
    ax2.set_ylabel("mass error (ppm)", fontsize=9)
    ax2.set_title("Mass accuracy", loc="left", fontsize=11)

    # (c) signal & peak share by ROLE — analyte (M0+iso) / reagent ion / unexplained.
    # Splitting out the reagent is the honest "explained" picture: a Br- spectrum is
    # mostly the reagent ion, so a single "98% explained" number is misleading.
    rc = ctx.get("role_count", {}); rs = ctx.get("role_signal", {})
    def _roles(d):
        tot = sum(d.values()) or 1.0
        return (100 * (d.get("M0", 0) + d.get("iso_child", 0)) / tot,
                100 * d.get("reagent", 0) / tot,
                100 * (d.get("unexplained", 0) + d.get("artifact", 0)) / tot)
    ax3 = fig.add_axes([0.11, 0.47, 0.33, 0.17])
    if rc or rs:
        for y, d in zip([1, 0], [rc, rs]):
            an, rg, ot = _roles(d)
            ax3.barh(y, an, color="#1D9E75")
            ax3.barh(y, rg, left=an, color="#378ADD")
            ax3.barh(y, ot, left=an + rg, color="#D85A30")
            for v, x in ((an, an / 2), (rg, an + rg / 2), (ot, an + rg + ot / 2)):
                if v >= 7:
                    ax3.text(x, y, f"{v:.0f}", va="center", ha="center", fontsize=7, color="white")
        ax3.set_yticks([1, 0]); ax3.set_yticklabels(["by count", "by signal"], fontsize=9)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("% — green analyte · blue reagent · red unexpl.", fontsize=7.3)
    ax3.set_title("Signal & peaks by role", loc="left", fontsize=11)

    # (d) assignments by ACTUAL ion channel (which adduct each peak was assigned on)
    adc = ctx.get("adduct_counts", {}); asig = ctx.get("adduct_signal", {})
    items = sorted(adc.items(), key=lambda kv: kv[1], reverse=True)
    ax4 = fig.add_axes([0.58, 0.47, 0.34, 0.17])
    y = list(range(len(items)))[::-1]
    ax4.barh(y, [v for _, v in items], color="#378ADD")
    ax4.set_yticks(y); ax4.set_yticklabels([k for k, _ in items], fontsize=8, family="monospace")
    xmax = max([v for _, v in items], default=1)
    for yi, (k, v) in zip(y, items):
        st = f" · {asig[k] * 100:.0f}% sig" if k in asig else ""
        ax4.text(v + xmax * 0.02, yi, f"{v} · {_ADDUCT_DESC.get(k, '')}{st}", va="center", fontsize=6.8)
    ax4.set_xlim(0, xmax * 1.9); ax4.set_xlabel("M0 count (a compound can appear in several channels)", fontsize=7.5)
    ax4.set_title("Reagent / ion channels assigned on", loc="left", fontsize=11)

    idn = ctx["tiers"].get("Assigned", 0); cn = ctx["tiers"].get("Candidate", 0)
    lines = [("h", "Reading this page"), ("gap", 0.3),
             ("b", f"• {idn} Assigned vs {cn} Candidate. Match score = the server "
                   "isotope-scored compound match (0-1)."),
             ("b", "• Mass accuracy: ppm error of the matched peaks (boxes = IQR, line = median; "
                   "near 0 = well calibrated)."),
             ("b", "• Ion channel names the adduct each compound was assigned on. It counts a "
                   "compound once per adduct, so bright species recur across channels — read the"),
             ("b", "  signal % on the bars (e.g. protonation usually dominates the signal)."),
             ("b", "• Unexplained peaks are a third of the m/z bins but only a few % of the signal "
                   "(dim, near-noise)."),
             ("b", "• 'By role' splits explained signal into analyte (M0 + isotopes) vs the reagent "
                   "ion vs unexplained —"),
             ("b", "  so a reagent-dominated negative-mode spectrum isn't read as fully characterised.")]
    # the NH4->amine caveat only applies to positive urea-CIMS (where NH4 adducts
    # exist); never print it on a negative-mode (e.g. Br-) report.
    pos = any(("NH4" in str(k)) or ("CH4N2O" in str(k)) for k in ctx.get("adduct_counts", {}))
    if pos:
        lines += [("dim", "[M+NH4]+ is mass/isotope-identical to [M+H]+ of the +NH3 amine -- kept as NH4 only"),
                  ("dim", "when its trace co-varies (r>=0.7) with the protonated/urea parent, else re-read as the"),
                  ("dim", "amine (a parsimony prior, not a measurement). Peak roles -> Methods page.")]
    else:
        lines += [("dim", "Peak roles are defined on the Methods page.")]
    _text_lines(fig, lines, y0=0.40, dy=0.029, bottom=0.05)
    _close(pdf, fig)


def composition(ctx, pdf):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4)            # text page FIRST, then the VK figure
    fig.text(0.08, 0.95, "Composition of the assigned peaks", fontsize=15, weight="bold", color=INK)
    comp = ctx.get("composition", {})
    het = ctx.get("hetero", {})
    scf = ctx.get("sig_comp_frac", {})
    lines = [("h", f"Distinct neutral compounds by backbone ({ctx['n_neutrals']} total)"),
             ("gap", 0.3),
             ("dim", "by COUNT — each compound once, regardless of how bright it is")]
    for kl in ("CHO", "CHON", "CHOS"):
        if kl in comp:
            lines.append(("m", f"   {kl:6s} {comp[kl]:>4}   ({_pct(comp[kl], ctx['n_neutrals']):.0f}%)"))
    if scf:
        lines += [("gap", 0.8),
                  ("h", "Same compounds, weighted by signal"),
                  ("gap", 0.3),
                  ("dim", "where the chemistry actually is — a few bright species carry most signal")]
        for kl in ("CHO", "CHON", "CHOS"):
            if kl in scf:
                lines.append(("m", f"   {kl:6s} {scf[kl]*100:>3.0f}% of assigned signal"))
    sh = ctx.get("shadow", {}); coll = ctx.get("comp_collapsed", {})
    if sh.get("n_shadowed") and ctx.get("positive"):    # the re-read is positive urea-CIMS only
        lines += [("gap", 0.8),
                  ("dim", f"Note: {sh['n_shadowed']} CHON neutrals share an exact NH3-shifted CHO twin that is"),
                  ("dim", "also assigned — mass-degenerate [M+NH4]+/[M+H]+ pairs from the amine re-read,"),
                  ("dim", f"counted twice. Read as their CHO parent: {coll.get('CHO','?')} CHO / "
                          f"{coll.get('CHON','?')} CHON ({sh['collapsed_neutrals']} distinct).")]
    lines += [("gap", 0.8),
              ("h", "Heteroatom additions (within the backbone classes above)"),
              ("gap", 0.3)]
    for k, v in het.items():
        lines.append(("m", f"   {k:24s} {v}"))
    present = "/".join(k for k in ("CHO", "CHON", "CHOS") if k in comp)
    lines += [("gap", 0.8),
              ("dim", f"Si/F/halogen are folded into the {present} backbone, not split out"),
              ("dim", "(a siloxane with no N is CHO; a fluorinated species with N is CHON)."),
              ("dim", "Si = PDMS/silicone inlet bleed; F/halogen are reagent/contaminant ladders.")]
    _text_lines(fig, lines, y0=0.89, dy=0.028)
    _close(pdf, fig)
    if "vk" in ctx["fig"]:
        _image_page(pdf, ctx["fig"]["vk"], "")          # figure carries its own title


def scrutiny(ctx, pdf):
    """Assignments flagged for chemical-plausibility scrutiny (Candidate-tier only):
    high-heteroatom mass coincidences, implausibly carbon-rich skeletons, or a
    halogen in a positive-mode neutral. Flagged, NOT removed — listed so a reader
    can discount them. Renders only when something is flagged."""
    fl = ctx.get("flagged", [])
    if not fl:
        return
    import matplotlib.pyplot as plt
    intro = [("dim", f"{len(fl)} Candidate-tier neutral(s) whose formula looks more like a mass"),
             ("dim", "coincidence than a real molecule (many heteroatoms, very low H/C, or a"),
             ("dim", "wrong-mode halogen). Flagged for review, NOT removed; Assigned (isotope-"),
             ("dim", "scored) assignments are never flagged."),
             ("gap", 0.8)]
    intro += [("dim", "score=match · ppm=mass error · iso=confirmed isotopologues (0=none) · "
                      "the arrow marks where a chemically plausible alternative existed")]
    head = [("m", f" {'score':>5} {'ppm':>6} {'iso':>3}  {'neutral':<15} why  (-> sane alternative)"),
            ("gap", 0.2)]
    MAXW = 110                                     # hard cap so no row overflows the page width
    rows = []
    for d in fl:
        sc = f"{d['ion_score']:.2f}" if d.get("ion_score") is not None else "  -  "
        ppm = f"{d['ppm']:+.2f}" if d.get("ppm") is not None else "  -  "
        iso = str(d.get("iso", "?"))
        why = d["reason"]
        if d.get("sane_alt"):
            why = f"{why}  -> {d['sane_alt']}"
        line = f" {sc:>5} {ppm:>6} {iso:>3}  {d['neutral_formula']:<15} {why}"
        rows.append(("m", line if len(line) <= MAXW else line[:MAXW - 1] + "…"))
    PER = 34                                       # paginate so long flagged lists don't clip
    npages = max(1, (len(rows) + PER - 1) // PER)
    for pi in range(npages):
        fig = plt.figure(figsize=A4)
        ttl = "Assignments flagged for scrutiny" + ("" if npages == 1 else f"  (page {pi + 1}/{npages})")
        fig.text(0.06, 0.95, ttl, fontsize=15, weight="bold", color=INK)
        block = (intro if pi == 0 else []) + head + rows[pi * PER:(pi + 1) * PER]
        _text_lines(fig, block, x=0.055, y0=0.91, dy=0.0188, size=8.0)
        _close(pdf, fig)


def reference_lists(ctx, pdf):
    """Reference-list corroboration & rescue (renders only when the run's context
    unlocked a literature peaklist and something matched). Two findings: assigned
    Candidate neutrals corroborated by a published list, and UNEXPLAINED peaks that
    match a known formula by mass under the reagent adducts (leads to verify). A
    soft prior tagged with its source — never a measurement, never overrides a tier.
    Also writes tables/reflist_matches_<tag>.csv with the full match set."""
    rl = ctx.get("reflist")
    if not rl or (not rl.get("corroborated") and not rl.get("rescue")):
        return
    import matplotlib.pyplot as plt
    corr = rl["corroborated"]; rescue = rl["rescue"]
    # full match table to disk (nothing hidden behind the page cap)
    try:
        tab = PT.run_paths(ctx["out_dir"]).ensure().tables
        rows = [{"kind": "corroborated_candidate", "obs_mz": "", "neutral_formula": f,
                 "adduct": "", "ppm": "", "list": h[0]["list"],
                 "conditions": "|".join(h[0]["conditions"])} for f, h in corr.items()]
        rows += [{"kind": "unexplained_rescue", "obs_mz": m["obs_mz"],
                  "neutral_formula": m["formula"], "adduct": m["adduct"], "ppm": m["ppm"],
                  "list": m["list"], "conditions": m.get("name", "")} for m in rescue]
        pd.DataFrame(rows).to_csv(f"{tab}/reflist_matches_{ctx['tag']}.csv", index=False)
    except Exception:
        pass

    lines = [("dim", "Known-molecule peaklists unlocked by this run's context "
                     f"({', '.join(rl['tags']) or 'none'}):")]
    for c in rl["cites"]:
        c = c if len(c) <= 92 else c[:91] + "…"      # keep the cite on one line
        lines.append(("dim", f"  • {c}"))
    lines += [("gap", 0.6),
              ("h", f"Candidate assignments corroborated: {len(corr)} of {rl['n_candidate']}"),
              ("gap", 0.25),
              ("dim", "a Candidate-tier neutral whose formula is a published product of this system"),
              ("m", "   neutral            seen-in (conditions)")]
    for f, h in sorted(corr.items())[:12]:
        lines.append(("m", f"   {f:18s} {', '.join(h[0]['conditions'])}"))
    if len(corr) > 12:
        lines.append(("dim", f"   … +{len(corr) - 12} more (tables/reflist_matches_{ctx['tag']}.csv)"))

    nbest = sum(1 for m in rescue if abs(m["ppm"]) <= 1.0)
    lines += [("gap", 0.7),
              ("h", f"Unexplained peaks matching a known formula: {len(rescue)} of {rl['n_unexplained']}"),
              ("gap", 0.25),
              ("dim", f"matched BY MASS under the reagent adducts ({nbest} within 1 ppm) — LEADS to verify,"),
              ("dim", "not assignments; isotope/co-variation confirmation still required."),
              ("m", "    obs m/z     formula      adduct        ppm   identity")]
    for m in sorted(rescue, key=lambda x: abs(x["ppm"]))[:18]:
        nm = f"  {m['name']}" if m.get("name") else ""
        lines.append(("m", f"   {m['obs_mz']:9.4f}   {m['formula']:>10}   {m['adduct']:<12} "
                           f"{m['ppm']:+.2f}{nm}"))
    if len(rescue) > 18:
        lines.append(("dim", f"   … +{len(rescue) - 18} more (tables/reflist_matches_{ctx['tag']}.csv)"))
    lines += [("gap", 0.6),
              ("dim", "Soft prior: formula membership is literature evidence, not a measurement. The"),
              ("dim", "near-0-ppm matches far exceed chance (a list this size yields ~single-digit random"),
              ("dim", "hits at this tolerance); larger-ppm matches are weaker and flagged by their ppm.")]
    # paginate so the (variable-length) corroboration + rescue tables never clip
    PER = 40
    pages = [lines[i:i + PER] for i in range(0, len(lines), PER)] or [[]]
    for pi, chunk in enumerate(pages):
        fig = plt.figure(figsize=A4)
        ttl = "Reference-list corroboration & rescue" + (
            "" if len(pages) == 1 else f"  (page {pi + 1}/{len(pages)})")
        fig.text(0.08, 0.95, ttl, fontsize=15, weight="bold", color=INK)
        _text_lines(fig, chunk, y0=0.90, dy=0.0195, size=8.5)
        _close(pdf, fig)


def gka(ctx, pdf):
    """GKA homologous-series findings — a small-multiple grid of Kendrick
    mass-defect plots, one per repeat-unit family (alkyl / oxidation / alkoxylate
    / siloxane / fluorinated), each rotated to flatten its own series into
    horizontal ladders. The print counterpart of the interactive rotating-GKA
    widget. Rendered on demand from the merged ledger (neutral_formula only)."""
    from peaky.reporting import gka_figure as GK
    merged = ctx.get("merged")
    if merged is None or "neutral_formula" not in getattr(merged, "columns", []) \
            or not len(merged):
        return
    fig_dir = ctx.get("fig_dir") or ctx["out_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, f"gka_{ctx['tag']}.png")
    GK.render_gka(merged, png, title=ctx.get("batch_name") or ctx["label"], top_chains=None)
    ctx["fig"]["gka"] = png
    _image_page(pdf, png, "")           # figure carries its own title


def qc_massdefect(ctx, pdf):
    """Two-panel mass-defect / mass-error QC figure, rendered from the BRIGHTEST
    selected sample's FULL ledger (all roles, incl iso_child + unexplained) —
    NOT the M0-only merged ledger. Panel (a): mass defect vs m/z over the five
    tier/role categories (Assigned/Candidate · parent/iso-child + unexplained);
    panel (b): ppm mass error vs m/z for the Assigned + Candidate M0 rows with a
    0-line and a linear trend (a calibration-drift read). Skips when there is no
    per-file ledger to source from."""
    from peaky.reporting import qc_figure as QC
    led = ctx.get("bright_ledger")
    if led is None or not len(led) or "mz" not in getattr(led, "columns", []):
        return
    fig_dir = ctx.get("fig_dir") or ctx["out_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, f"massdefect_masserror_{ctx['tag']}.png")
    QC.render_qc(led, png, title=ctx.get("batch_name") or ctx["label"])
    ctx["fig"]["qc"] = png
    _image_page(pdf, png, "")           # figure carries its own title


def families(ctx, pdf):
    for p in ctx["fig"].get("changing", []):
        _image_page(pdf, p, "")                         # A4-portrait page, own title
    cc = ctx.get("changing_csv")
    if cc is None or not len(cc):
        return
    import matplotlib.pyplot as plt
    from peaky.chem import chemistry as C
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.93, "Analyte families (temporal clusters)", fontsize=15, weight="bold", color=INK)
    sizes = cc.groupby("cluster").size().sort_values(ascending=False)
    big = [c for c in sizes.index if sizes[c] >= 3]      # only the PLOTTED clusters
    singletons = int((sizes < 3).sum())
    lines = [("h", f"Co-varying clusters ({len(big)} with >=3 members)"), ("gap", 0.3),
             ("dim", "members = ion channels (a neutral's [M+H]+ / cluster / adduct ions are clustered "
                     "separately — they often don't co-vary; see the per-cluster workbook for the breakdown)"),
             ("gap", 0.3),
             ("m", "  cluster   n   median O/C   top neutral formulas (by intensity)"),
             ("gap", 0.2)]
    for cid in big:
        g = cc[cc.cluster == cid]
        ocs = [cnt.get("O", 0) / cnt.get("C", 1) for cnt in
               (C.parse_formula(str(f)) for f in g["neutral_formula"]) if cnt.get("C", 0)]
        oc = np.median(ocs) if ocs else float("nan")
        top = ", ".join(g.sort_values("median_cps", ascending=False)["neutral_formula"]
                        .dropna().astype(str).drop_duplicates().head(4))
        lines.append(("m", f"  {int(cid):>5}   {len(g):>2}     {oc:>5.2f}      {top}"))
    if singletons:
        lines += [("gap", 0.5),
                  ("dim", f"+ {singletons} singleton / <3-member peaks — shown together in the "
                          "last 'remaining peaks' panel of the changing-cluster figure.")]
    _text_lines(fig, lines, y0=0.86, dy=0.026, size=9)
    _close(pdf, fig)


def changers(ctx, pdf):
    """Large standalone changes: single channels that change a lot (>= fold) with no
    family — pulled out so they're not buried in the flat panel (user-requested).
    A4-portrait pages (own title), embedded like the other cluster figures."""
    for p in ctx["fig"].get("changers", []):
        _image_page(pdf, p, "")             # fit-to-A4 (the PNG is already A4 portrait)


def _unexplained_gate_page(ctx, pdf):
    """Caption page placed JUST BEFORE the unexplained-cluster figures: spells out
    the brightness/persistence/variation gates and the live funnel, so a reader at
    the figure understands why only a fraction of the unexplained peaks are drawn
    (the rest are dim/sparse/flat and listed only in the CSV)."""
    import matplotlib.pyplot as plt
    cs = ctx.get("clusters", {})
    un = cs.get("unassigned", {}); g = cs.get("gates", {})
    if not un:
        return
    tag = ctx["tag"]
    nbins = un.get("n_ts_bins"); nany = un.get("n_unassigned_any")
    ngate = un.get("n_after_brightness_persistence"); nvary = un.get("n_varying_plotted")
    nflat = un.get("n_flat_bunched"); ncl = un.get("n_clusters")
    dropped = (nany - ngate) if (nany is not None and ngate is not None) else None
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, "Unexplained peaks — how this set is gated", fontsize=15,
             weight="bold", color=INK)
    lines = [
        ("b", f"Of {nbins} m/z bins in the time series, {nany} match no assigned species"),
        ("b", f"(M0 / isotope / reagent / artifact) within {g.get('match_tol_ppm', 8):.0f} ppm. To be TRACKED over"),
        ("b", "time a bin must additionally clear two bars:"),
        ("gap", 0.4),
        ("m", f"   • brightness:  median ≥ {g.get('unassigned_median_cps_floor', 50):.0f} cps"),
        ("m", f"   • persistence: detected in ≥ {g.get('min_trace_points', 8)} of the time points"),
        ("gap", 0.4),
        ("b", f"{ngate} bins pass. These are then split by time behaviour:"),
        ("gap", 0.3),
    ]
    nunion = un.get("n_entered_union")
    if nunion:
        lines += [
            ("m", f"   • {nunion} UNIFIED — present in ≥ {g.get('union_presence_min', 0.3):.0%} of samples and not an"),
            ("m", "                isotope satellite; these join the SAME clustering as the"),
            ("m", f"                assigned channels ({un.get('n_in_families', 0)} landed in co-varying families,"),
            ("m", "                legend '? m/z'). Isotope satellites rejected: "
                  f"{un.get('n_isotope_rejected', 0)}."),
        ]
    lines += [
        ("m", f"   • {nvary} VARYING  — a sustained change (cv ≥ {g.get('varying_cv_min', 0.3):.2f}) or a transient"),
        ("m", f"                burst (peak/median ≥ {g.get('varying_burst_range', 1.7):.1f}); drawn individually,"),
        ("m", f"                grouped into {ncl} co-varying cluster(s)."),
        ("m", f"   • {nflat} FLAT / non-varying — bunched into one faint median-only panel."),
    ]
    if dropped:
        lines += [
            ("gap", 0.6),
            ("dim", f"The {dropped} bins below the brightness/persistence bar are dim, sparse,"),
            ("dim", "near-noise peaks — little signal, no trackable shape. They are NOT plotted;"),
            ("dim", f"the full membership of the gated set is in tables/clusters_unassigned_{tag}.csv."),
        ]
    lines += [("gap", 0.8),
              ("dim", "(Parameter values are listed on the Methods page.)")]
    _text_lines(fig, lines, y0=0.88, dy=0.030, size=10)
    _close(pdf, fig)


def clusters(ctx, pdf):
    for p in ctx["fig"].get("flat", []):
        _image_page(pdf, p, "")                         # A4-portrait pages, own titles
    un = ctx["fig"].get("unassigned", [])
    if un:
        _unexplained_gate_page(ctx, pdf)                # explain the funnel first
        for p in un:
            _image_page(pdf, p, "")


def _selection_lines(ctx) -> list:
    """The Methods-page bullet for how the assigned subset was chosen, rendered
    from `batch_summary.json['selection']` (the recorded selector — never a
    hard-coded rule, so the page cannot go stale against the code). One line per
    statement; `_text_lines` wraps them to the page width."""
    if ctx.get("n_files", 1) <= 1:
        return [("b", "• Single sample assigned: no subset selection, no merge.")]
    b = (ctx.get("batch") or {}).get("selection") or {}
    if b.get("method") != "presence-cover":
        return [("b", "• Sample subset: the assigned files were merged by m/z; this run's "
                      "batch_summary.json carries no selection record.")]
    out = [
        ("b", f"• Sample selection: greedy presence set-cover — k = {b.get('k', '?')} of "
              f"{b.get('n_samples', '?')} samples cover {b.get('achieved_coverage', 0):.0%} "
              f"of the {b.get('n_bins', '?')} m/z bins present in "
              f"≥{b.get('min_prevalence', 2)} samples (no height floor; bins at "
              f"{b.get('tol_ppm', '?')} ppm, the merge tolerance), then merged by m/z — "
              "a single averaged file misses analytes present only part of the run."),
        ("b", f"• Selection stopped on '{b.get('stop_reason', '?')}' "
              f"(k_min {b.get('k_min', '?')}, k_max {b.get('k_max', '?')}, min gain "
              f"{b.get('min_gain', 0):.1%} of the bins)."),
    ]
    if b.get("stop_reason") == "k_max":
        out.append(("b", f"• WARNING: the k_max={b.get('k_max')} budget bound while the batch was "
                         f"still gaining ({b.get('next_gain', 0):.2%} of the bins per extra "
                         "sample) — coverage is incomplete; raise --k-max."))
    return out


def methods(ctx, pdf):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.93, "Methods & caveats", fontsize=15, weight="bold", color=INK)
    lines = [
        ("h", "Pipeline"), ("gap", 0.3),
        *_selection_lines(ctx),
        ("b", "• Each file: multi-pass formula assignment, server isotope-scored matching"),
        ("b", "  (match_compounds), isotope-envelope completion, calibrated tiering."),
        ("b", "• Clusters: log-correlation (raw or reagent-normalised) of the full-batch"),
        ("b", "  time series, complete-linkage at r>0.6 (signed distance keeps anti-phase apart)."),
        ("gap", 0.5),
        ("dim", f"Parameters: m/z merge tol {ctx.get('batch', {}).get('tol_ppm', 6.0)} ppm · "
                "cluster r>0.6 · amine co-variation r>=0.7."),
    ]
    g = ctx.get("clusters", {}).get("gates", {})
    if g:
        if g.get("entry_gate") == "episode":
            assigned_rule = [
                ("b", f"• An assigned channel is TRACKED only if detected (nonzero) in ≥{g.get('min_consecutive_bins', 3)} consecutive"),
                ("b", f"  time bins — a real episode, not a sporadic spike (unexplained bins ≥{g['unassigned_median_cps_floor']:.0f} cps median)."),
            ]
        else:
            assigned_rule = [
                ("b", f"• A bin/channel is TRACKED only if detected in ≥{g['min_trace_points']} time points and above the"),
                ("b", f"  brightness floor (unexplained ≥{g['unassigned_median_cps_floor']:.0f} cps median; "
                      f"assigned ≥{g['assigned_clustering_floor_cps']:.0f} cps)."),
            ]
        lines += [
            ("gap", 0.6),
            ("h", "Time-series gating (cluster & unexplained figures)"), ("gap", 0.3),
            *assigned_rule,
            ("b", f"• 'Varying' = cv ≥ {g['varying_cv_min']:.2f} OR transient peak/median ≥ {g['varying_burst_range']:.1f}; "
                  "flat traces are bunched."),
            ("b", f"• Clusters: correlation r > {g['cluster_corr_r']:.1f} (complete linkage), near-identical shapes"),
            ("b", f"  merged at r ≥ {g['merge_corr_r']:.2f}, minimum {g['min_cluster_members']} members; "
                  f"standalone ≥{g['big_change_fold']:g}× changers surfaced separately."),
            ("b", f"• Unexplained = bins matching no assigned species within {g['match_tol_ppm']:.0f} ppm "
                  "(see the gate page)."),
        ]
    lines += [
        ("gap", 1),
        ("h", "Peak roles"), ("gap", 0.3),
        ("b", "• M0 = an assigned compound's monoisotopic peak (the identification)."),
        ("b", "• iso_child = its isotope satellite (13C / 81Br / 37Cl ...)."),
        ("b", "• reagent = reagent-ion / reagent-cluster peaks (e.g. Br3-)."),
        ("b", "• artifact = instrument peaks (FT ringing / sidelobes of a bright peak)."),
        ("b", "• unexplained = no confident formula."),
        ("gap", 1),
        ("h", "Caveats"), ("gap", 0.3),
        ("b", "• Report coverage both by count and by signal — they differ a lot."),
        ("b", "• 'Unexplained' is dominated by small near-noise peaks (low signal)."),
        ("b", "• Confidence (Assigned/Candidate) applies to M0 compounds only."),
        ("b", "• Mass-degenerate peaks can flip formula across files (see jitter)."),
    ]
    rf = ctx.get("role_signal_frac", {})
    if rf.get("reagent", 0) >= 0.05:
        lines += [("b", f"• The reagent ion carries ~{rf['reagent']*100:.0f}% of the signal; "
                        "'explained signal'"),
                  ("b", "  is split into analyte vs reagent on the quality page.")]
    if any(("NH4" in str(k)) or ("CH4N2O" in str(k)) for k in ctx.get("adduct_counts", {})):
        lines += [("b", "• Positive urea-CIMS: [M+NH4]+ adducts are mass/isotope-identical to the"),
                  ("b", "  protonated +NH3 amine; uncorroborated ones are re-read as the amine"),
                  ("b", "  (co-variation r>=0.7), which raises the CHON count — a parsimony prior,"),
                  ("b", "  not a measurement (see Composition for the degeneracy it leaves).")]
    _text_lines(fig, lines, y0=0.92, dy=0.0216, size=9.5)
    if ctx.get("generated"):
        fig.text(0.08, 0.035, f"peaky · generated {ctx['generated']}", fontsize=8, color=GREY)
    _close(pdf, fig)


def assignments_table(ctx, pdf):
    """Appendix — EVERY assigned compound, with its ion channels and the isotopes
    confirmed on each. One row per (neutral, channel): a compound assigned on
    several adducts lists each channel separately, with the neutral printed once
    per group. Channels are ordered within a compound; compounds are ordered by
    neutral mass. 'isotopes' = the isotopologue labels the server confirmed for
    that channel (union across the selected files). Paginated; the full data
    is also in merged_ledger.csv + the per-file ledgers."""
    import matplotlib.pyplot as plt

    from peaky.chem import chemistry as C
    merged = ctx.get("merged")
    if merged is None or not len(merged) or "neutral_formula" not in merged.columns:
        return
    iso = ctx.get("iso_by_channel", {})
    max_h = ctx.get("max_h_by_channel", {})
    df = merged.copy()
    df["neutral_formula"] = df["neutral_formula"].astype(str)
    df["adduct"] = df["adduct"].astype(str)

    def _nmass(f):
        try:
            return C.neutral_mass(f)
        except Exception:
            return float("inf")
    nm = {f: _nmass(f) for f in df["neutral_formula"].unique()}
    df["_nm"] = df["neutral_formula"].map(nm)
    df = df.sort_values(["_nm", "mz"], kind="mergesort")

    def _cps(h):                              # compact cps: 3.3M / 120k / 840
        if h is None or not (h == h) or h <= 0:
            return "  -"
        if h >= 9.95e5:
            return f"{h / 1e6:.1f}M"
        if h >= 995:
            return f"{h / 1e3:.0f}k"
        return f"{h:.0f}"

    # fixed-width monospace columns (A4 portrait fits ~108 mono chars at size 6.5)
    NW, AW, IW = 15, 19, 36                  # neutral / adduct / isotopes field widths
    _scope = ("brightest height of the channel in ANY sample of the full batch"
              if ctx.get("max_h_scope") == "batch"
              else "brightest height of the channel across the selected samples")
    head = (f"{'neutral':<{NW}}{'m/z':>10}  {'channel':<{AW}}{'tier':<11}{'score':>6}"
            f"{'max cps':>8}{'  f':>4}  isotopes")
    rows: list = [("m", head),
                  ("dim", f"  max cps = {_scope} · f = files seen · isotopes = confirmed")]
    last = None
    for _, r in df.iterrows():
        nf, ad = r["neutral_formula"], r["adduct"]
        shown = "" if nf == last else (nf if len(nf) <= NW else nf[:NW - 1] + "…")
        last = nf
        labs = sorted(iso.get((nf, ad), ()), key=lambda s: (len(s), s))
        itxt = ", ".join(labs)
        if len(itxt) > IW:
            itxt = itxt[:IW - 1] + "…"
        sc = r.get("ion_score")
        scs = f"{float(sc):.2f}" if pd.notna(sc) else "  - "
        tier = str(r.get("tier", ""))[:10]
        nf_files = r.get("n_files", "")
        adv = ad if len(ad) <= AW else ad[:AW - 1] + "…"
        ih = _cps(max_h.get((nf, ad)))
        rows.append(("m", f"{shown:<{NW}}{float(r['mz']):>10.4f}  {adv:<{AW}}"
                          f"{tier:<11}{scs:>6}{ih:>8}{str(nf_files):>4}  {itxt}"))

    header, body = rows[:2], rows[2:]
    PER = 50
    npages = max(1, (len(body) + PER - 1) // PER)
    n_neutrals = df["neutral_formula"].nunique()
    for pi in range(npages):
        fig = plt.figure(figsize=A4)
        ttl = f"Appendix — assigned compounds: channels & isotopes"
        if npages > 1:
            ttl += f"   (page {pi + 1}/{npages})"
        fig.text(0.06, 0.965, ttl, fontsize=13, weight="bold", color=INK)
        if pi == 0:
            fig.text(0.06, 0.945, f"{len(df)} channels across {n_neutrals} distinct neutral "
                     f"compounds ({ctx['n_m0']} M0 assignments)", fontsize=9, color=GREY)
        _text_lines(fig, header + body[pi * PER:(pi + 1) * PER],
                    x=0.06, y0=0.925, dy=0.0172, size=7.0)
        _close(pdf, fig)


SECTIONS = [cover, findings, coverage, composition, scrutiny, reference_lists, gka,
            qc_massdefect, families, changers, clusters, methods, assignments_table]


def build(out_dir: str, *, tag: str, label: str, ts_path: str | None = None,
          out_pdf: str | None = None, generated: str = "", batch_name: str | None = None,
          dataset: str | None = None,
          run_id: str | None = None, sections=SECTIONS) -> str:
    """Build the PDF report for one batch run. `out_dir` holds the run artifacts.
    `batch_name` titles the report (else taken from the TS, else the reagent label).
    `run_id` (the timestamped run folder name) is stamped on the cover as the Report ID."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    ctx = load_context(out_dir, tag=tag, label=label, ts_path=ts_path,
                       generated=generated, batch_name=batch_name, dataset=dataset,
                       run_id=run_id)
    if out_pdf is None:
        # name the PDF with the Report ID when we have one, so the file is
        # self-identifying even when moved out of its run folder.
        fname = f"report_{run_id}.pdf" if run_id else f"report_{tag}.pdf"
        out_pdf = os.path.join(PT.run_paths(out_dir).ensure().report, fname)
    with PdfPages(out_pdf) as pdf:
        for section in sections:
            try:
                section(ctx, pdf)
            except Exception as e:                       # one bad section must not kill the report
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=A4)
                fig.text(0.08, 0.9, f"[section '{section.__name__}' failed: {e}]",
                         fontsize=10, color="#B00020", wrap=True)
                _close(pdf, fig)
    return out_pdf


def compress_pdf(in_pdf: str, out_pdf: str | None = None, *, max_px: int = 850,
                 quality: int = 58, min_mb: float = 2.0, log=print) -> str | None:
    """Write a size-reduced COMPANION of a report PDF, for emailing/sharing.

    The bulk of a report is the embedded figure rasters; this downsamples each to
    `max_px` on its long edge and re-encodes it as JPEG (`quality`), while leaving
    the page TEXT vector (so formulas/labels stay crisp). The original `in_pdf` is
    never modified — a sibling `<name>_compressed.pdf` is written and its path
    returned. Returns None (no-op) when: PyMuPDF/Pillow are not installed (optional
    deps — `pip install 'mascope-peaky[compress]'`), the input is already under
    `min_mb`, or anything goes wrong. Deliberately kept out of `build()` so the
    primary report stays byte-for-byte deterministic (see test_determinism)."""
    if out_pdf is None:
        out_pdf = (in_pdf[:-4] if in_pdf.lower().endswith(".pdf") else in_pdf) + "_compressed.pdf"
    try:
        if os.path.getsize(in_pdf) < min_mb * 1e6:
            return None
    except OSError:
        return None
    try:
        import io
        import fitz                      # PyMuPDF (optional)
        from PIL import Image            # Pillow (optional)
    except Exception:
        log("[report] compress skipped (install 'mascope-peaky[compress]' for PyMuPDF+Pillow)")
        return None
    try:
        doc = fitz.open(in_pdf)
        seen: set = set()
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    raw = doc.extract_image(xref)
                    im = Image.open(io.BytesIO(raw["image"]))
                    w, h = im.size
                    if max(w, h) > max_px:
                        s = max_px / max(w, h)
                        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
                    page.replace_image(xref, stream=buf.getvalue())
                except Exception:
                    pass                  # one bad image must not abort compression
        doc.save(out_pdf, garbage=4, deflate=True, clean=True)
        doc.close()
        return out_pdf
    except Exception as e:                # never let compression break a run
        log(f"[report] compress failed ({e}); keeping the full report only")
        return None
