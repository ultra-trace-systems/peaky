"""Offline smoke test for pdf_report.py — builds a PDF from a tiny synthetic run.
Run: python3 tests/test_pdf_report.py"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import pdf_report as R  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


with tempfile.TemporaryDirectory() as d:
    os.makedirs(f"{d}/per_file", exist_ok=True)
    os.makedirs(f"{d}/tables", exist_ok=True)   # cluster tables live under tables/ (paths.RunPaths)
    # minimal merged ledger (M0 rows) spanning a few classes; C10H19NO2 is the
    # NH3-shifted shadow of C10H16O2 (the ammonium/amine degeneracy), and one row
    # has formula_agree=False (drives the single-source disagreement count).
    # `admitted_by` / `occurrence` = the admission provenance the merge carries;
    # exactly one row was eligible by persistence only (drives the cover line).
    pd.DataFrame([
        dict(mz=169.1223, neutral_formula="C10H16O2", adduct="[M+H]+", tier="Assigned", ion_score=0.9, n_files=6, formula_agree=True, admitted_by="height", occurrence=0.98),
        dict(mz=200.0, neutral_formula="C10H19NO2", adduct="[M+NH4]+", tier="Candidate", ion_score=0.7, n_files=3, formula_agree=False, admitted_by="height", occurrence=0.41),
        dict(mz=183.0, neutral_formula="C9H10N2O", adduct="[M+H]+", tier="Candidate", ion_score=0.6, n_files=2, formula_agree=True, admitted_by="occurrence", occurrence=0.93),
        dict(mz=223.06, neutral_formula="C6H18O3Si3", adduct="[M+H]+", tier="Candidate", ion_score=0.5, n_files=1, formula_agree=True, admitted_by="height", occurrence=0.12),
        dict(mz=247.0, neutral_formula="C3H2F6O", adduct="[M+Br]-", tier="Assigned", ion_score=0.8, n_files=4, formula_agree=True, admitted_by="height", occurrence=0.77),
        dict(mz=400.0, neutral_formula="C9H12N4O12", adduct="[M+Na]+", tier="Candidate", ion_score=0.94, n_files=1, formula_agree=True, admitted_by="height", occurrence=float("nan")),  # N-monster, flagged
    ]).to_csv(f"{d}/merged_ledger.csv", index=False)
    # one per-file ledger with roles (drives the role breakdown); the M0 row
    # carries a tier + ppm_error + peak_id so the mass-defect/mass-error QC figure
    # (qc_massdefect) has a calibrated point and a parented iso-child.
    pd.DataFrame([
        dict(peak_id=1, mz=169.1223, role="M0", neutral_formula="C10H16O2", adduct="[M+H]+", height=10000,
             tier="Assigned", ppm_error=0.3, parent_peak_id=None),
        dict(peak_id=2, mz=170.1256, role="iso_child", neutral_formula=None, adduct=None, height=1100,
             tier=None, ppm_error=None, parent_peak_id=1),
        dict(peak_id=3, mz=250.0, role="reagent", neutral_formula=None, adduct=None, height=500,
             tier=None, ppm_error=None, parent_peak_id=None),
        dict(peak_id=4, mz=300.0, role="unexplained", neutral_formula=None, adduct=None, height=80,
             tier=None, ppm_error=None, parent_peak_id=None),
    ]).to_csv(f"{d}/per_file/s1_ledger.csv", index=False)
    pd.DataFrame({"neutral_formula": ["C10H16O2", "C10H14O2", "C9H12O2"],
                  "cluster": [1, 1, 1], "cv": [0.5, 0.6, 0.4],
                  "median_cps": [9000, 4000, 2000]}).to_csv(f"{d}/tables/clusters_changing_Ur.csv", index=False)

    ctx = R.load_context(d, tag="Ur", label="Ur⁺ CIMS")
    check("load_context: merged loaded", ctx["n_m0"] == 6, ctx.get("n_m0"))
    check("load_context: tiers counted", ctx["tiers"].get("Assigned") == 2, ctx["tiers"])
    check("load_context: composition is CHO/CHON/CHOS backbone",
          set(ctx["composition"]) <= {"CHO", "CHON", "CHOS"}, ctx["composition"])
    check("load_context: heteroatom side-counts Si + F",
          ctx["hetero"]["Si-bearing"] == 1 and ctx["hetero"]["F-bearing"] == 1,
          ctx.get("hetero"))
    check("load_context: per-adduct channel counts present", bool(ctx.get("adduct_counts")),
          ctx.get("adduct_counts"))
    check("load_context: role breakdown present", "unexplained" in ctx.get("role_count", {}),
          ctx.get("role_count"))
    check("load_context: brightest full per-file ledger retained for the QC figure",
          ctx.get("bright_ledger") is not None and "role" in ctx["bright_ledger"].columns,
          None if ctx.get("bright_ledger") is None else list(ctx["bright_ledger"].columns)[:3])
    check("load_context: max_h_by_channel = brightest per-file M0 height per channel",
          ctx.get("max_h_by_channel", {}).get(("C10H16O2", "[M+H]+")) == 10000.0,
          ctx.get("max_h_by_channel"))
    check("load_context: max_h_by_channel omits channels absent from per-file M0 "
          "(appendix renders '-')",
          ("C3H2F6O", "[M+Br]-") not in ctx.get("max_h_by_channel", {}),
          ctx.get("max_h_by_channel"))
    # max cps = the WHOLE-BATCH max (from the TS), not just the ~rep files. Write a
    # tiny TS where an UNSELECTED sample (s2) is brighter than the assigned file.
    pd.DataFrame([
        dict(sample_item_id="s1", datetime_utc="2026-06-20T00:00:00Z", mz=169.1223, height=10000.0),
        dict(sample_item_id="s2", datetime_utc="2026-06-20T02:00:00Z", mz=169.1223, height=30000.0),
    ]).to_parquet(f"{d}/mini_ts.parquet")
    ctx_ts = R.load_context(d, tag="Ur", label="Ur⁺ CIMS", ts_path=f"{d}/mini_ts.parquet")
    check("max cps is the WHOLE-BATCH max (brighter non-rep sample wins over rep file)",
          ctx_ts.get("max_h_by_channel", {}).get(("C10H16O2", "[M+H]+")) == 30000.0
          and ctx_ts.get("max_h_scope") == "batch",
          (ctx_ts.get("max_h_by_channel"), ctx_ts.get("max_h_scope")))

    check("load_context: role-signal split (analyte/reagent/unexplained)",
          set(ctx.get("role_signal_frac", {})) == {"analyte", "reagent", "unexplained"},
          ctx.get("role_signal_frac"))
    check("load_context: signal-weighted composition present",
          "CHO" in ctx.get("sig_comp_frac", {}), ctx.get("sig_comp_frac"))
    check("load_context: amine-shadow detected (C10H19NO2 = C10H16O2+NH3)",
          ctx.get("shadow", {}).get("n_shadowed") == 1, ctx.get("shadow"))
    check("load_context: two-way collapsed composition present",
          ctx.get("n_collapsed") == 1, ctx.get("n_collapsed"))
    check("load_context: single-source disagreements from formula_agree (=1)",
          ctx.get("n_disagree") == 1 and ctx.get("n_multifile") == 4,
          (ctx.get("n_disagree"), ctx.get("n_multifile")))
    check("load_context: top species by signal carries the bright CHO",
          ctx.get("top_species") and ctx["top_species"][0]["neutral_formula"] == "C10H16O2",
          ctx.get("top_species"))
    check("load_context: polarity detected positive (NH4/[M+H]+ present)",
          ctx.get("positive") is True, ctx.get("positive"))
    check("load_context: plausibility flags the Candidate N-monster only",
          [f["neutral_formula"] for f in ctx.get("flagged", [])] == ["C9H12N4O12"],
          ctx.get("flagged"))
    # findings + scrutiny sections build (degrade gracefully without an event TS)
    out_f = R.build(d, tag="Ur", label="Ur⁺ CIMS", out_pdf=f"{d}/rf.pdf",
                    sections=[R.findings, R.scrutiny])
    check("build: findings+scrutiny sections standalone OK",
          os.path.exists(out_f) and os.path.getsize(out_f) > 1500)

    # the mass-defect / mass-error QC section builds standalone + writes its figure
    out_q = R.build(d, tag="Ur", label="Ur⁺ CIMS", out_pdf=f"{d}/rq.pdf",
                    sections=[R.qc_massdefect])
    check("build: qc_massdefect section standalone OK",
          os.path.exists(out_q) and os.path.getsize(out_q) > 1500)
    check("build: qc_massdefect wrote the figure under figures/",
          os.path.exists(f"{d}/figures/massdefect_masserror_Ur.png"))

    out = R.build(d, tag="Ur", label="Ur⁺ CIMS", generated="2026-01-01")
    check("build: PDF created", os.path.exists(out), out)
    check("build: PDF non-trivial size", os.path.getsize(out) > 5000, os.path.getsize(out))
    check("build: PDF magic header", open(out, "rb").read(4) == b"%PDF")

    # a failing section must not kill the build (resilience)
    def boom(ctx, pdf):
        raise ValueError("intentional")
    out2 = R.build(d, tag="Ur", label="Ur⁺ CIMS", out_pdf=f"{d}/r2.pdf",
                   sections=[R.cover, boom, R.methods])
    check("build: resilient to a failing section", os.path.exists(out2) and os.path.getsize(out2) > 3000)

    # --- cover: the persistence-admission line on the DEFAULT config ---------------
    # batch_summary.json['admission'] stores the KNOB ('auto' by default) under
    # occurrence_min and the RESOLVED fraction under occurrence_threshold; the
    # cover once formatted the knob with :.0% -> ValueError, swallowed by build(),
    # which replaced the whole cover page. Call the section DIRECTLY (outside
    # build's try/except) so the exception, if any, surfaces here.
    import json
    from matplotlib.backends.backend_pdf import PdfPages

    def _cover_lines(ctx):
        """Render R.cover into a throwaway PDF and capture the text lines it emits."""
        captured = []
        orig = R._text_lines
        R._text_lines = lambda fig, lines, **kw: (captured.extend(lines), orig(fig, lines, **kw))
        try:
            with PdfPages(f"{d}/cover_probe.pdf") as pdf:
                R.cover(ctx, pdf)
        finally:
            R._text_lines = orig
        return [t for s, t in captured if isinstance(t, str)]

    json.dump({"admission": {"occurrence_min": "auto", "occurrence_threshold": 0.44,
                             "n_peaks": 318589, "n_persistent_peaks": 209523,
                             "n_persistent_traces": 1280, "n_spectra": 230,
                             "tol_ppm": 6.0}}, open(f"{d}/batch_summary.json", "w"))
    ctx_a = R.load_context(d, tag="Ur", label="Ur⁺ CIMS")
    check("load_context counts the persistence-only merged peaks (admitted_by == 'occurrence')",
          ctx_a.get("n_admitted_occurrence") == 1, ctx_a.get("n_admitted_occurrence"))
    try:
        lines_a = _cover_lines(ctx_a)
        pl = [t for t in lines_a if "persistence only" in t]
        check("cover renders on the default config (occurrence_min='auto') without raising",
              True)
        check("cover states the RESOLVED threshold (44%) and the knob in brackets, not 0.8",
              len(pl) == 1 and "≥44%" in pl[0] and "occurrence-min auto" in pl[0]
              and "1 of 6" in pl[0] and "80%" not in pl[0], pl)
    except Exception as e:  # noqa: BLE001
        check("cover renders on the default config (occurrence_min='auto') without raising",
              False, repr(e))
    # --- cover: the per-file persistence count, the derived floor, the traces --------
    # The merge keeps the highest tier then score, so an occurrence-admitted row almost
    # always loses to a height-admitted one in the same bin: the MERGED count (0-3 on
    # real batches) under-reports per-file admissions (14-63) by an order of magnitude
    # and on one batch suppressed the line entirely. The cover states both.
    pd.DataFrame([
        dict(peak_id=1, mz=169.1223, role="M0", neutral_formula="C10H16O2", adduct="[M+H]+",
             height=10000, tier="Assigned", ppm_error=0.3, parent_peak_id=None, admitted_by="height"),
        dict(peak_id=5, mz=183.0, role="M0", neutral_formula="C9H10N2O", adduct="[M+H]+",
             height=3, tier="Candidate", ppm_error=0.5, parent_peak_id=None, admitted_by="occurrence"),
        dict(peak_id=6, mz=311.0, role="M0", neutral_formula="C12H24O8", adduct="[M+H]+",
             height=2, tier="Candidate", ppm_error=0.4, parent_peak_id=None, admitted_by="occurrence"),
        dict(peak_id=4, mz=300.0, role="unexplained", neutral_formula=None, adduct=None, height=80,
             tier=None, ppm_error=None, parent_peak_id=None, admitted_by=""),
    ]).to_csv(f"{d}/per_file/s2_ledger.csv", index=False)
    json.dump({"admission": {"occurrence_min": "auto", "occurrence_threshold": 0.44,
                             "n_peaks": 318589, "n_persistent_peaks": 209523,
                             "n_persistent_traces": 1280, "n_spectra": 230, "tol_ppm": 6.0},
               "gate": {"x_edge": 5.0, "transient_share": {"1.0": 0.34, "5.0": 0.17},
                        "share_at_1": 0.34, "share_at_x": 0.17, "max_transient_share": 0.2,
                        "n_peaks": 318589, "bound": False, "source": "derived"},
               "traces": {"n_rows": 1303, "n_recentred": 587, "n_guarded": 40,
                          "median_abs_move_ppm": 3.56, "n_traces": 1234, "n_collapsed": 69,
                          "sigma_ppm": 3.79, "stamp_tol_ppm": 9.48}},
              open(f"{d}/batch_summary.json", "w"))
    ctx_f = R.load_context(d, tag="Ur", label="Ur⁺ CIMS", dataset="AP oxidation demo-set")
    check("load_context counts the PER-FILE persistence admissions (2 rows across the ledgers)",
          ctx_f.get("n_admitted_occurrence_files") == 2, ctx_f.get("n_admitted_occurrence_files"))
    check("load_context carries the dataset name (the reference-list unlock reads it)",
          ctx_f.get("dataset") == "AP oxidation demo-set", ctx_f.get("dataset"))
    try:
        lines_f = _cover_lines(ctx_f)
        pf = [t for t in lines_f if "persistence only" in t]
        check("cover states the per-file total AND the merged count",
              len(pf) == 1 and "2 per-file peak rows" in pf[0] and "1 of 6 merged" in pf[0], pf)
        gl = [t for t in lines_f if "brightness floor derived" in t]
        check("cover states the batch-derived floor with its transient shares",
              len(gl) == 1 and "5x the noise edge" in gl[0] and "17%" in gl[0] and "34% at 1x" in gl[0], gl)
        tl = [t for t in lines_f if t.strip().startswith("traces:")]
        check("cover states the trace reconciliation and the stamp window",
              len(tl) == 1 and "587 of 1303" in tl[0] and "69 competing labels" in tl[0]
              and "±9.48 ppm" in tl[0], tl)
    except Exception as e:  # noqa: BLE001
        check("cover renders with the gate + traces blocks without raising", False, repr(e))
    os.remove(f"{d}/per_file/s2_ledger.csv")

    # path off (occurrence_threshold None): no persistence line, no error
    json.dump({"admission": {"occurrence_min": 0, "occurrence_threshold": None,
                             "n_peaks": 0, "n_persistent_peaks": 0, "n_persistent_traces": 0,
                             "n_spectra": 230,
                             "tol_ppm": 6.0}}, open(f"{d}/batch_summary.json", "w"))
    ctx_o = R.load_context(d, tag="Ur", label="Ur⁺ CIMS")
    try:
        lines_o = _cover_lines(ctx_o)
        check("cover with the persistence path off renders without the persistence line",
              not any("persistence only" in t for t in lines_o))
    except Exception as e:  # noqa: BLE001
        check("cover with the persistence path off renders without the persistence line",
              False, repr(e))
    # a summary without an admission block at all (older run folders) is also fine
    json.dump({"selection": {}}, open(f"{d}/batch_summary.json", "w"))
    try:
        lines_n = _cover_lines(R.load_context(d, tag="Ur", label="Ur⁺ CIMS"))
        check("cover without an admission block renders (no persistence line)",
              not any("persistence only" in t for t in lines_n))
    except Exception as e:  # noqa: BLE001
        check("cover without an admission block renders (no persistence line)", False, repr(e))

    # run versioning: run_id + a date+time 'generated' on the cover (title page)
    RID = "Sample-run-Ur-CIMS_2026-06-20_143512"
    ctx_r = R.load_context(d, tag="Ur", label="Ur⁺ CIMS", run_id=RID)
    check("load_context carries run_id for the cover", ctx_r.get("run_id") == RID)
    out3 = R.build(d, tag="Ur", label="Ur⁺ CIMS", generated="2026-06-20 14:35",
                   run_id=RID, out_pdf=f"{d}/r3.pdf")
    check("build with run_id + timestamped generated -> PDF", os.path.exists(out3))
    # default PDF filename carries the Report ID (self-identifying outside its folder)
    out4 = R.build(d, tag="Ur", label="Ur⁺ CIMS", run_id=RID)
    check("default PDF filename includes the Report ID", os.path.basename(out4) == f"report_{RID}.pdf",
          os.path.basename(out4))
    try:
        import fitz  # PyMuPDF — verify the cover text if available
        cover = fitz.open(out3)[0].get_text()
        check("cover shows the Report ID (with time)", RID in cover, cover[:400])
        check("cover 'generated' carries date AND time", "2026-06-20 14:35" in cover, cover[:400])
    except ImportError:
        pass

    # --- Methods + cover render the RECORDED selector (batch_summary['selection']),
    # never a hard-coded rule: a 2-file run with a k_max-bound presence cover ---
    import json as _json
    import shutil as _sh
    _sh.copy(f"{d}/per_file/s1_ledger.csv", f"{d}/per_file/s2_ledger.csv")
    _json.dump({"selection": {"method": "presence-cover", "k": 2, "n_samples": 40,
                              "n_bins": 1234, "min_prevalence": 2, "tol_ppm": 6.0,
                              "achieved_coverage": 0.71, "stop_reason": "k_max",
                              "next_gain": 0.012, "k_min": 6, "k_max": 2, "min_gain": 0.005},
                "tol_ppm": 6.0}, open(f"{d}/batch_summary.json", "w"))
    ctx_s = R.load_context(d, tag="Ur", label="Ur⁺ CIMS")
    _mtxt = " ".join(t for _s, t in R._selection_lines(ctx_s) if isinstance(t, str))
    check("methods: selection bullet is rendered from batch_summary['selection']",
          "presence set-cover" in _mtxt and "k = 2 of 40" in _mtxt and "1234" in _mtxt
          and "71%" in _mtxt and "6.0 ppm" in _mtxt, _mtxt)
    check("methods: a k_max-bound selection is WARNED on the page",
          "WARNING" in _mtxt and "k_max=2" in _mtxt and "raise --k-max" in _mtxt, _mtxt)
    check("methods: no stale hard-coded rule text",
          "time-spaced" not in _mtxt and "max-TIC" not in _mtxt, _mtxt)
    out5 = R.build(d, tag="Ur", label="Ur⁺ CIMS", out_pdf=f"{d}/r5.pdf",
                   sections=[R.cover, R.methods])
    check("build: cover + methods with a selection record -> PDF",
          os.path.exists(out5) and os.path.getsize(out5) > 3000)
    _bs_single = {"selection": {}}
    _json.dump(_bs_single, open(f"{d}/batch_summary.json", "w"))
    _mtxt2 = " ".join(t for _s, t in R._selection_lines(R.load_context(d, tag="Ur", label="x"))
                      if isinstance(t, str))
    check("methods: a run folder without a selection record says so (no invented rule)",
          "no selection record" in _mtxt2 and "time-spaced" not in _mtxt2, _mtxt2)
    os.remove(f"{d}/per_file/s2_ledger.csv"); os.remove(f"{d}/batch_summary.json")

    # --- compress_pdf: optional size-reduced companion ---
    check("compress_pdf is a no-op when the input is already small (returns None)",
          R.compress_pdf(out4, min_mb=100.0) is None)
    try:
        import io  # noqa: F401
        import fitz
        from PIL import Image  # noqa: F401
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        big = os.path.join(d, "big.pdf")
        with PdfPages(big) as pp:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.random.rand(1600, 1600, 3))     # large embedded raster
            pp.savefig(fig, dpi=200); plt.close(fig)
        small = R.compress_pdf(big, min_mb=0.0, max_px=200, quality=40)
        check("compress_pdf shrinks an image-heavy PDF",
              bool(small) and os.path.getsize(small) < os.path.getsize(big),
              f"{os.path.getsize(big)} -> {os.path.getsize(small) if small else None}")
        check("compressed PDF is valid + keeps the page count",
              bool(small) and open(small, "rb").read(4) == b"%PDF"
              and fitz.open(small).page_count == fitz.open(big).page_count)
    except ImportError:
        pass

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
