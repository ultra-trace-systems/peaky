# Peaky — QC figure & PDF report assembly

This document explains **two reporting transforms**: the two-panel mass-defect /
mass-error QC figure (numbers read straight off a full ledger), and how the
iterable A4 PDF is assembled from a run's on-disk artifacts — including the
*honest "explained"* signal-share accounting. It is a module deep-dive companion
to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline, §7–8),
[`OUTPUTS.md`](OUTPUTS.md) (every artifact the report reads), and
[`ASSIGNMENT.md`](ASSIGNMENT.md) (the tiers it visualises).

**Code:** `peaky/reporting/qc_figure.py` (the QC figure) and
`peaky/reporting/pdf_report.py` (the report spine: `load_context` → `SECTIONS` →
`build`). Both are pure functions of their inputs; matplotlib stamps the fixed
`SOURCE_DATE_EPOCH`, so the primary report is byte-deterministic.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `qc_figure.py` / `pdf_report.py`; if you change one there, change it
> here.

---

## 1. What this stage does

- **QC figure** — from one **full per-sample ledger** (every role, including
  iso-children and the unexplained residual), draw (a) a **mass-defect map** split
  into five tier/role categories and (b) a **ppm mass-error vs m/z** plot of the
  Assigned + Candidate M0 rows with a fitted **calibration-drift trend**.
- **PDF report** — load the run's artifacts once into a `ctx` dict (the only place
  numbers are derived), then draw an ordered list of `SECTIONS`, each a
  `section(ctx, pdf)` function. The genuine accounting transform here is the
  **role signal share** that keeps "98 % explained" from masquerading as "98 %
  analyte characterised".

```
QC FIGURE  (qc_figure.py)               PDF REPORT  (pdf_report.py)
brightest full per-sample ledger        run artifacts (merged_ledger.csv,
   │ split_categories (5 tier/roles)      per_file/*, figures/, *_summary.json)
   │ mass_defect = mz − round(mz)          │ load_context  → ctx (all derived numbers)
   ▼                                       ▼
 panel (a): defect vs m/z              SECTIONS: cover · findings · coverage ·
 panel (b): ppm vs m/z + polyfit        composition · scrutiny · reference_lists ·
   trend (slope mppm/Th, intercept)      gka · qc_massdefect · families · changers ·
   ▼                                      clusters · methods · assignments_table
 gka_… / qc_… PNG                         │ build → PdfPages (one bad section ≠ fatal)
                                          ▼  compress_pdf (optional companion)
                                        report_<run-id>.pdf
```

---

## 2. Inputs

- **QC figure** — a FULL per-sample ledger (all roles). The report passes the
  **brightest** representative ledger (`ctx["bright_ledger"]`, max total height).
- **PDF** — `out_dir` (the run folder): `merged_ledger.csv`, `per_file/*_ledger.csv`,
  `tables/selected_samples.csv`, the `figures/`, and the `*_summary.json` files.
  `batch_name`, `run_id`, `generated` text are passed in.

---

## 3. The transformation, stage by stage

### A. The QC figure (`qc_figure.py`)

1. **Mass defect** (`mass_defect`). `md = mz − round(mz)` ∈ [−0.5, 0.5).

2. **Split categories** (`split_categories`). A full ledger → five panel-(a) sets:
   the grey **`unexplained`** cloud, **Assigned·M0** / **Candidate·M0** (split by
   the M0 row's tier), and **Assigned·iso-child** / **Candidate·iso-child** (each
   iso-child inherits its **parent's** tier via `parent_peak_id`). Rows whose tier
   is neither Assigned nor Candidate are dropped from the coloured sets (the
   unexplained cloud is unaffected). `_norm_tier` tolerates the legacy `Identified`
   spelling.

3. **ppm points** (`ppm_points`). Assigned + Candidate M0 rows carrying a finite
   mass error → panel (b). The panel plots **`ppm_error_cal`** (the *calibrated*
   ppm) when that column is present, falling back to raw `ppm_error` otherwise.
   `ppm_error_cal = ppm_error − cal_mu`, where `cal_mu` is the robust **offset**
   (median) of the corroborated CHO/CHON core from `tiers._calibrate` — so the
   cloud is centred on the run's own drift rather than on 0. Raw `ppm_error` is
   kept alongside for provenance.

4. **Render + trend** (`render_qc`). Panel (a) draws the categories in `CATEGORIES`
   order (grey cloud underneath, `zorder` 1; coloured assignments on top). Panel
   (b) plots the calibrated ppm vs m/z with a 0-line and, when ≥ 2 points span m/z,
   a `np.polyfit` degree-1 **linear trend** — the annotation reports
   `slope·1000` **mppm/Th**, the intercept (ppm), and the median. Because the
   points are already offset-corrected, a well-calibrated run sits on the 0-line:
   e.g. the brightest Ur file's Assigned ppm median moves +0.125 → +0.011 (mean
   +0.121 → +0.007) and |ppm| > 0.5 outliers drop 133 → 118; Br's median moves
   −0.13 → ~+0.02. A residual slope/offset is the calibration-drift read the tier
   engine self-calibrates from.

### B. The PDF report (`pdf_report.py`)

5. **Load context once** (`load_context`). Read the artifacts and derive every
   number a section needs: `n_m0`, `tiers`, `n_neutrals`, **`composition`** (by
   CHO/CHON/CHOS backbone), `hetero` side-counts (Si/F/Cl-Br bearing),
   `score_by_tier`, `adduct_counts`, the representative `samples`, pooled
   `role_count`, the **`bright_ledger`** (max-height full ledger → the QC figure),
   `role_signal`, **`role_signal_frac`**, `neutral_signal`, `expl_mz`,
   `iso_by_channel`, `ppm_by_cat`, `adduct_signal`, and the reference-list priors.

6. **Honest "explained" share** (`role_signal_frac`). Split total signal into
   `analyte = (M0 + iso_child)/total`, `reagent = reagent/total`,
   `unexplained = (unexplained + artifact)/total`. A Br⁻ spectrum is mostly the
   reagent ion, so the coverage headline reports analyte share separately —
   "explained" by count is *not* "analyte characterised" by signal.

7. **Draw the sections** (`SECTIONS`, `build`). 13 ordered sections (cover →
   `assignments_table`). `build` iterates them into a `PdfPages`; a section that
   throws is caught and rendered as a one-page error note — **one bad section never
   kills the report**. The `qc_massdefect` section calls `render_qc` on
   `bright_ledger`; the `gka` section embeds `gka_<tag>.png`. Figures embed via
   `_image_page` (fit-to-A4 at `dpi` 200, or `native=True` → page = image size at
   `src_dpi` 170 for tall cluster panels).

8. **Optional compressed companion** (`compress_pdf`). Downsample each embedded
   raster to `max_px` (850) on its long edge, re-encode JPEG at `quality` 58 (text
   stays vector); skipped if the input is under `min_mb` (2.0) or PyMuPDF/Pillow
   aren't installed. Deliberately **outside `build`** so the primary report stays
   byte-for-byte deterministic.

---

## 4. Constants reference

`qc_figure.py` + `pdf_report.py`.

| constant | value | role |
| --- | --- | --- |
| `CATEGORIES` | 5 tier/role tuples | panel-(a) sets, colours, markers, draw order (grey cloud `zorder` 1) |
| `PPM_TIER_COLORS` | Assigned / Candidate | panel-(b) colours |
| `mass_defect` range | [−0.5, 0.5) | `mz − round(mz)` |
| `cal_mu` | per-run (ppm) | robust offset of the corroborated CHO/CHON core (`tiers._calibrate`); `ppm_error_cal = ppm_error − cal_mu` |
| `CAL_SIGMA_FLOOR` | 0.15 | floor on the core's `1.4826·MAD` sigma (`tiers._calibrate`) |
| `CAL_MIN_N` | 20 | min core peaks required before an offset is fit |
| `A4` | (8.27, 11.69) in | portrait page size |
| `SECTIONS` | 13 functions | the ordered report spine (cover…assignments_table) |
| `_image_page` `dpi` / `src_dpi` | 200 / 170 | fit-to-A4 raster dpi / native page-size dpi |
| `_ISO_C13` | 1.0033548 | ¹³C−¹²C, to recover an isotopologue's mass error |
| `compress_pdf` `max_px` | 850 | long-edge pixel cap when downsampling figures |
| `compress_pdf` `quality` | 58 | JPEG quality of the compressed companion |
| `compress_pdf` `min_mb` | 2.0 | skip compression below this input size |

---

## 5. Metrics, defined

- **mass defect** — `mz − round(mz)`; the y-axis of panel (a). Assigned points
  trace the CH/CHO band; the grey cloud shows where unexplained signal sits.
- **ppm trend** — degree-1 `polyfit` of the plotted ppm (`ppm_error_cal` when
  present, else `ppm_error`) vs `mz`: `slope` (reported as mppm/Th), `intercept`
  (ppm at m/z 0), and the `median` ppm; the calibration-drift diagnostic.
- **`ppm_error_cal`** — `ppm_error − cal_mu`, the offset-calibrated mass error
  panel (b) prefers; raw `ppm_error` is retained for provenance.
- **`role_signal_frac`** — signal share split analyte / reagent / unexplained; the
  honest coverage number behind the headline.
- **`composition`** — neutral counts by CHO/CHON/CHOS backbone (Si/F/halogen folded
  in, via `analyte_viz.backbone_class`).
- **`score_by_tier` / `adduct_signal`** — mean `ion_score` per tier; signal share
  per ion channel.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `figures/qc_<tag>.png` (via `qc_massdefect`) | two-panel mass-defect + ppm-error QC figure |
| `figures/gka_<tag>.png` | the GKA findings page ([`GKA.md`](GKA.md)) embedded by the `gka` section |
| `report/report_<run-id>.pdf` | the standard iterable A4 report (13 sections) |
| `report/report_<run-id>_compressed.pdf` | optional size-reduced companion (full report untouched) |

---

## 7. Properties, invariants & gotchas

- **The QC figure reads the FULL ledger, the GKA figure the MERGED (M0-only) one.**
  QC shows the whole spectrum at once (iso-children + the unexplained cloud); GKA
  shows only committed neutrals.
- **iso-children inherit the parent's tier** — the tier lives only on M0 rows, so a
  satellite is coloured by the M0 it points at.
- **The report is iterable by design.** Add/reorder a `section(ctx, pdf)` and list
  it in `SECTIONS`; nothing else couples. `ctx` is loaded once and sections degrade
  gracefully when an artifact is missing.
- **One bad section is non-fatal** — it becomes an error page, so a partial run
  still yields a report.
- **Honest coverage.** Count-coverage and signal-coverage differ sharply on
  reagent-CIMS; `role_signal_frac` is why the report never conflates "explained"
  with "analyte characterised".
- **Determinism.** Both figures and the primary PDF are pure functions of the
  inputs with a fixed content epoch, so a re-run is byte-identical; `compress_pdf`
  is kept out of `build` to preserve that (see
  [`ARCHITECTURE.md §7`](ARCHITECTURE.md#7-reproducibility--provenance)).
- **Compression is best-effort** — no optional deps, or an under-`min_mb` input, is
  a silent no-op that never touches the full report.
- **Calibration is offset-only, and display/provenance only.** `cal_mu` is a single
  robust *median* offset — a per-file **linear (slope)** term was tested and
  rejected (the slope is tiny; on Br it merely widens the window). The calibrated
  ppm changes only what panel (b) plots and the `ppm_error_cal` column; it drives
  **no tier changes** (tiering was already calibration-aware, centring its z-gate
  on the same robust median rather than 0).
- **The QC figure is one file, not the whole run.** It is always sourced from the
  single **brightest** representative ledger (`ctx["bright_ledger"]`,
  `pdf_report.py`), so `cal_mu` and every panel-(b) number are that one file's
  drift — not a run-wide average.

---

## 8. Code map

| function | role |
| --- | --- |
| `qc_figure.mass_defect` | `mz − round(mz)` |
| `qc_figure.split_categories` | full ledger → the five tier/role panel-(a) sets |
| `qc_figure.ppm_points` | Assigned+Candidate M0 with finite ppm → panel (b) |
| `qc_figure.render_qc` | draw both panels + the calibration-drift trend |
| `pdf_report.load_context` | read run artifacts → the `ctx` of derived numbers |
| `pdf_report.SECTIONS` / `build` | the ordered spine; assemble the PdfPages (fault-tolerant) |
| `pdf_report.cover…assignments_table` | the individual page builders |
| `pdf_report._image_page` | embed a PNG (fit-to-A4 or native page size) |
| `pdf_report.compress_pdf` | optional raster-downsampled email companion |
