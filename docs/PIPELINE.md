# Peaky — Pipeline index ("how the numbers are transformed")

This is the index to Peaky's **per-module deep-dives** — one doc per pipeline
module, each tracing *inputs → stages (with the exact constants/thresholds) →
outputs → gotchas* for that module. Read them in the order below to follow a
number from a Mascope peak all the way to a figure.

For the design and the one-ledger model see [`ARCHITECTURE.md`](ARCHITECTURE.md);
for what assignment *produces* (for a scientist) see [`ASSIGNMENT.md`](ASSIGNMENT.md);
for where every artifact lands see [`OUTPUTS.md`](OUTPUTS.md).

```
INGEST       Mascope  → peak / score DataFrames     DATA_IO · SCORING
CHEMISTRY    m/z      → candidate neutral formulas  CHEMISTRY · ISOTOPES · REAGENTS
ASSIGNMENT   candidates → committed formulas        ASSIGNMENT (+ ASSIGNMENT_DETAIL)
BATCH        many samples → one merged ledger       SAMPLING · MERGE · TIMESERIES
CLUSTERING   time series → co-varying families      CLUSTERING
REPORTING    ledger → figures + PDF                 VANKREVELEN · GKA · QC_AND_REPORT
```

---

## 1. Ingest — Mascope → DataFrames

| doc | module | transform |
| --- | --- | --- |
| [`DATA_IO.md`](DATA_IO.md) | `io/io_mascope.py` | the only Mascope I/O: pull + cache peaks / batch time series, detect adducts, estimate offset, flatten the score tree |
| [`SCORING.md`](SCORING.md) | `io/local_scoring.py` | in-process IsoSpec + `score_pattern_v2` — a candidate `(neutral × adduct)` → a 0–1 match score at the sample's own mass width (default backend; `PEAKY_LOCAL_SCORING`) |

## 2. Chemistry — m/z → candidate formulas

| doc | module | transform |
| --- | --- | --- |
| [`CHEMISTRY.md`](CHEMISTRY.md) | `chem/chemistry.py` | exact masses, integer-DBE / Senior / oxygen-cap gates, the valence-legal formula grid, the complexity penalty |
| [`ISOTOPES.md`](ISOTOPES.md) | `chem/isotopes.py` | isotopologue-envelope prediction + the prescan that shrinks the grid from satellite-pair evidence |
| [`REAGENTS.md`](REAGENTS.md) | `chem/reagents.py` + `chem/profiles.py` | the reagent profile (the single mode switch) + the labeled reagent-ion cluster library |

## 3. Assignment — candidates → committed formulas

*Already documented — not duplicated here.*

| doc | module | transform |
| --- | --- | --- |
| [`ASSIGNMENT.md`](ASSIGNMENT.md) | `assignment/*` | what assignment does + what the results mean (for a scientist) |
| [`ASSIGNMENT_DETAIL.md`](ASSIGNMENT_DETAIL.md) | `assignment/*` | the full pass-by-pass internals (arbitration, tiers, cleanup, degeneracy) |

## 4. Batch — many samples → one merged ledger

| doc | module | transform |
| --- | --- | --- |
| [`SAMPLING.md`](SAMPLING.md) | `batch/sampling.py` | which real samples get assigned: time-representative or brightest-coverage set-cover |
| [`MERGE.md`](MERGE.md) | `batch/assign_batch.py` | assign each rep file, then offset-aware m/z merge + the file-to-file jitter accounting |
| [`TIMESERIES.md`](TIMESERIES.md) | `batch/timeseries.py` | the samples × m/z-bin matrix, reagent-normalised variability, and TS disposition |

## 5. Clustering — time series → co-varying families

| doc | module | transform |
| --- | --- | --- |
| [`CLUSTERING.md`](CLUSTERING.md) | `batch/clustering.py` + `batch/cluster.py` | per-channel traces → log-shape correlation → complete-linkage families (changing / changers / flat) |

## 6. Reporting — ledger → figures + PDF

| doc | module | transform |
| --- | --- | --- |
| [`VANKREVELEN.md`](VANKREVELEN.md) | `reporting/analyte_viz.py` | the analyte table + H/C–O/C Van Krevelen + per-ion/compound RAW time traces |
| [`GKA.md`](GKA.md) | `reporting/gka_figure.py` + `assignment/series_gka.py` + `series_detect.py` | Kendrick/GKA series: anchor propagation, decoy-controlled detection, the KMD figure |
| [`QC_AND_REPORT.md`](QC_AND_REPORT.md) | `reporting/qc_figure.py` + `reporting/pdf_report.py` | the mass-defect/mass-error QC figure + the iterable A4 PDF assembly |
