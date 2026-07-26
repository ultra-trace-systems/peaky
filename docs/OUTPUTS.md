# Peaky — Outputs reference

Where Peaky writes things, and what each artifact is. The batch run-folder layout
is the single source of truth in [`peaky/paths.py`](../peaky/paths.py) (`RunPaths`),
shared by the writers and the report reader so the filenames can't drift.

---

## Batch run — one versioned folder per run

`peaky batch` creates **one timestamped folder** under `--out-dir`
(default `~/peaky-output`), so a re-run never overwrites a previous one:

```
<batch-slug>_<YYYY-MM-DDTHHMMSSZ>/     ← folder name == Report ID (UTC stamp)
```

### Run root (flat — read by several modules + the cross-run registry)

| Artifact | What it is / what it's for |
|---|---|
| `merged_ledger.csv` | **The result.** Every merged peak (one row each): role, neutral formula + adduct, scores, ppm, confidence, tier, provenance. The provenance anchor. |
| `run_manifest.json` | **Reproducibility manifest.** Pins the run to its exact code (package + per-module version + content hash + git commit), input-data hash (`ts_sha1`), resolved config (incl. `select` / `coverage_target`), and output hash (`merged_ledger_sha1`). |
| `batch_summary.json` | Run counts + per-file calibration offsets (M0/tier counts, n_files, offsets) and the selection strategy used (`select`, `coverage_target`). |
| `per_file/<sid>_ledger.csv` | The full single-sample ledger for **each** assigned sample, kept for audit / re-merge. |
| `index.jsonl` | **At the `--out-dir` base, not inside the run folder.** Cross-run registry — one compact row per run, loadable with `pandas.read_json(lines=True)` to find or diff runs. |

### `figures/` — all `.png`

| Artifact | What it is |
|---|---|
| `van_krevelen_<tag>.png` | Van Krevelen of the assigned analytes (Si excluded — the clean atmospheric view). |
| `van_krevelen_full_<tag>.png` | Full Van Krevelen — every assigned peak by CHO/CHON/CHOS backbone (Si/F/halogen folded in). |
| `clusters_changing_<tag>_p*.png` | Correlation-cluster panels (A4 portrait, paginated) of the dynamic, co-varying analyte families. |
| `clusters_flat_<tag>_p1.png` | The uncorrelated/flat remainder + Si contamination, bunched into one overview. |
| `clusters_changers_<tag>_p*.png` | Big standalone changers — single channels that move ≥~5–10× on their own. |
| `clusters_unassigned_<tag>_p*.png` | The same clustering applied to the **unexplained** residual. |
| `gka_<tag>.png` | The static GKA findings page embedded in the report. |

### `tables/` — all `.csv` / `.xlsx`

| Artifact | What it is |
|---|---|
| `selected_samples.csv` | Which samples were assigned + why: a `role` (`time-grid` / `max-TIC`, or `coverage-winner` for `--select brightest`) and `bins_won` (significant m/z bins the sample is brightest for). |
| `jitter.csv` | Per-(cluster, file) mass-jitter table — raw vs calibration-adjusted ppm spread of each merged assignment. |
| `van_krevelen_full_<tag>.csv` | The full-VK data behind the figure (one row per assigned neutral). |
| `clusters_changing_<tag>.csv` / `.xlsx` | Cluster membership; the XLSX has one tab per cluster (formula / channel / m/z / match_score / tier). |
| `clusters_flat_<tag>.csv`, `clusters_changers_<tag>.csv`, `clusters_unassigned_<tag>.csv` | Membership for the flat / changers / unassigned figures. |
| `channel_agreement_<tag>.csv` | QC: how often a multi-channel neutral's ion channels agree in time. |
| `plausibility_audit_<tag>.csv` | One row per peak the hardened plausibility layer touched (demoted or relabelled): `before_tier`, `after_tier_or_role`, the `reason`, the supporting `evidence` (O/C or DBE/C or series r), the `degeneracy_note`, and `n_iso`. Always written (header-only when nothing was touched). |

### `report/` — the PDF

| Artifact | What it is |
|---|---|
| `report_<run-id>.pdf` | **The standard iterable A4 report** (cover · findings · coverage · composition · scrutiny · GKA · families · changers · clusters · methods). The cover shows the Report ID + a date+time "generated" line. |
| `report_<run-id>_compressed.pdf` | Optional size-reduced companion for emailing (needs `pip install mascope-peaky[compress]`). The full report is left byte-for-byte untouched. |

### `data/` — bulky inputs kept with the run

| Artifact | What it is |
|---|---|
| `<tag>_ts.parquet` | The full-batch per-sample peak time series — written here **only** when fetched live (no on-disk source). A parquet passed by `--ts` is *referenced*, never copied. |

---

## The time-series parquet — schema

Two files carry the batch time series. Both are **one row per (sample × peak)** —
a long/tidy table, not a matrix — so a 995-sample batch with ~2400 peaks each is
~2.4 M rows.

| File | Content |
|---|---|
| `per_file/_batch_ts.parquet` | The **annotated** series: raw peaks **+ the assignment columns** below. This is the one to hand to downstream software. |
| `data/<tag>_ts.parquet` | The **raw** series only (no assignment columns), kept when the TS was fetched live. |

**Raw columns** (from Mascope; exactly which are present depends on how the TS was
fetched — `sample_item_id`, `mz` and `height` are always there):

| Column | Arrow type | Meaning |
|---|---|---|
| `sample_batch_name` | `large_string` | Batch the sample belongs to. |
| `sample_item_id` | `large_string` | **Sample key** — one acquisition file. |
| `sample_item_name` | `large_string` | Human-readable sample name. |
| `datetime_utc` | `timestamp[us, tz=UTC]` | Acquisition time — the x-axis of every trace. |
| `peak_id` | `large_string` | Mascope's per-sample peak id. Unique *within* a sample; **not** stable across samples, so it cannot be used to join a peak to the same peak in another file. |
| `mz` | `double` | The peak's **raw fitted** m/z **in that sample** — it jitters sample to sample and is *not* the calibrated mass. |
| `height` | `double` | Peak height (cps) — the quantity to plot. |
| `area` | `double` | Integrated peak area. |
| `sparsity` | `double` | Mascope peak-shape/quality metric. |

**Assignment columns** (added by `timeseries.annotate_peaks`, `_batch_ts.parquet`
only). All are `<NA>`/`NaN` on a peak that matched no assigned ion — which is
normal: ~29 % of raw peaks are unassigned:

| Column | Arrow type | Meaning |
|---|---|---|
| `neutral_formula` | `large_string` | Assigned neutral formula, e.g. `C8H4O3`. |
| `adduct` | `large_string` | Ionisation channel, e.g. `[M+H]+`, `[M+I]-`. |
| `tier` | `large_string` | `Assigned` (trust it) or `Candidate` (tentative). |
| `ion_mz` | `double` | The **calibrated ledger m/z** this peak was matched to. Join key: all rows sharing an `ion_mz` are the same ion. |
| `dup_candidate` | `bool` | `True` for a peak that fell inside an ion's mass window but **lost** the one-to-one contest. Its four columns above stay `<NA>`. An audit trail — the row is never dropped. |
| `intensity_suspect` | `bool` | **Trust the formula, do not quantify this channel.** The ion's m/z lands on the ringing sidelobe of a saturating neighbour, so the height here is the neighbour's, not the analyte's. Carried from the merged ledger's own column. |

### The one-to-one guarantee

Within one sample, a given `(neutral_formula, adduct)` is stamped on **at most one
peak**. So

```python
df = pd.read_parquet("per_file/_batch_ts.parquet")
trace = (df[df.neutral_formula.notna()]
           .groupby(["neutral_formula", "adduct", "datetime_utc"])["height"].sum())
```

yields exactly one point per ion per sample — no double counting.

This has to be enforced because the stamp is a **mass match**, not a peak-identity
join: the merged ledger holds one row per ion with no peak ids, and only ~6 of a
batch's samples are ever assigned, so the other ~989 have no per-peak decision to
carry over. Left unconstrained the match is many-to-one — a shoulder or split peak
inside the same window gets stamped with the same formula as the real peak
(measured: 2385 duplicated (sample, ion) pairs, 61 ions, on a 2.4 M-row uronium
batch). The **assignment itself never does this** (verified: 9784 per-file M0 keys,
zero owned by more than one peak — the shoulder is left `unexplained`), so the
duplication was purely an artifact of the re-match. Two rules restore it:

1. **One-to-one** — per `(sample, ion)` keep the single best peak; the rest get
   `dup_candidate = True`.
2. **Consensus** — "best" means nearest the ion's *consensus* m/z, not the bare
   ledger mass. Without this the winner flips between two raw tracks sample by
   sample — whichever happens to be present — splicing two different peaks into one
   trace (measured: 232 and 378 flips for two ions). The consensus is built by
   splitting an ion's candidates into tracks (a gap wider than `halfwin` starts a
   new one) and picking one by two rules:
   - **A track is scored by its BRIGHTEST member, not its summed height.** Summing
     conflates brightness with prevalence, and an **FT ringing sidelobe** of a bright
     neighbour is ubiquitous-but-dim — it recurs beside its parent in *every* sample.
     Summed height handed `C12H19NO6 [M+H]+` to its sidelobe track (1576 cps × 559
     samples) over the real peak (2390 cps × 70).
   - **The ledger mass is anchored.** Offset 0 is where the *assignment* committed
     the formula, so the track holding it is displaced only by one at least
     `ANCHOR_MARGIN` (2×) brighter. `C19H34O6Si [M+NH4]+` clears that bar (1205 vs
     473 cps) and correctly moves; `C14H28O3Si [M+H]+` at 1.86× does not.

Both default on; `annotate_peaks(..., one_to_one=False)` / `consensus=False` restore
the raw behaviour.

> **Known residual — a stamp is a mass match, not proof of identity.** Where a
> sample's real peak is **absent**, a neighbour inside the tolerance still collects
> the stamp, and that neighbour is sometimes an FT ringing sidelobe. Scale on the
> Wind-zone-2 batch: **0.28 %** of stamped rows (4777 of 1.73 M) sit >1 mDa from
> their ledger mass, and **10 ions of 2127** span more than 0.5 mDa across the
> batch. The worst of these — where the channel's whole intensity is a
> neighbour's sidelobe — are now detected and marked `intensity_suspect`
> (see below); the rest are visible as a wide `(mz - ion_mz)` spread.

### Sidelobe-contaminated channels (`intensity_suspect`)

A saturating peak **rings**: FT/Gibbs sidelobes sit a few mDa either side of it at
a roughly fixed fraction of its height. When an assigned ion's m/z lands on one,
the *formula* can still be right while the *height* is the neighbour's.
`C18H30O6` is the worked example — clean on `[M+H]+` at m/z 343.211, but its urea
adduct at 403.244 rides 11.5 mDa from a 520 000-cps `C20H34O8` peak at a locked
0.71 % of it. A trace built from that channel tracks `C20H34O8`, not `C18H30O6`.

**Static features cannot detect this.** Over 25 498 raw tracks across 30 campaign
runs, contaminated channels are *indistinguishable* from real ions that merely sit
near a bright peak:

| | contaminated | real, near a bright peak |
|---|---|---|
| satellite fraction of parent | 0.69 % | 0.23 % *(smaller!)* |
| \|Δm/z\| to parent | 11.5 mDa | 10.1 mDa |
| **ratio-to-parent cv (time series)** | **0.033 – 0.051** | **0.21 – 1.09** |

Only the time series separates them: a sidelobe holds a near-constant ratio to its
parent; an independent ion varies on its own. So `timeseries.flag_sidelobe_channels`
runs at **merge level**, where the batch TS exists — not in per-file cleanup — and
sets `intensity_suspect` plus `sidelobe_parent_mz` on the merged ledger.
`SIDELOBE_CV = 0.08` sits in the empty gap, biased to under-flag. Scored against
that labelled set: **6/6 contaminated channels caught, 0 false positives of 72**.

The assignment is **never** altered — no retraction, no tier change — because the
neutral is usually real and corroborated on another channel. Only quantification is
in question:

```python
df = pd.read_parquet("per_file/_batch_ts.parquet")
quant = df[df.neutral_formula.notna() & ~df.intensity_suspect]   # safe to integrate
```

---

## Single-sample run — `peaky assign`

Writes into `--output-dir` with the prefix `<sample-id>_<YYYYMMDD-HHMM>`:

| Artifact | What it is |
|---|---|
| `<prefix>_ledger.csv` | Every peak: role (`M0` / `iso_child` / `reagent` / `artifact` / `unexplained`), formula, adduct, all scores (incl. arbitration `eff_score`/`eff_margin`/`tied`), ppm, confidence, `tier` + `tier_reason`, candidate/degeneracy density, provenance, commentary, alternatives, isotopologues. |
| `<prefix>_assignments.xlsx` | The styled multi-sheet workbook: Summary · Read-me legend · **Assigned** · **Candidates** · Unassigned (evidence-characterized) · By class · Unique formulas · Isotopologues · Peak ownership · Target list · Reagent ions. Frozen headers, autofilters, tier/confidence color chips. |
| `<prefix>_summary.md` | Narrative + top assignments + coverage. |
| `<prefix>_manifest.json` | Module versions, prescan fingerprint, series-evidence table, per-pass timing. |
| `<prefix>_gka.html` | Interactive rotating-GKA widget (self-contained, no server). |
| `<prefix>_gka_unexplained.html` | The same widget over the **unexplained residual only** — the place to hunt for missed homologous structure. |
| `checkpoints/` | Per-pass ledger checkpoints (an audit trail of what each pass committed). |

---

## Reproducibility note

Every figure, table, and ledger above is a **pure function of the input data** —
byte-identical whenever you re-run the same data. The **only** thing the run
timestamp changes is the PDF cover's "generated" line + the Report ID + the
run-folder name + `run_manifest.json`. See
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-reproducibility--provenance).

> `peaky report --run-dir <folder> ...` regenerates the `figures/` + `report/`
> artifacts of an existing run **offline** (no assignment, no network) from the
> ledgers already on disk + the TS parquet.
