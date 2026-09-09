# Peaky — Scoring (in-process IsoSpec + `score_pattern_v2`)

This document explains **how a candidate neutral formula gets a match score** —
the in-process backend that replaced the network `match_compounds` round-trip.
It is a module deep-dive companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the
whole pipeline), [`DATA_IO.md`](DATA_IO.md) (which dispatches here from
`score_candidates`), and [`ASSIGNMENT.md`](ASSIGNMENT.md) (which consumes the
scores). The cardinal rule still holds: **Mascope is the scorer.** This module
runs Mascope's *own* released maths (`mascope_tools.composition`) locally; Peaky
never invents a mass or isotope score.

**Code:** `peaky/io/local_scoring.py` (`score_candidates_local`, the engine);
the dispatch + the `PEAKY_LOCAL_SCORING` switch live in
`peaky/io/io_mascope.py` (`score_candidates` / `_score_candidates_local`).

> Keep this in sync with the code. Every threshold below is a named constant in
> `local_scoring.py`; if you change one there, change it here.

---

## 1. What this stage does

Given a sample's peaks and a list of candidate **neutral** formulas, score each
`(neutral × adduct)` ion against the spectrum and return one row per predicted
isotopologue — the **same schema** `io_mascope.flatten_match_tree` produces, so
the passes downstream don't know or care which backend ran.

The motivation is structural: the backend `match_compounds` is a *deep-annotation*
primitive — per `(formula × adduct)` it computes the full theoretical envelope and
returns the whole matched-**and-unmatched** tree, `O(candidates × adducts ×
envelope)` work and tens of thousands of rows for an `O(matches)` signal. That
drove the timeouts and OOM. The local path computes the identical envelope with
**IsoSpec** (`predict_isotopes`), scores it with Mascope's own
**`score_pattern_v2`** — a Gaussian mass likelihood at the sample's fitted width
times an intensity likelihood at the peak's own signal-to-noise, charging a
predicted line that is absent where the noise says it should have been visible —
and emits **only matched isotopologues** — no network, no 30k-row trees.

What a sample is judged at arrives as a **`PatternScoring`**
(`io_mascope.scoring_for_sample`): the width and offset fitted from the sample's
own anchors, and the line-matching window of its instrument class. Its
predecessor `score_pattern` scaled the mass term by a fixed 5 ppm and averaged
its terms over the lines a candidate *matched*, so an envelope that predicted
three lines and found one cost nothing — and on a TOF, whose ordinary error is a
whole Orbitrap window, every candidate scored near zero and a reference run
committed almost nothing.

```
sample peaks (mz, height, peak_id)        candidate neutral formulas
        │  dedup on peak_id, sort by mz            │
        └──────────────┬───────────────────────────┘
                       ▼   for each (neutral × adduct)
        combine_formula_and_ionization → ion formula
                       ▼
        predict_isotopes (IsoSpec) → (pred_mz, pred_int, labels)
                       ▼   anchor_on_monoisotopic: the ion's own line first
                       ▼   pred_rel = pred_int / pred_int[0]
        match M0 in ±scoring.mz_tolerance_ppm  ── not found ──► drop candidate
                       │ found
                       ▼   match each isotope; keep if intensity-error ≤ tol
        score_pattern_v2(obs_ppm − mu, obs_int, obs_snr, pred_rel, sigma_ppm)
                       ▼   one score per ion, copied onto every iso row
        flat per-isotopologue rows  (matched: peak id + ppm; else None)
```

---

## 2. Inputs

- **`peaks`** — the sample's raw peaks. `mz` / `height` / `peak_id` are read, and
  `signal_to_noise` where the server sends it; rows are **deduped on `peak_id`**
  (raw server peaks repeat per match) and sorted by m/z so the window search can
  use `np.searchsorted`. Without the noise column the score runs in its no-SNR
  mode and charges an absent line on predicted abundance alone, which is a
  different reading of every faint line — so `fetch_peaks` caches to a versioned
  file rather than reusing a frame from before the column existed.
- **`scoring`** — a `mascope_tools.composition.PatternScoring`, built per sample
  by `io_mascope.scoring_for_sample`: `sigma_ppm` and `mu_ppm` from
  `fit_mass_accuracy` over the sample's own targeted matches (falling back to the
  instrument class's `resolve_fallback_sigma_ppm` below eight anchors), and
  `mz_tolerance_ppm` from `resolve_match_tolerance_ppm` — 5 ppm Orbitrap, 15 TOF.
  Passing nothing scores at the library's defaults, which are an Orbitrap's.

  **The width and the offset have different bars.** The library reports both or
  neither, and below its minimum it returns `(0.0, None)` — a zero offset that
  reads exactly like a measured one. A robust spread needs a distribution; a
  median needs a few points that agree. So the offset is taken from the anchors
  whenever there are at least `MIN_OFFSET_ANCHORS` (3) of them, even where the
  width falls back to the class, and `pattern_scoring` records `sigma_source` and
  `mu_source` separately. This is not cosmetic: on a nitrate source sitting
  1.2 ppm low with six anchors, scoring at `mu = 0` charged that error to every
  candidate and handed the peak to whichever reading's own error cancelled it —
  the reference committed a third of what it had, and disagreed where it did.

  The minimum is five rather than three because an anchor is a peak the server
  matched within the instrument's *matching* window — 15 ppm on a TOF, five times
  its accuracy — so a mis-matched anchor sits in the set looking like a
  measurement. On a bromide TOF sample with three anchors, two mis-matched to the
  same wrong species, the median landed at −10.4 ppm for a source the engine
  measures within 0.3 ppm of calibration.
- **`formulas`** — candidate neutral formulas (the grid + cheminfo union).
- **channels** — either peaky `adducts` (labels like `[M+Br]-`) or already-resolved
  mascope **`mechanisms`** strings (`+Br-`); the dispatcher passes the latter, which
  skips `adduct_to_mech`. Each is parsed once via `utils.parse_ionization`.

---

## 3. The transformation, stage by stage

All thresholds are the named constants from `local_scoring.py` (see §4).

1. **Prepare peaks.** Keep `[mz, height, peak_id]`, drop null m/z,
   `drop_duplicates(peak_id)`, sort by m/z → parallel `mzs` / `ints` / `pids`
   arrays.

2. **Build the ion** (`utils.combine_formula_and_ionization`). neutral + parsed
   ionization → the ion formula string (e.g. `C6H12BrO6-`). `adduct_to_mech`
   (when adducts, not mechanisms, are passed) collapses multi-part adducts by
   concatenating the added pieces: `[M+HBr+Br]-` → `+HBrBr-` (= +HBr₂).

3. **Predict the envelope** (`predict_isotopes`, IsoSpec). → `pred_mz`,
   `pred_int`, `labels` for the charged ion, then **`anchor_on_monoisotopic`**:
   the ion's own line is moved first, whatever order IsoSpec returned it in.
   Everything downstream reads index 0 as the ion — the intensity the envelope is
   normalized to, the anchor every score term is measured against, `is_base` —
   and for a dibromide the most abundant configuration is the mixed 79/81 line
   two mass units up, which is not the peak the candidate was proposed for.
   Normalize to that line: **`pred_rel = pred_int / pred_int[0]`**, so a brighter
   satellite runs above 1. Empty envelope → skip.

4. **Match the monoisotopic base (i = 0).** Window half-width
   `d = pred_mz · scoring.mz_tolerance_ppm · 1e-6`; `searchsorted` for
   `[mz−d, mz+d]`, take the **closest** peak. **If M0 is not detected, the
   candidate is dropped entirely** (`base_int is None` → not a candidate at all)
   — there is no scoring an isotope envelope with no anchor.

5. **Match each heavier isotope (i > 0).** Same window + closest peak, but a peak
   is **accepted only if its intensity matches**: relative observed
   `rel_obs = ints[k]/base_int`, intensity error
   `ierr = |pred_rel[i] − rel_obs| / pred_rel[i]`, kept iff
   **`ierr ≤ INTENSITY_TOLERANCE` (0.4)** (= mascope_tools
   `ISOTOPE_MATCHING_INTENSITY_TOLERANCE`). A peak in the mass window with the
   wrong abundance is **not** attributed — it stays unmatched.

6. **Score the ion** (`score_pattern_v2`). One score per ion, on
   `(obs_ppm − scoring.mu_ppm, obs_int, obs_snr, pred_rel)` at
   `sigma_ppm=scoring.sigma_ppm`: each line contributes a Gaussian mass
   likelihood times an intensity likelihood whose tolerance is set by that
   peak's signal-to-noise, an **absent** line contributes a miss penalty iff the
   noise says it should have been visible (`pred_rel[i]·SNR_base ≥ 3`) and is
   excluded otherwise, and the per-line likelihoods combine as a
   predicted-abundance-weighted geometric mean. Reported `ppm_error` stays the
   raw measured error: the offset belongs to the calibration, not to the row.
   The same `compound_score` / `ion_score` is copied onto every isotopologue row
   of that ion.

7. **Categorize** (`_category`). `score ≥ PROBABLE_THRESHOLD (0.8)` → `probable`;
   `≥ POSSIBLE_THRESHOLD (0.4)` → `possible`; else `unlikely`. Bands on the fit's
   scale: v1 put a correct assignment near 0.95 whatever its envelope did, while
   the fit charges a missing line and a mass off the sample's own width, so a
   correct ion with a lone unremarkable peak lands in the 0.5–0.8 range these
   bands split.

8. **Emit rows.** One row per predicted isotopologue. Matched rows carry the
   attributed `sample_peak_id`, `sample_peak_mz/intensity`, and a real
   `ppm_error`; unmatched isotopologues carry `None` for peak id, score, and ppm
   (so the envelope is described but the gaps are honest).

---

## 4. Constants reference

All in `peaky/io/local_scoring.py`.

| constant | value | role |
| --- | --- | --- |
| `PROBABLE_THRESHOLD` | 0.8 | score ≥ → `probable`, on the fit's scale |
| `POSSIBLE_THRESHOLD` | 0.4 | score ≥ → `possible`; below → `unlikely` |
| `INTENSITY_TOLERANCE` | 0.4 | max relative abundance error to attribute an isotope (= `ISOTOPE_MATCHING_INTENSITY_TOLERANCE`) |
| `scoring.mz_tolerance_ppm` | 5 / 15 | half-window for matching a predicted line to a peak, from the instrument class (`resolve_match_tolerance_ppm`) |
| `scoring.sigma_ppm` | fitted | the mass term's width: the sample's own, else its class's `resolve_fallback_sigma_ppm` (0.3 Orbitrap, 3.0 TOF), widened by `PRED_SIGMA_PPM` |
| `MIN_OFFSET_ANCHORS` | 5 | anchors below which no offset is claimed either; above it the median stands even when the width cannot be fitted. Five because an anchor is a match within the instrument's *matching* window, so a mis-match can sit in the set - and two of them cannot carry a median of five |
| `k_detect` / `miss_penalty` | 3.0 / 0.3 | when an absent line is charged, and what it costs (in `mascope_tools`) |

---

## 5. Metrics, defined

- **ion score** — `score_pattern_v2(obs_ppm − mu, obs_int, obs_snr, pred_rel,
  sigma_ppm=…)`: a single 0–1 number, the fit quality of the envelope against
  this sample's own measurement. The whole arbitration downstream ranks on this.
  It is the same number Mascope's assignment engine ranks and tiers on, from the
  same library function, so a comparison between the two engines is a comparison
  of their assignments rather than of their scorers.
- **`pred_rel`** — predicted isotope intensities normalized to the base
  (`pred_int / pred_int[0]`); the reference the observed pattern is judged against.
- **`ierr` (intensity error)** — `|pred_rel − rel_obs| / pred_rel`; the gate that
  decides whether a mass-window peak is *really* this isotope (≤ 0.4) or a
  coincidence.
- **`ppm_error`** — `(obs_mz − pred_mz)/pred_mz · 1e6`, populated only on matched
  rows.

---

## 6. Outputs

One DataFrame, **identical columns to `flatten_match_tree`** (so it is a true
drop-in):

| column group | content |
| --- | --- |
| compound | `compound_formula`, `compound_score`, `compound_category` |
| ion | `ion_formula`, `ion_score`, `ion_category`, `mechanism_id` (`im.mascope_notation`) |
| isotopologue | `isotope_formula`, `iso_label` (`M0` / `13C` / …), `is_base` (`i == 0`), `theo_mz`, `rel_abundance` |
| match | `iso_score`/`iso_category` (None if unmatched), `sample_peak_id`, `sample_peak_mz`, `sample_peak_intensity`, `ppm_error`, `abundance_error` (None for the base) |

`io_mascope._score_candidates_local` stamps `frame.attrs` with
`match_batches = 0`, `match_batch_failures = []`, `match_formulas = len(formulas)`
so callers see a uniform shape across both backends.

---

## 7. Properties, invariants & gotchas

- **Same maths, same schema — only the *source* moves.** The scores are Mascope's
  authored code (`mascope_tools`, public PyPI, same authors), executed locally and
  pinned to a library version. This keeps the "Mascope is the scorer" invariant.
- **No M0, no candidate.** An envelope whose monoisotopic line isn't detected is
  dropped before scoring — there is nothing to anchor relative abundances to.
- **Intensity-gated isotope attribution.** Being inside the ±5 ppm window is *not*
  enough; the peak's abundance must match the prediction within 40 %, or it stays
  unmatched. This is what stops a dense spectrum from "confirming" every isotope
  by mass coincidence.
- **Only matched isotopologues are scored data.** Unmatched envelope lines are
  still emitted (so the predicted pattern is visible) but with `None` scores/ppm —
  never a fabricated zero.
- **Deterministic.** No network and no RNG, so a re-run over the same inputs is
  byte-identical — this is what lets the default scorer satisfy
  `test_determinism.py` (the `match_compounds` fallback re-introduces server-side
  variation). See [`ARCHITECTURE.md §7`](ARCHITECTURE.md#7-reproducibility--provenance).
- **Mixed +/- adducts are unsupported** by `adduct_to_mech` (it raises); a
  multi-part adduct must be all-add or all-subtract.
- **The switch is opt-out, not opt-in.** Local scoring is the default;
  `PEAKY_LOCAL_SCORING=0/false/no/off` falls back to the network path (see
  [`DATA_IO.md §3`](DATA_IO.md#3-the-transformation-stage-by-stage) and
  [`MASCOPE_TOOLS_INTEGRATION.md`](MASCOPE_TOOLS_INTEGRATION.md)).

---

## 8. Code map

| function (`local_scoring.py` unless noted) | role |
| --- | --- |
| `score_candidates_local` | the engine: peaks + formulas → flat per-isotopologue table |
| `adduct_to_mech` | peaky adduct label → mascope mechanism string (multi-part collapse) |
| `_category` | score → `probable` / `possible` / `unlikely` |
| `utils.parse_ionization` / `combine_formula_and_ionization` (mascope_tools) | parse channel; build the ion formula |
| `predict_isotopes` (mascope_tools, IsoSpec) | theoretical isotope envelope of the ion |
| `score_pattern_v2` (mascope_tools) | the fit score: mass and intensity likelihoods per line, a detectability-gated charge for an absent one, combined as an abundance-weighted geometric mean |
| `io_mascope.score_candidates` | backend dispatcher (local default ↔ `match_compounds`) |
| `io_mascope._score_candidates_local` / `_local_scoring_enabled` | local bridge (peaks from cache, mechanism names) + `PEAKY_LOCAL_SCORING` switch |
