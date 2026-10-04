# Peaky — Isotopes (pattern prediction + prescan → grid constraints)

This document explains **two number transforms keyed off isotope masses**: (a)
predicting an ion's isotopologue envelope `[(Δm, rel-intensity, label)]`, and (b)
the *prescan* that reads satellite pairs off the raw peak list to **shrink the
formula grid before any scoring call**. It is a module deep-dive companion to
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline),
[`CHEMISTRY.md`](CHEMISTRY.md) (the grid this constrains), and
[`SCORING.md`](SCORING.md) (the authoritative isotope *judge* — this module is
**not** a scorer).

**Code:** `peaky/chem/isotopes.py`. The prescan answers two cheap questions —
*which heteroatoms show isotope-pair evidence?* and *how large is C?* — so the
candidate search shrinks before the server/local scorer is ever called.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `isotopes.py`; if you change one there, change it here.

---

## 1. What this stage does

Two related jobs, both pure arithmetic on isotope masses + natural abundances:

- **`isotope_pattern(ion_formula)`** — convolve each element's per-atom isotope
  distribution into the ion's resolved M+1/M+2/M+4 envelope. This is what lets the
  envelope-completion pass recognize an unexplained peak as a committed parent's
  satellite.
- **`prescan(peaks)` → `constrain_ranges`** — walk the brightest peaks for
  characteristic satellite pairs (¹³C, ⁸¹Br, ³⁷Cl, ³⁴S, ²⁹Si) at known Δm with
  *plausible intensity ratios*, then **cap C** and **zero out heteroatoms with no
  spectral evidence** in the grid box — a large combinatorial saving, gated by what
  the context allows.

```
PRESCAN (shrink the grid)                       PATTERN (predict an envelope)
raw peaks (mz, height)                          ion_formula
  │ strip reagent m/z (±reagent_ppm 15)           │ parse → element counts
  │ sort by height, scan brightest ≤400           │ convolve per-atom _ISO_DIST
  ▼ for each parent: probe parent+Δm              ▼ (prune 1e-6)
  13C ratio → estimated_max_C                     merge lines < merge_da (0.006)
  81Br/37Cl/34S/29Si ratio window → has_X         keep rel ≥ min_rel (0.03)
  ▼                                               ▼  Δm in (0.4, max_shift 6.5]
constrain_ranges:                                [(Δm, rel, label)] sorted
  C ≤ estimated_max_C + 4 ; zero Br/Cl/S/Si with no evidence (∧ context cap)
```

---

## 2. Inputs

- **`prescan`** — a peak table (`mz`, `height`); optional `reagent_mzs` to strip
  known reagent clusters first so bare Brₙ doesn't masquerade as analyte Br.
- **`constrain_ranges`** — the base grid box, a `PrescanResult`, and
  `context_caps` (per-element max the reagent context permits).
- **`isotope_pattern`** — an **ion** formula string.

---

## 3. The transformation, stage by stage

### Prescan (grid-shrinking)

1. **Strip reagent peaks.** Drop any peak within **`reagent_ppm` (15)** of a known
   reagent-cluster m/z, then sort by height (the brightest peaks carry the isotope
   information) and scan the top **`n_scan` = min(N, 400)**.

2. **¹³C → carbon count.** Probe `parent + D_13C (1.003355)` within
   **`ppm_tol` (8)**. If the satellite/parent ratio is in **[0.003, 0.9]**,
   estimate `n_C = round(ratio / R_13C_PER_C)` with `R_13C_PER_C = 0.0107` (1.1 %
   per carbon); keep the running max.

3. **⁸¹Br.** Probe `parent + D_81BR (1.9979521)`. Ratio in **[0.6, 1.4]** → a
   single Br (`81Br/79Br ≈ 0.97`). Ratio in **[1.5, 2.6]** → check the M+4 at
   `parent + 2·D_81BR` is **[0.6, 1.4]** → Br₂ (`has_multi_Br`).

4. **³⁷Cl.** `parent + D_37CL (1.997050)`, ratio in **[0.22, 0.45]** (one Cl,
   `37Cl/35Cl ≈ 0.32`) → `has_Cl`.

5. **³⁴S.** `parent + D_34S (1.995796)`, ratio in **[0.025, 0.09]** (per S,
   `34S/32S ≈ 0.044`) → `has_S`.

6. **²⁹Si.** `parent + D_29SI (0.999568)`, ratio in **[0.035, 0.08]** (per Si,
   `29Si/28Si ≈ 0.051`) → `has_Si`.

7. **Constrain the grid** (`constrain_ranges`). Cap C at
   **`estimated_max_C + 4`** (headroom) when estimated; for each of Br/Cl/S/Si,
   set the range to `(0, 0)` when there is **no evidence** *or* the context cap is
   ≤ 0, else clamp to the context cap.

### Pattern prediction

8. **Convolve** (`isotope_pattern`). Build the distribution as the per-atom
   convolution of `_ISO_DIST` (each element's `[(Δm_from_lightest, abundance)]`),
   keyed on the mass shift rounded to mDa. Prune paths below **`PRUNE` 1e-6** — soft
   enough that multi-atom cross terms survive (a 4-Si M+4 is a sum of many ~1e-3
   paths; an aggressive prune underpredicts it and a silanediol M+4 wrongly
   survives as a contaminant).

9. **Merge + threshold.** Keep lines with `0.4 < Δm ≤ max_shift (6.5)`; merge any
   two closer than **`merge_da` (0.006)** (the picker resolves them as one peak, so
   ⁸¹Br at +1.9978 and ³⁰Si at +1.9968 *add*); emit `(Δm, rel, label)` for every
   line with `rel ≥ min_rel (0.03)`. `_label_for_shift` names the dominant
   achievable isotopologue (restricted to combos the formula can form — a 1-Br
   ion's M+4 is `81Br+30Si`, never `81Br2`) when within **0.012 Da**, else `M+N`.

---

## 4. Constants reference

All in `peaky/chem/isotopes.py`.

| constant | value | role |
| --- | --- | --- |
| `D_13C` | 1.003355 | ¹³C−¹²C Δm |
| `D_37CL` | 1.997050 | ³⁷Cl−³⁵Cl Δm |
| `D_81BR` | 1.9979521 | ⁸¹Br−⁷⁹Br Δm (consistent with `passes._DBR`) |
| `D_34S` | 1.995796 | ³⁴S−³²S Δm |
| `D_29SI` | 0.999568 | ²⁹Si−²⁸Si Δm |
| `D_18O` | 2.004246 | ¹⁸O−¹⁶O Δm |
| `R_13C_PER_C` | 0.0107 | ¹³C satellite per carbon (1.1 %) |
| `R_37CL_PER_CL` | 0.3196 | ³⁷Cl/³⁵Cl |
| `R_81BR_PER_BR` | 0.9728 | ⁸¹Br/⁷⁹Br |
| `R_34S_PER_S` | 0.0443 | ³⁴S/³²S |
| `R_29SI_PER_SI` | 0.0510 | ²⁹Si/²⁸Si |
| `_HEAVY_ELEMENTS` | Br, Cl, Si, S | the M+2 drivers |
| `prescan` `ppm_tol` / `reagent_ppm` | 8.0 / 15.0 | satellite match / reagent-strip windows |
| `prescan` `n_scan` | min(N, 400) | brightest peaks scanned |
| ¹³C accept ratio | [0.003, 0.9] | range that yields a carbon estimate |
| ⁸¹Br accept ratio | [0.6,1.4] (1 Br) · [1.5,2.6]+M+4 (Br₂) | single vs di-bromide |
| ³⁷Cl / ³⁴S / ²⁹Si accept ratio | [0.22,0.45] / [0.025,0.09] / [0.035,0.08] | one-atom evidence windows |
| C headroom (`constrain_ranges`) | `estimated_max_C + 4` | cap above the estimate |
| `isotope_pattern` `min_rel` / `max_shift` / `merge_da` | 0.03 / 6.5 / 0.006 | line floor / mass span / merge window |
| `PRUNE` | 1e-6 | convolution path floor (deliberately soft) |
| `_label_for_shift` tol | 0.012 Da | name an isotopologue vs fall back to `M+N` |

---

## 5. Metrics, defined

- **satellite ratio** — `height(parent+Δm) / height(parent)`; compared to a
  per-element acceptance window, **not** scored. Outside the window → no evidence.
- **`estimated_max_C`** — `round(¹³C-ratio / 0.0107)`, the running max over scanned
  parents; the carbon ceiling fed to the grid.
- **`rel` (envelope)** — line intensity ÷ M0 intensity, after convolution + merge.
- **`Δm`** — intensity-weighted exact mass shift of a (possibly merged) envelope
  line.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `PrescanResult` | `has_Br/has_Cl/has_S/has_Si`, `has_multi_Br`, `estimated_max_C`, `evidence[]` (the hits); `as_dict()` for the manifest |
| `constrain_ranges` | a tightened grid box (C capped, unsupported heteroatoms zeroed, context-clamped) |
| `isotope_pattern` | `[(Δm, rel_intensity, label)]` sorted by shift, lines ≥ `min_rel` |

---

## 7. Properties, invariants & gotchas

- **The prescan is NOT a scorer.** It only proposes which heteroatoms/Cmax to *let
  into the grid*; Mascope's maths (local or server) is the authoritative isotope
  judge ([`SCORING.md`](SCORING.md)). A ratio outside a window means "don't widen
  the grid", never "reject the formula".
- **Evidence-gated heteroatoms.** With no ⁸¹Br/³⁷Cl/³⁴S/²⁹Si pair, that element is
  forced to `(0,0)` — the combinatorial saving and the reason a neutral halogen
  must *earn* its place.
- **Context still bounds it.** `constrain_ranges` clamps to `context_caps`; a
  context that allows 0 of an element zeroes it regardless of a stray pair.
- **The prune is intentionally soft (1e-6).** Multi-atom M+4 cross terms are sums
  of many tiny paths; pruning hard underpredicts the M+4 and lets a silanediol M+4
  masquerade as a Cl doublet.
- **Merged lines add.** Two isotopologues closer than 6 mDa are reported as one
  line at their intensity-weighted mass, mirroring what the peak picker resolves.
- **Labels are formula-aware.** `_label_for_shift` only names combinations the
  formula can form, so a 1-Br ion never gets an `81Br2` label.
- **Two label conventions share the ledger's `iso_label`.** The scorer's labels
  count heavy atoms from the ion's MONO line (`81Br2`, `M0` = the mono line below
  a heavy parent, `13C+81Br`) and it commits an ion's most ABUNDANT isotopologue
  (a Br2 ion on 79Br81Br); peaky's own labels (`_label_for_shift`, pass 4's
  `(pair)`, the reclaim tails) count from the PARENT line (`2x81Br`, `81Br(pair)`,
  `81Br+13C`, `M+5`). No column records the producer, so the per-file isotope facts
  (`evidence._measure`, which the merge vote's class reads) read a child both ways against the committed line (C11+c, `resolve_child`
  and `judge_source` below). The evidence scale probes isotope lines with its
  own line model (docs/EVIDENCE_LEVELS.md §5.3).
- **`D_81BR` was corrected** to 1.9979521 (AME2020) from a stale 0.16 mDa-low
  value. The residual pass's doublet constant is the exact 1.9979535 of
  chemistry's masses since C11+c (it was the same stale value); the two agree to
  1.4 µDa, far inside every window that reads them.

---

## 8. Code map

| function | role |
| --- | --- |
| `isotope_pattern` | convolve per-atom `_ISO_DIST` → resolved `(Δm, rel, label)` envelope |
| `_label_for_shift` | name the dominant *achievable* isotopologue at a shift (or `M+N`) |
| `prescan` | scan brightest peaks for ¹³C/⁸¹Br/³⁷Cl/³⁴S/²⁹Si pairs → `PrescanResult` |
| `_find_partner` | nearest peak to a target m/z within ppm (the satellite probe) |
| `constrain_ranges` | apply prescan evidence to the grid box (cap C, zero unsupported heteroatoms) |
| `PrescanResult` | the evidence record (+ `as_dict` for the run manifest) |
| `split_label` / `parse_label_part` | (C11+c) a child label's `+` parts and each part's heavy atoms (`kx`, counts, `(pair)`, `81Br/37Cl(pair)`, `M0`, `M+n`) |
| `heavy_probability` | P(a heavy configuration) vs the all-light line: the multinomial over the ion's atoms |
| `committed_configuration` | the parent's committed isotopologue, read off its m/z (5 / 20 ppm) |
| `resolve_child` | a label read both ways against the committed line: shift, readings, joint expectation, the elements it adds |
| `fit_position_sigma` / `position_window_ppm` | σ(h)² = a² + b²/h self-fitted on a source's `13C` children; max(1 ppm, 4σ(h)) |
| `judge_source` / `line_facts` | one source's children and isotopologues lists placed and banded; what a pair's kept lines give |
