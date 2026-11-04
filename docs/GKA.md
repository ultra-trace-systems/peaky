# Peaky — GKA & homologous series (Kendrick analysis)

This document explains **how repeat-unit (homologous-series) structure is found
and turned into numbers** — the Pass-2 anchor-propagation engine, the
decoy-controlled automatic detector, and the static Generalized Kendrick Analysis
(GKA) figure. It is a module deep-dive companion to
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline, passes 2–3),
[`ASSIGNMENT.md`](ASSIGNMENT.md) (which validates these proposals), and
[`QC_AND_REPORT.md`](QC_AND_REPORT.md) (which embeds the GKA page).

**Code:** `peaky/assignment/series_gka.py` (the Pass-2 engine + KMD math),
`peaky/assignment/series_detect.py` (automatic, decoy-controlled detection), and
`peaky/reporting/gka_figure.py` (the static figure). The engine **proposes**;
the pass director (`run_pass2` in passes/directors.py) decides truth via Mascope + isotopologue corroboration.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in these three modules; if you change one there, change it here.

---

## 1. What this stage does

A homologous series is a set of formulas separated by an exact repeat unit (CH₂,
O, CO₂, CF₂, …). In a Generalized Kendrick plot such a series is a **horizontal
row of constant Kendrick mass-defect** (a CH₂ addition leaves both the KMD and the
DBE unchanged). Three jobs use that fact:

- **Propagate** (`series_gka`): step locked anchors by ± repeat units to *propose*
  neighbouring neutral formulas (Pass 2).
- **Detect** (`series_detect`): count repeat-unit links across residual peaks,
  controlled by **decoy** units, to *open contaminant families* on evidence (Pass 3).
- **Render** (`gka_figure`): freeze the canonical CH₂ KMD plot + per-family
  small-multiples for the report.

```
ANCHORS (Pass 1)                 RESIDUAL peaks                 committed ledger
   │ propose_for_peak             │ detect_series                │ _neutral_masses
   │  anchor ± n·unit (n≤1)       │  _link_count vs decoy mean   │  (drop F-monsters)
   │  ion m/z within ppm 3        │  significant: links≥12 ∧     ▼
   ▼   support = #anchors ±1      │   enrichment≥3               present_families
 Proposal[]  → passes/            ▼  action → open family       (longest ladder > 3)
 (validated by match_compounds)  evidence table (manifest)       ▼
                                                          render_gka: KMD panels
                                       GKM = mz·X / mass(base);  GKD = GKM − round(GKM)
```

---

## 2. Inputs

- **Engine** — a set of anchor neutral formulas (Pass-1 locked), the analyte
  adducts, and target peak m/z.
- **Detector** — the ledger's `unexplained` (+ optional Low/Suspect M0) peaks.
- **Figure** — a committed ledger (reads `neutral_formula` + `role`/`tier` only).

---

## 3. The transformation, stage by stage

### A. The Pass-2 series engine (`series_gka`)

1. **Repeat units** (`REPEAT_UNITS`). Element-count deltas for each unit; organic
   growth is `ORGANIC_UNITS = (CH2, H2O, O, CO, CO2, C2H2O)`, contaminants are
   `CONTAMINANT_UNITS = (C2H6OSi, CF2)`, plus cluster/ladder units (HBr, HCl,
   C2F4, C2H4O, C3H6O, C2H4O2).

2. **Propose** (`propose_for_peak`, `ppm` 3.0, `max_steps` 1). For each anchor ×
   unit × `n ∈ {−1,+1}`, form `formula_add(anchor, unit, n)`, compute the ion m/z
   under each adduct, and keep it if `|ppm| ≤ 3`. Each proposal gets a
   **`n_supporting_anchors`** count (`_support_count`: how many anchors sit ±1 unit
   away). Sorted by support desc, then `|ppm|`.

3. **Chains** (`find_homolog_series`, `min_len` 3). Group formulas into maximal
   runs spaced by one unit (start only at chain heads with no −1 neighbour); return
   runs ≥ `min_len`. The same routine powers both the figure and the engine.

4. **GKA / KMD math.** `GKM(mz, base, X) = mz·X / mass(base)`,
   `GKD = GKM − round(GKM)` (Alton et al., AMT 2023). `X` = the base's nucleon
   number reproduces traditional Kendrick; a larger `X` expands the defect scale to
   separate congested series.

### B. Automatic detection (`series_detect`)

5. **Population.** The `unexplained` peaks (+ Low/Suspect M0 when
   `include_low_confidence`), filtered by `min_height`.

6. **Link counting** (`detect_series`, `ppm` 5.0). For each unit in `UNIT_LIBRARY`,
   `_link_count` = peaks whose `+1 unit` partner exists within ppm; `_chain3_count`
   = peaks with partners at both `+1` and `+2` (chains ≥ 3).

7. **Decoy control.** Re-run the link count at the unit mass shifted by
   `_DECOY_OFFSETS (0.0317, −0.0473, 0.0689 Da)` — irrational-ish offsets off any
   real unit — and take the mean. `enrichment = n_links / decoy_mean`. A unit is
   **`significant`** iff `n_links ≥ min_links (12)` **and** `enrichment ≥
   min_enrichment (3.0)`.

8. **Open families** (`families_from_evidence`). A significant unit with an
   `action` opens that contaminant family: `CF2`/`C2F4` → `fluorinated`,
   `C2H6OSi` → `siloxane`, `SO3` → `organosulfate`. `unit_members` / `unit_chains`
   restrict an opened family's *targets* to the chain members that justified it (so
   a dense mass-filler like F can't claim the whole residual). The evidence table
   goes into the run manifest.

### C. The static figure (`gka_figure`)

9. **Neutral masses** (`_neutral_masses`). One entry per distinct assigned neutral,
   **excluding unconfirmed-fluorine "monsters"** (`_is_f_monster`: F ≥ 4, not a
   PFCA `CₙHF₍₂ₙ₋₁₎O₂`, no Cl/Br/S anchor — ¹⁹F is monoisotopic, so these are mass
   coincidences) so the fluorinated panel shows real PFAS. A monster the ledger did
   **not** refuse is kept: one read off its CF2 ladder (`assignment/fluorine.py`)
   carries no `below_assignability` flag, and a full ladder may reach Assigned,
   which a refused row never does (on the merged ledger, which carries the tier but
   not the flag, only an Assigned one is kept).

10. **Family panels** (`present_families`, `_panel`, base `kmd`). One small-multiple
    per `FAMILIES` entry (alkyl/CH₂, oxidation/O, alkoxylate/C2H4O, siloxane/C2H6OSi,
    fluorinated/CF2), each rotated to its base so that family flattens to horizontal
    ladders. A family earns a panel **only if its longest ladder has >
    `PANEL_MIN_MEMBERS` (3) members** ("no series → don't plot"); connector lines
    are drawn only for ladders with > 3 members. Panels **zoom to their drawn span**
    (`ZOOM_PAD` 0.10). Element families (Si/F) highlight every element-bearing peak.

---

## 4. Constants reference

`series_gka.py` · `series_detect.py` · `gka_figure.py`.

| constant | value | role |
| --- | --- | --- |
| `ORGANIC_UNITS` | CH2, H2O, O, CO, CO2, C2H2O | default CHO/CHON growth units (engine) |
| `CONTAMINANT_UNITS` | C2H6OSi, CF2 | siloxane / PFAS series units |
| `propose_for_peak` `ppm` / `max_steps` | 3.0 / 1 | propagation mass window / max units per step |
| `find_homolog_series` `min_len` | 3 | shortest chain returned by the engine |
| `UNIT_LIBRARY` | CH2…SO3 (+ exact mass, action) | the auto-detector's scanned units |
| `_DECOY_OFFSETS` | 0.0317, −0.0473, 0.0689 Da | irrational shifts for the chance-alignment baseline |
| `detect_series` `ppm` | 5.0 | link-match window |
| `detect_series` `min_links` | 12 | absolute link floor for significance |
| `detect_series` `min_enrichment` | 3.0 | links/decoy_mean floor for significance |
| `PANEL_MIN_MEMBERS` | 3 | a family's longest ladder must be **> 3** to draw |
| `ZOOM_PAD` | 0.10 | fraction of span padded around each panel |
| `render_gka` `min_len`/`highlight_min_len`/`top_chains` | 4 / 5 / 10 | figure series floors + max chains/panel |

---

## 5. Metrics, defined

- **GKM / GKD (KMD)** — `mz·X/mass(base)` and its fractional part; a homologous
  series in the base unit is a **row of constant GKD** (horizontal in the panel).
- **`n_supporting_anchors`** — how many existing anchors sit ±1 unit from a proposal;
  ≥ 2 (one below *and* above) is the GKA strength signal that earns higher confidence.
- **`n_links` / `n_chains3`** — peaks with a `+1` partner / with both `+1` and `+2`
  partners; the raw series signal.
- **`decoy_mean` / `enrichment`** — chance-alignment baseline / `n_links ÷ decoy_mean`;
  the signal-to-noise that gates significance.
- **longest ladder** — members in a family's longest base-unit chain; the
  draw/no-draw gate (> 3).

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `propose_for_peak` → `Proposal[]` | candidate neutral formulas (target_mz, neutral, adduct, anchor, unit, n_steps, ppm, n_supporting_anchors) — validated downstream |
| `detect_series` DataFrame | per unit: `n_links`, `n_chains3`, `decoy_mean`, `enrichment`, `significant`, `action` (→ run manifest) |
| `families_from_evidence` | contaminant families to open (fluorinated / siloxane / organosulfate) |
| `figures/gka_<tag>.png` | the GKA small-multiple grid + per-family rollup bar |
| `family_summary` / `detect_series` (figure) | per-family `n_series` / `n_members` / `longest` |

---

## 7. Properties, invariants & gotchas

- **The engine proposes; it never decides truth.** Every proposal is validated by
  the pass director (`run_pass2` in passes/directors.py) against Mascope `match_compounds` and kept only if the isotopologue
  pattern corroborates it — a ≥ 2-anchor proposal earns higher confidence.
- **Decoys make detection honest.** A unit is significant only when its link count
  is both well above the irrational-offset decoy baseline (enrichment ≥ 3) **and**
  absolutely large (≥ 12 links) — chance alignments in a dense residual don't open
  a family.
- **F-monsters are excluded from the figure.** ¹⁹F is monoisotopic, so an
  unconfirmed high-F formula is a mass coincidence; the fluorinated panel shows the
  real PFCAs and the readings a CF2 ladder lifted, not the demoted coincidence fits.
- **"No series → don't plot."** A family (organic *or* element) appears only if its
  longest ladder exceeds 3 members; a lone D3→D4 siloxane pair is dropped.
- **CH₂ is the canonical base** because it leaves KMD and DBE unchanged — that is
  why a CH₂ series is exactly a horizontal row, and the rows you see *are* the GKA
  finding. Other rotations (oxidation, alkoxylate) are the other panels.
- **Targets are scoped to the chain.** `unit_members`/`unit_chains` keep an
  evidence-opened family from claiming peaks beyond the chain that justified it.
- **Panels zoom to their data** so a family clustered in a narrow window isn't lost
  across the full 50–750 Da axis.

---

## 8. Code map

| function | role |
| --- | --- |
| `series_gka.REPEAT_UNITS` / `unit_mass` / `formula_add` | repeat-unit deltas; exact unit mass; formula ± n·unit |
| `series_gka.gkm` / `gkd` | the GKA mass / mass-defect math |
| `series_gka.propose_for_peak` / `_support_count` | anchor propagation + multi-anchor support count |
| `series_gka.find_homolog_series` | maximal repeat-unit chains (engine + figure) |
| `series_detect.detect_series` / `_link_count` / `_chain3_count` | decoy-controlled link counting |
| `series_detect.unit_members` / `unit_chains` / `families_from_evidence` | scope targets; open families on evidence |
| `gka_figure._neutral_masses` / `_is_f_monster` | assigned neutrals (F-monsters excluded) |
| `gka_figure.present_families` / `_longest_ladder` / `_panel` | choose + draw the family panels |
| `gka_figure.kmd` / `render_gka` | KMD math (base CH₂) + the full figure |
