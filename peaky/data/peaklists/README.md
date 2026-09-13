# Reference peaklists

A curated catalog of **known-molecule formula lists** from published chemical
systems (oxidation HOM, contaminant families, …). Peaky uses them as a **soft,
context-gated prior** to:

1. **corroborate** Candidate-tier assignments whose neutral formula appears on a
   list for the sample's chemistry, and
2. **rescue / annotate** unexplained peaks — by mass, under the run's reagent
   adducts — so a reader can see *"this unexplained 311.05 matches a known
   monoterpene-HOM, C10H16O7, at −0.3 ppm"*.

This is **not** the Pass-0 known-species lock (`passes._known_species`, for
contaminants the grid cannot reach). It is a post-hoc prior: it never overrides
an isotope-scored **Assigned**, never fabricates, and every match carries the
source list id for provenance.

## Adding a list

Drop in one self-describing JSON here. Nothing else couples — `reflists.load_catalog()`
globs this directory. **Only add credible, citable sources**, and fill `references`.

## Schema (`schema_version: 1`)

```jsonc
{
  "schema_version": 1,
  "id": "monoterpene_hom_kang2024",        // unique, file-name-safe
  "system": "monoterpene_OH_oxidation",    // free-text system family
  "label": "Monoterpene OH-oxidation HOM (α-pinene proxy)",
  "data_version": "2024.2",                // bump when the data changes
  "polarity": "negative",                  // native measurement polarity (informational)
  "native_detection": "[M+NO3]-",          // how the SOURCE detected them (informational)
  "applies_to_contexts": ["monoterpene_ox","limonene_ox","ap_ox","biogenic_soa"],
  "references": [                           // REQUIRED — where the formulas come from
    { "authors": "...", "title": "...", "publisher": "...", "year": 2022,
      "isbn": "...", "section": "...", "detection": "...", "note": "..." }
  ],
  "provenance": {                          // how this artifact was produced + caveats
    "extracted_from": "...", "verification": "...", "caveat": "..." },
  "n_species": 830,
  "species": [
    { "formula": "C10H16O7",               // NEUTRAL formula (matchable)
      "conditions": ["pure","NOx"],        // sub-experiments it appeared in (optional)
      "radical": false }                   // claim checked against DBE parity (default false)
  ]
}
```

### Conventions
- `formula` is always the **neutral** molecule. Lists published as detected ions
  (e.g. `[M+NO3]-` clusters or `[M-H]-`) must be normalized to the neutral on import.
- Radical status follows **DBE parity**, read off the formula:
  DBE = 1 + Σ nᵢ(vᵢ − 2)/2 with valences C, Si 4; N, P 3; O, S 2; H and the
  halogens 1. A half-integer DBE is an odd-electron neutral (RO•, RO2•). The H
  count is not the rule once N or P is present: the organic nitrate C10H15NO8 has
  odd H and an integer DBE, so it is closed-shell. Radicals are excluded from
  default matching (the pipeline assigns closed-shell neutrals); API callers may
  pass `include_radicals=True` to `match_by_mass` / `match_assigned` (a keyword
  argument on those two functions, not a run-level switch).
- `radical` (false when absent) is the author's claim; the formula decides.
  `load_catalog()` warns when a list's claims disagree with parity, and
  `tests/test_peaklists.py` fails a bundled list whose claims disagree or which
  holds a species with a negative DBE. A negative DBE is what a cation entered
  with its extra proton gives — the tetrabutylammonium cation `C16H36N` sits at
  −0.5, a quaternary-ammonium chloride written as one formula at −1 — so that
  test catches protonated cations and their salts, not ions in general: a
  deprotonated anion written without its charge (`C10H14NO8` for `[M-H]-`) has a
  non-negative DBE and would read as a radical instead.
- The loader refuses (warns and skips, `ReferenceList.skipped`) an entry whose
  formula uses an element off `chemistry.M` (`chemistry.dbe` would score it as
  divalent: sodium acetate `C2H3NaO2` comes out DBE 1.5), does not round-trip as
  a neutral Hill formula (charge or bracket notation, `[C10H14NO8]-`), or has a
  negative DBE. Parity is only read off a formula that passes.
- Masses are **not** stored — they are recomputed (`chemistry.ion_mz`) for whatever
  reagent adduct the run uses, so one list serves Br⁻ / NO3⁻ / I⁻ / urea⁺ runs alike.

## Current lists
| id | system | n | source |
|----|--------|---|--------|
| `monoterpene_hom_kang2024` | monoterpene OH-oxidation HOM | 830 | Kang, FZ Jülich E&U 557 (2022), App. A |
| `isoprene_ox_wennberg2018` | isoprene OH/HO₂, OH/NO and NO₃ oxidation products (reduced mechanism, closed-shell neutrals) | 27 | Wennberg et al., Chem. Rev. 118, 3337 (2018) |
| `contaminants_keller2008` | MS background contaminants (always active) | 50 | Keller et al., Anal. Chim. Acta 627 (2008) 71–81 |
