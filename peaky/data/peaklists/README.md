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

## Pinning: `catalog.lock` and schema v2 snapshots (`peaky.lists`)

**The rule: a run never fetches a list.** Every list a run reads is a file in
this directory (or in the user directory, `$PEAKY_LISTS_DIR`, default
`~/.cache/peaky/lists`), pinned by content hash in that directory's
`catalog.lock`. `reflists.load_catalog()` recomputes each list's hash and
refuses a mismatch with a `SnapshotIntegrityError` that names the list -- a
list is read exactly as pinned or not at all. The run manifest
(`run_manifest.json["lists"]`, `batch_summary.json["lists"]`, the single-sample
`<sid>_manifest.json`) records, per active list, `{id, version, sha256, roles,
pinned, how_activated}`.

- `peaky lists lock [--dir]` rewrites the lock over every snapshot in a directory.
- `peaky lists verify [--dir]` loads with verification (exit 1 on a mismatch).
- `peaky lists show <id>` prints one list's provenance, roles, pinning, counts.
- `peaky lists import <source> [--from-file CSV] [--dest DIR | --bundle]` runs an
  **importer** (`peaky/lists/importers/`): the only code that knows a source's
  URL and format. It fetches the published file, verifies the publisher's
  checksum (size + md5 from the Zenodo record AND the importer's own
  constants), asserts the exact header it was written for (a changed column
  fails here, loudly, never inside a run), converts the rows, writes the
  snapshot and updates the lock. `--bundle` writes into this directory: only
  for a redistributable licence (the repo is public).
- `--lists ID[,ID]` on `peaky assign` / `peaky batch` (or `$PEAKY_LISTS`)
  activates a list by id beside the keyword-unlocked ones; an unknown id stops
  the run before any work.

The **hash** is `sha256` over the canonical species payload: the species
reduced to `formula, name, inchikey, inchikey_block, cas, ids, radical`,
sorted by (formula, inchikey, name), compact key-sorted JSON. Re-indenting or
re-ordering a file changes nothing; one changed byte of one species is refused.

### Schema v2 (an imported snapshot)

```jsonc
{
  "schema_version": 2,
  "id": "norman_sle_s22_epacons",
  "system": "consumer_products", "label": "...", "polarity": "both",
  "data_version": "NORMAN-SLE-S22.0.1.0",     // = source.version
  "always_active": false, "applies_to_contexts": ["consumer_products", ...],
  "roles": ["name"],                           // name | class | propose | isomer_count
  "family_scope": "compound",                  // compound: a named hit can make 3c; class: a tag
  "source": { "kind": "norman-sle", "doi": "10.5281/zenodo.2648759", "concept_doi": "10.5281/zenodo.2648758",
              "record_id": 2648759, "version": "NORMAN-SLE-S22.0.1.0",
              "file": "EPA_ConsumerProductSuspects_01032018.csv", "bytes": 482416,
              "md5": "b44bc1224bf080ead559fc2f5ba7ca33", "retrieved_utc": "...", "licence": "CC-BY-4.0",
              "url": "https://zenodo.org/api/records/2648759/files/.../content", "encoding": "cp1252" },
  "importer": { "name": "peaky.lists.importers.epacons", "version": "0.1.0" },
  "references": [ ... ], "provenance": { "attribution": "...", "excluded": [ ... ], ... },
  "sha256": "<species hash>", "n_species": 1664,
  "species": [ { "formula": "C10H19O6PS2", "name": "Malathion", "inchikey": "JXSJBGJIGXNWCI-UHFFFAOYSA-N",
                 "inchikey_block": "JXSJBGJIGXNWCI", "cas": "121-75-5", "ids": {"dtxsid": "DTXSID4020791"},
                 "radical": false } ]
}
```

**Roles.** `name`: the entries are NAMED context-list entries (a pinned reading
the list names reads level 3c; `levels/lists.py::ContextLists`). `class`: class
entries, a tag only. `propose` and `isomer_count` are RESERVED -- a snapshot may
declare them, nothing consumes them yet (`propose` = a pass-0 proposer family
replacing the hard-coded dictionaries in `passes/directors.py`; `isomer_count`
= the number of known structures per formula, for `evidence.load_isomer_space`).

**Lists are proposers, never evidence.** A name reaches 3c only after the ion is
established and the split pinned; a big list names something near every
formula, so each named hit carries a *list specificity* tag (how many formula
neighbours +-CH2 / +-H2 / +-O the same list also names). Detection of an
off-grid formula a list names comes from the existing routes: active-list
formulas enter the curated set (pass 7's P/I rule relaxes from >= 3 to >= 2
channels for a curated formula) and the reflist rescue (pass 8).

**Identity keys.** The neutral formula in Hill notation is the match key;
`inchikey_block` the structure key; names are display only; CAS is carried,
never a key. Masses are recomputed by peaky; the importer compares the
source's and records a disagreement > 1 mDa under `provenance.mass_warnings`.
Several compounds sharing one formula load as one entry whose display name
joins the first two (`A; B (+n more)`), with `n_compounds` and `names` in
`meta_of`.

v1 files (the three lists above this section) keep loading unchanged; they are
pinned by the same lock.

## Current lists
| id | schema | system | n | source |
|----|--------|--------|---|--------|
| `monoterpene_hom_kang2024` | 1 | monoterpene OH-oxidation HOM | 830 | Kang, FZ Jülich E&U 557 (2022), App. A |
| `isoprene_ox_wennberg2018` | 1 | isoprene OH/HO₂, OH/NO and NO₃ oxidation products (reduced mechanism, closed-shell neutrals) | 27 | Wennberg et al., Chem. Rev. 118, 3337 (2018) |
| `contaminants_keller2008` | 1 | MS background contaminants (always active) | 50 | Keller et al., Anal. Chim. Acta 627 (2008) 71–81 |
| `norman_sle_s22_epacons` | 2 | US EPA consumer-product suspects (glycol ethers, Texanol, DEET, siloxanes, plasticizers, organophosphate pesticides); activate with `--lists` | 1664 compounds / 766 formulas | Phillips et al., ES&T 52, 3125 (2018); NORMAN-SLE S22, [10.5281/zenodo.2648758](https://doi.org/10.5281/zenodo.2648758), CC-BY-4.0 |

### Attribution
`norman_sle_s22_epacons`: NORMAN Suspect List Exchange, S22 | EPACONS | US EPA
Consumer Product Suspect List (Phillips et al. 2018),
https://doi.org/10.5281/zenodo.2648758, licensed CC-BY-4.0. Redistributed here
as a derived snapshot (formulas in Hill notation, masses recomputed, 41 entries
that are not a neutral molecule over peaky's mass table -- salts, organometallics,
mixtures without a formula -- excluded and listed under `provenance.excluded`).
