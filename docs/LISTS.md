# Reference-list snapshots

How peaky reads external chemical lists (NORMAN-SLE suspect lists, PubChemLite,
curated compendia) so that a run made today gives the same names, and says
why, when it is re-executed years from now.

## The rule

**A run never touches the network.** It reads snapshot files from a catalog
directory, resolved through that directory's lockfile, verifies each one's
content hash, and records in its manifest exactly which snapshots it read.
Importers fetch; runs read.

```
source (Zenodo / API)  ->  importer  ->  snapshot (schema v2 JSON)  ->  catalog.lock  ->  run
   network, by hand         fails loudly     pinned by sha256          verified at load    manifest["lists"]
```

Why not an API call at run time: reproducibility (the database changes under
you), cost (thousands of lookups per file against rate-limited public APIs),
deployment (instrument PCs are offline or firewalled), longevity (URLs and API
shapes change). Live lookups belong to the agent layer and to `peaky lists
import`, both of which are explicit, recorded steps.

## Layout

| what | where |
|---|---|
| bundled catalog | `peaky/data/peaklists/*.json` + `catalog.lock` (ships in the wheel) |
| user catalog | `$PEAKY_LISTS_DIR`, default `~/.cache/peaky/lists`, with its own `catalog.lock` |
| schema + hash | `peaky/lists/snapshot.py` |
| lock | `peaky/lists/lock.py` |
| importers | `peaky/lists/importers/<source>.py`, registered in `IMPORTERS` |
| runtime loader | `peaky/assignment/reflists.py::load_catalog` (verifies, caches per process) |
| manifest record | `reflists.lists_record` -> `run_manifest.json["lists"]`, `batch_summary.json["lists"]`, `<sid>_manifest.json["lists"]` |

A user-directory list of an id the bundled catalog also has wins, with a
warning; the manifest records the file either way.

## Commands

```bash
peaky lists verify                     # every list, every directory; exit 1 on a mismatch
peaky lists show norman_sle_s22_epacons
peaky lists lock --dir peaky/data/peaklists   # after adding or editing a bundled list
peaky lists import norman_sle_s22_epacons          # -> the user directory
peaky lists import norman_sle_s22_epacons --bundle # -> the bundled catalog (redistributable licence only)
peaky batch --batch "Site A Ur 122-600 ..." --dataset "Site A campaign" --reagent Ur --lists norman_sle_s22_epacons
```

`--lists ID[,ID]` (or `$PEAKY_LISTS`) activates lists by id beside the ones
the batch / dataset names unlock by keyword. An unknown id stops the run
before any work; a list whose hash does not match its lock stops it too.

## What a list does, and does not do

A list is a **proposer and a namer, never evidence**:

- Its formulas enter the curated set (`cfg.reflist_formulas` ->
  `curated_formulas`): the plausibility and degeneracy gates spare a curated
  reading, and pass 7 (certified-neutral discovery) relaxes its P/I rule from
  >= 3 to >= 2 converging ion channels for one. Pass 8 (reflist rescue) matches
  unexplained peaks by mass to list formulas and commits them when the isotope
  line confirms.
- A `name`-role entry is a NAMED context-list entry: a reading whose ion is
  established and whose neutral/adduct split is pinned reads level **3c** when
  the list names its neutral (`docs/EVIDENCE_LEVELS.md`). Nothing below that
  changes: the levels are computed from the spectra, the name only adds the
  identity step.
- Each named hit carries a *list specificity* tag: how many formula neighbours
  (+-CH2, +-H2, +-O) of the neutral the same list also names. A list that
  names every neighbour names everything; the tag is the reader's warning and
  the obvious place for a floor later.

The known-answer test (`tests/test_lists_known_answer.py`) shows the division:
with peaky's hard-coded organophosphate dictionaries removed, malathion
(C10H19O6PS2) in a synthetic uronium spectrum is certified by pass 7 from three
ion channels converging on one neutral mass (the CHON grid cannot reach a P/S2
formula), and the pinned EPACONS snapshot names it -> 3c, `context =
reflist:norman_sle_s22_epacons = Malathion`; the same run without the list
finds it and leaves it unnamed.

## Schema v2 and the lock

See `peaky/data/peaklists/README.md` for the schema block. The content hash is
over the canonical species payload (`formula, name, inchikey, inchikey_block,
cas, ids, radical`; sorted; compact JSON), so layout never matters and a
changed species always does. The lock pins `{version, sha256, file,
source_doi, roles, schema_version}` per id. Updating a list = re-import (new
snapshot + lock entry) in one commit; an old run's manifest still names the
hash it read.

## Writing an importer

One module per source under `peaky/lists/importers/`, registered in
`IMPORTERS`. It must: pin the record id, file name, byte size and the
publisher's md5 as constants; read the publisher's record and refuse a file
whose checksum differs (a new version is a deliberate constant change, not a
surprise); assert the exact header; derive `formula` with
`chemistry.parse_formula` / `format_formula` and exclude anything
`reflists._not_a_neutral` refuses (recorded under `provenance.excluded` with
the reason); compare the source's monoisotopic mass to peaky's; strip `, ppm`
from names (the batch's known-label parser ends a label there); build the
snapshot with `source`, `importer`, `references`, `provenance.attribution`
(the licence's attribution text) and `sha256`; validate; write; update the
lock. Only CC-BY / public-domain sources may be bundled.

## Reserved roles, not yet consumed

- `propose`: a pass-0 proposer family from data, to retire the dictionaries in
  `passes/directors.py::_known_species` (the gates -- >= 2 channels or a
  diagnostic isotope -- stay in code). Not needed for detection: pass 7 and
  pass 8 already find a listed off-grid formula.
- `isomer_count`: known structures per formula (PubChemLite for Exposomics,
  CC-BY-4.0, Zenodo concept 10.5281/zenodo.5995885), to replace the hand-made
  `peaky/data/isomer_space.csv` with "plausible N of known M".
