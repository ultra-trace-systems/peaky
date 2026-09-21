# Rosters — what a run is expected to find

Two short, cited lists the scorecard (`scripts/scorecard.py`) measures a run
against. They are **expectations**, not priors: nothing in the assignment
engine reads them, so a hit here is evidence about the engine, not a product
of it (the reference peaklists under `../peaklists/` are the priors).

| file | what | rows |
|---|---|---|
| `alpha_pinene.csv` | alpha-pinene oxidation products: first-generation monomers, HOM monomers, dimers, organonitrates | 32 |
| `contaminants.csv` | the usual suspects in a chamber / inlet: plasticisers, siloxanes, perfluoro acids, organophosphates | 28 |

Columns: `name`, `formula` (the NEUTRAL), `class` (the roster is scored per
class), `reference` (author, year, journal, DOI), `note`.

Two rows share a formula on purpose (`C8H12O4` = norpinic acid / terpenylic
acid; `C9H14O3` = norpinonic / pinalic acid): a formula assignment cannot tell
them apart, and the scorecard counts a formula once.

**Status: unreviewed.** Written 2026-09-21 for Scoreboard v0 from the cited
sources; the user reviews both files before the recall numbers are quoted
anywhere. Add a row with its citation; do not add a row without one.
