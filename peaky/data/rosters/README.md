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

**Status: reviewed 2026-09-22.** Both files signed off as they stand. One exception to
the citation rule: an observed contaminant with no literature identity may be listed
by formula with the observation as its reference, named `unnamed <formula>`
(`contaminants.csv`: unnamed C22H42O6, plasticiser class). A run's rows that are
found but not on the roster (a C11 acid family on an alpha-pinene run) stay
"found but unexpected"; the roster is not widened to cover contamination the run
could not anticipate.