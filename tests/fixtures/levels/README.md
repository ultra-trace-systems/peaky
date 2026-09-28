# Level fixtures — the four golden sets behind `tests/test_evidence.py`

Fifty-two per-file ledgers from four real runs (and one neutral-pair table, one label-twin table and two lock tables), trimmed to what the evidence
levels read: the `M0` and `iso_child` rows and the twenty-two columns the
predicates in `docs/EVIDENCE_LEVELS.md` named when they were cut. They predate
the twenty-third, `tentative_lead` (C19(c), 2026-09-27): it reads False here,
their leads sit in `below_assignability`, and no vector moved with the split --
except on the four pairs rule H lifts (C11+b, below). Gzipped CSV (`pandas.read_csv`
opens them directly); 3.6 MB in all with the tables and `expected_levels.csv`.

| set | fixtures | source | M0 rows | iso rows | golden vector (2b/3a/3b/4a/4b/4c/4d/5a/5b) |
|---|---|---|---:|---:|---|
| `tv_nitrate`, `tv_bromide` | 2 | one TOF, two reagent channels on the same air; each ledger is one source and the two corroborate each other | 1,373 | 570 | 1373 · 21/15/107/16/143/15/10/37/1009 |
| `tof_01` … `tof_28` | 28 | a mixed bromide/nitrate TOF, 18 cover + 10 residual files, pooled as ONE source | 8,313 | 4,286 | 3364 · 6/15/182/15/258/82/95/138/2573 |
| `orbi_01` … `orbi_12` + `orbi_label_twins.csv` + `orbi_iso_checks.csv` | 12 + 2 | a labelled-nitrate Orbitrap sampling the same air as the TOF set, ONE source; the TOF and Orbitrap sets corroborate each other; levelled with its lock table (rule H, below); its label-twin table (rule K, measured by `batch/label_twins.py` on the run's stamped time series) | 9,773 | 2,238 | 1707 · 0/11/217/9/206/139/0/35/1090 (without the lock table 0/11/217/9/203/139/0/35/1093; with the label-twin table 0/11/174/9/206/151/0/38/1118, with both 0/11/174/9/209/151/0/38/1115) |
| `ur_01` … `ur_10` + `ur_neutral_pairs.csv` + `ur_iso_checks.csv` | 10 + 2 | a uronium Orbitrap batch, ONE source, no corroboration; levelled with its neutral-pair table (rule U, measured by `batch/neutral_pairs.py` on the run's stamped time series) and its lock table (rule H) | 8,076 | 2,263 | 1161 · 4/4/0/331/377/291/0/82/72 (with the neutral-pair table alone 4/4/0/331/376/291/0/82/73; without either 4/4/0/25/682/291/0/82/73) |

`expected_levels.csv` holds one row per `(source, neutral, adduct)` with the
level and every predicate input, written by `scripts/level_ledger.py`; the
in-core `evidence` stage must reproduce it row for row and the three vectors
exactly. Each pair corroborates by what the other source pins on its own (4b or
better with no cross set, `docs/EVIDENCE_LEVELS.md` §6.4); the file was
regenerated for that rule (183 of 6,444 levels moved, all downward; the vectors
before it were 21/15/107/38/162/9/9/33/979, 6/16/182/38/260/79/91/135/2557 and
0/12/217/44/215/119/0/30/1070). Rebuild it from the fixtures with
`build_expected_levels.py` (outside the repo, beside the fixture builder), which
reproduces the previous file exactly under the previous script. C17
(2026-09-27: `multiline` counts two ELEMENTS the neutral supplies, each by an
in-band line; 15N and 18O join the ratio band) regenerated it again: 14 rows
changed. Seven are levels, all in the TOF set (bromide-adduct 4a rows that counted
the reagent's 81Br fall to 4b; the vector was 6/15/182/22/247/82/99/138/2573);
seven gain the isotope axis from an in-band 18O line at an unchanged level (six
nitrate-channel rows of the TV set and HNO2 on the TOF). C11+a (2026-09-27:
`reagent_only_iso` only where the ION carries more of the reagent halogen than
the neutral) regenerated it again: 14 rows changed, all in the bromide-channel
sets. Six are levels, all in the TOF set: `[M-H]-` rows 4d -> 4b (C10H10BrClO6,
C12H20BrCl, C15H27BrO5, C18H27BrO4 -- brominated neutrals whose 81Br line is
their own -- and the Br-free C35H32O4S, C35H36O6S; the vector was
6/15/182/15/254/82/99/138/2573); eight lose the flag at an unchanged level
(three `[M-H]-` rows of `tv_bromide`, five of the TOF set). The hold (the same
day: an ion carrying NONE of the reagent halogen keeps the flag -- its 1:1 +2 Da
line argues against the formula -- until C11+c's count-aware band) takes back the
eight Br-free rows: the two Br-free levels return to 4d (vector
6/15/182/15/258/82/95/138/2573) and six rows keep the flag; the file now differs
from the pre-C11+a one in the six brominated-neutral rows alone.

C11+b (rule H, decision D8, 2026-09-28): a halogen lock of the batch's time series
lifts a tentative lead, never a row below assignability, so as cut the fixtures
could not show it. The Orbitrap and uronium sets now carry `tentative_lead` and
`lead_by` (False / empty on every row) and, on the four pairs the live runs lock
AND hold as leads, the flag moved from `below_assignability` to `tentative_lead`
with the setter the source row's note names: the Orbitrap set's HBr `[M+^NO3]-`
(orbi_03, orbi_07), C6H9ClO3 `[M-H]-` (orbi_02, orbi_06) and C6H10Cl2O4 `[M-H]-`
(orbi_12) -- `spec_gapfill`, "speculative residual fit -- series gap-fill with no
supporting anchors" -- and the uronium set's C7H11ClO2 `[M+(CH4N2O)H]+` (ur_01,
ur_02, ur_06) -- `off_budget`, "outside the uronium element budget (Cl=1 > 0)".
A text edit: every other byte of the decompressed files is as it was. The flag
move alone moves no level (a lead is hard like below). `orbi_iso_checks.csv` and
`ur_iso_checks.csv` are the rule H lock rows (the full `tables/iso_checks.csv`
schema) of the live runs of those batches, recomputed by `batch/iso_checks.py` from
each run's stamped series, for the pairs the set holds: 15 of the labelled-nitrate
run's 16 locks (its C5H9ClO4 `[M]-.` line is not in the set) and the uronium run's
1. Levelled with them, exactly the four pairs move, 5b -> 4b "4b: one
corroboration (iso)": the Orbitrap vector 0/11/217/9/203/139/0/35/1093 ->
0/11/217/9/206/139/0/35/1090 (with rule K's table 0/11/174/9/206/151/0/38/1118 ->
0/11/174/9/209/151/0/38/1115), the uronium one 4/4/0/331/376/291/0/82/73 ->
4/4/0/331/377/291/0/82/72; `expected_levels.csv` changes in those four rows only
(`build_expected_levels.py` levels each set with its own tables).

Scrubbed: file names are positional, `sample_item_id` is dropped, and no site,
instrument or server name is inside. The privacy scanner skips `.gz`; the
decompressed content was scanned once and its only hits were seven neutral
*formulas* that happen to be exactly sixteen characters long (mixed-halogen
CHNOPS formulas), which share the shape of a Mascope id — they are formulas,
not ids. Rebuild from the regression set
(outside the repo) with `build_level_fixtures.py`.
