# Level fixtures — the four golden sets behind `tests/test_evidence.py`

Fifty-two per-file ledgers from four real runs (and one neutral-pair table, one label-twin table and two lock tables), trimmed to what the evidence
levels read: the `M0` and `iso_child` rows and the columns the predicates in
`docs/EVIDENCE_LEVELS.md` named when they were cut -- twenty-two on the TV and
uronium sets, twenty-one on the TOF and Orbitrap sets (no `resolvability`). They
predate `tentative_lead` (C19(c), 2026-09-27): where the column is missing it
reads False, their leads sit in `below_assignability`, and no vector moved with
the split. Since C11+b the Orbitrap and uronium sets carry `tentative_lead` and
`lead_by` (23 and 24 columns): False / empty on every row but the eight rows of
the four pairs rule H lifts (below), whose flag moved from
`below_assignability` to `tentative_lead` with its setter. Gzipped CSV (`pandas.read_csv`
opens them directly); 3.6 MB in all with the tables and `expected_levels.csv`.

| set | fixtures | source | M0 rows | iso rows | golden vector (2b/3a/3b/4a/4b/4c/4d/5a/5b) |
|---|---|---|---:|---:|---|
| `tv_nitrate`, `tv_bromide` | 2 | one TOF, two reagent channels on the same air; each ledger is one source and the two corroborate each other | 1,373 | 570 | 1373 · 21/15/107/15/145/15/9/37/1009 |
| `tof_01` … `tof_28` | 28 | a mixed bromide/nitrate TOF, 18 cover + 10 residual files, pooled as ONE source | 8,313 | 4,286 | 3364 · 6/15/182/14/267/85/84/135/2576 |
| `orbi_01` … `orbi_12` + `orbi_label_twins.csv` + `orbi_iso_checks.csv` | 12 + 2 | a labelled-nitrate Orbitrap sampling the same air as the TOF set, ONE source; the TOF and Orbitrap sets corroborate each other; levelled with its lock table (rule H, below); its label-twin table (rule K, measured by `batch/label_twins.py` on the run's stamped time series) | 9,773 | 2,238 | 1707 · 0/10/217/9/205/138/0/35/1093 (without the lock table 0/10/217/9/202/138/0/35/1096; with the label-twin table 0/10/174/9/205/150/0/38/1121, with both 0/10/174/9/208/150/0/38/1118) |
| `ur_01` … `ur_10` + `ur_neutral_pairs.csv` + `ur_iso_checks.csv` | 10 + 2 | a uronium Orbitrap batch, ONE source, no corroboration; levelled with its neutral-pair table (rule U, measured by `batch/neutral_pairs.py` on the run's stamped time series) and its lock table (rule H) | 8,076 | 2,263 | 1161 · 4/4/0/330/375/293/0/82/73 (with the neutral-pair table alone 4/4/0/330/374/293/0/82/74; without either 4/4/0/17/687/293/0/82/74) |

`expected_levels.csv` holds one row per `(source, neutral, adduct)` with the
level and every predicate input, written by `scripts/level_ledger.py`; the
in-core `evidence` stage must reproduce it row for row -- level and every
recorded fact (`test_every_fixture_row_matches_the_reference_script_fact_for_fact`)
-- and the vectors exactly. Each pair corroborates by what the other source pins on its own (4b or
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
1, with the silicon test's columns (the 2026-09-28 decision: the four Cl locks above
m/z 206 -- three here, the uronium one -- read no 29Si line; recomputed 2026-09-29
with the round-2 engine, the co-variation gate and the `unparted` reading, which
changes those four rows' `si29_seen` / `si29_mode` / note and nothing else; the
round-3 position test leaves every row as it is).
Levelled with them, exactly the four pairs move, 5b -> 4b "4b: one
corroboration (iso)": the Orbitrap vector 0/11/217/9/203/139/0/35/1093 ->
0/11/217/9/206/139/0/35/1090 (with rule K's table 0/11/174/9/206/151/0/38/1118 ->
0/11/174/9/209/151/0/38/1115), the uronium one 4/4/0/331/376/291/0/82/73 ->
4/4/0/331/377/291/0/82/72; `expected_levels.csv` changes in those four rows only
(`build_expected_levels.py` levels each set with its own tables).

C11+c (2026-09-30: an isotope child counts only at its label's exact spacing
from the parent's COMMITTED isotopologue -- max(1 ppm, 4 sigma(h)) self-fitted on
the set's own '13C' children, pcal and the neighbour's pull rescuing, 'M+n'
exempt -- and in band under its count-aware expectation relative to that line;
the whole label is read, the isotopologues list answers the same question, and
the Br-free hold on `reagent_only_iso` is released; docs/EVIDENCE_LEVELS.md
§3.2) regenerated `expected_levels.csv` from the unchanged fixtures:
47 of 7,605 levels move, and the isotope facts (`iso`, `reagent_only_iso`) of
others. The sets carry no width model: the committed line is read within the
class-less 20 ppm and an 'M+n' line within 12 mDa. tv (6): TV-BR C12H22O,
C18H14N2O8 `[M+Br]-` 5b -> 4b, C13H16O5 `[M+Br]-` 4a -> 4b, C20H33NO10 `[M+Br]-`
4d -> 4b, C14H14O5 `[M-H]-` 4b -> 5b; TV-NO3 C9H12O5S `[M-H]-` 4b -> 5b. tof (33):
twelve `[M+Br]-` 4d / 5a / 5b -> 4b on an M+3 '81Br+13C' carbon line or a line
proving a second Br, C10H16O8 `[M+Br]-` 4a -> 4b, the two Br-free C35H32O4S /
C35H36O6S `[M-H]-` 4d -> 5b (their 1:1 +2 line expects nothing on a Br-free ion:
mass-degenerate with no axis), C12H9BrN2 `[M+HBr+Br]-` 4b -> 4d (a Br3 ion on
its 79Br2 81Br line: its '81Br2' lines, 0.97x, sit where a Br2 ion puts one, and
its '2x81Br' 81Br3 line is out of band), and the rest dim readings whose child
sits off its exact spacing or out of its count-aware band. orbi (4): the
chlorinated paraffin C10H18Cl4 `[M+^NO3]-` 3a -> 5b (its one list line reads
2.0x a 37Cl1-committed parent, 0.48 expected: its recovery labels every ladder
line '37Cl', a separate card), C11H20O10 and C13H15ClO8 `[M-H]-` 4b -> 5b,
dibromoacetic acid C2H2Br2O2 `[M-H]-` 4c -> 4b (its 81Br2 and 79Br2 lines
around the 79Br81Br-committed M0). ur (4): C10H17NO4, C18H25NO `[M+H]+` 4b -> 4c,
C12H14N2O4Si `[M+H]+` 4a -> 4b, C17H22N4O10 urea 4b -> 5b. The release of the
hold moves no level: it clears the flag on three TOF `[M-H]-` rows (C22H20O21,
C35H32O4S, C35H36O6S), all 5b. Nor does D4's full-count line (a kept, in-band
line only the ion's full halogen count makes is the neutral's halogen): it moves
`reagent_only_iso` on six rows at unchanged levels -- tof IBr `[M+Br]-` (2b),
C13HBrO4 `[M+Br]-` and C27H27BrN2O2 `[M+HBr+Br]-` (5b) lose it on an in-band
79Br2 / 81Br3 line; tv C13H16 and tof C20H22N2O12, C8H14O5 `[M+HBr+Br]-` (5b),
Br-free, take it on their `M0` line.

Scrubbed: file names are positional, `sample_item_id` is dropped, and no site,
instrument or server name is inside. The privacy scanner skips `.gz`; the
decompressed content was scanned once and its only hits were seven neutral
*formulas* that happen to be exactly sixteen characters long (mixed-halogen
CHNOPS formulas), which share the shape of a Mascope id — they are formulas,
not ids. Rebuild from the regression set
(outside the repo) with `build_level_fixtures.py`.
