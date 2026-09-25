# Level fixtures — the three golden sets behind `tests/test_evidence.py`

Forty-two per-file ledgers from three real runs, trimmed to what the evidence
levels read: the `M0` and `iso_child` rows and the twenty-two columns the
predicates in `docs/EVIDENCE_LEVELS.md` name. Gzipped CSV (`pandas.read_csv`
opens them directly); 2.1 MB in all.

| set | fixtures | source | M0 rows | iso rows | golden vector (2b/3a/3b/4a/4b/4c/4d/5a/5b) |
|---|---|---|---:|---:|---|
| `tv_nitrate`, `tv_bromide` | 2 | one TOF, two reagent channels on the same air; each ledger is one source and the two corroborate each other | 1,373 | 570 | 1373 · 21/15/107/16/143/15/10/37/1009 |
| `tof_01` … `tof_28` | 28 | a mixed bromide/nitrate TOF, 18 cover + 10 residual files, pooled as ONE source | 8,313 | 4,286 | 3364 · 6/15/182/22/247/82/99/138/2573 |
| `orbi_01` … `orbi_12` | 12 | a labelled-nitrate Orbitrap sampling the same air as the TOF set, ONE source; the TOF and Orbitrap sets corroborate each other | 9,773 | 2,238 | 1707 · 0/11/217/9/203/139/0/35/1093 |

`expected_levels.csv` holds one row per `(source, neutral, adduct)` with the
level and every predicate input, written by `scripts/level_ledger.py`; the
in-core `evidence` stage must reproduce it row for row and the three vectors
exactly. Each pair corroborates by what the other source pins on its own (4b or
better with no cross set, `docs/EVIDENCE_LEVELS.md` §6.4); the file was
regenerated for that rule (183 of 6,444 levels moved, all downward; the vectors
before it were 21/15/107/38/162/9/9/33/979, 6/16/182/38/260/79/91/135/2557 and
0/12/217/44/215/119/0/30/1070). Rebuild it from the fixtures with
`build_expected_levels.py` (outside the repo, beside the fixture builder), which
reproduces the previous file exactly under the previous script.

Scrubbed: file names are positional, `sample_item_id` is dropped, and no site,
instrument or server name is inside. The privacy scanner skips `.gz`; the
decompressed content was scanned once and its only hits were seven neutral
*formulas* that happen to be exactly sixteen characters long (mixed-halogen
CHNOPS formulas), which share the shape of a Mascope id — they are formulas,
not ids. Rebuild from the regression set
(outside the repo) with `build_level_fixtures.py`.
