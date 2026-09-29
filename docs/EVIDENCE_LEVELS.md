# Evidence levels — how much a committed formula is worth

**Status: implemented (B2), validated on real runs (B3), verified (B4).**
`peaky/assignment/evidence.py` is the `evidence`
stage of `assign.run` and the pooled recompute of `assign_batch.run`;
`tests/test_evidence.py` pins the contract. The executable reference for every
predicate below is still `scripts/level_ledger.py`; the in-core stage reproduces
it row for row on the fixtures in `tests/fixtures/levels/` and hits the three
golden count vectors exactly. Where this document, the script and the code
disagree, the script is right and the other two are bugs (one known reading
difference is listed in §10).

## 1. What a level is

`tier` says whether the engine is willing to *print* a formula (Assigned) or
only to *offer* it (Candidate). It is a verdict on the assignment. A level
says what the evidence behind a committed formula is worth on the scale a
reader of an identification paper already knows — Schymanski et al. (2014),
*Environ. Sci. Technol.* 48, 2097 — numbered **downward, 1 = best**, adapted
to chemical ionization, where there is no chromatography, no fragment
spectrum, no library, and the reagent is part of the ion.

One level per **committed M0 row**. Isotope children, reagent ions,
artifacts and unexplained peaks carry no level (`NA`). The level is computed
per `(neutral_formula, adduct)` over the rows that share it inside one
*source* (a file, or a batch's pooled files — §6), and stamped on every M0
row of that pair.

Levels 1 and 2a are defined and never fire: 1 needs an authentic standard
measured in the same source and reagent chemistry; 2a needs a library
spectrum, and CIMS has none.

### 1.1 Claims — what a level lets a reader say (C13)

Nine rungs answer "how good is the evidence"; a reader of a result table
asks something coarser: may this formula be reported as a compound?
`evidence.claim_class(level)` reads the level as one of three **claims**
(`evidence.CLAIMS`):

| claim | levels | what a reader may say |
|---|---|---|
| `identified` | 1, 2a, 2b, 3a, 3b, 4a (`CLAIM_IDENTIFIED`) | the neutral is established: the formula can be reported as a compound or class |
| `ion` | 4b, 4c, 4d (`CLAIM_ION`) | the ion composition is pinned; the neutral / adduct split is open |
| `tentative` | 5a, 5b, no level | exact mass only, or the assignment argues with itself |

`CLAIM_MEANING` holds the sentence each output prints for a claim.

- **A pure function of the level.** `claim_class` reads `evidence_level` and
  nothing else — no predicate column, no axis, no tier. The value is stripped
  of whitespace; NaN, `pd.NA`, `None`, `''`, 5a, 5b and any string it does not
  know read `tentative`.
- **Only a committed formula makes a claim.** Every committed M0 row carries
  one (per file from the file's own level, §6.1; on the merged ledger from the
  pooled level, §6.2); isotope children, reagent ions, artifacts and
  unexplained peaks carry `NA`. A committed row with no level — a merged row
  whose reading no per-file ledger holds — reads `tentative`, never `NA`.
- **Ion-only rows count by their level.** An ion-only pair (§3) is 4d with its
  own ¹³C — `ion` — or 5a without — `tentative`. Where an output crosstabs the
  claim against the tier, the ion-only rows are a row of their own
  (`ion-only`) beside Assigned and Candidate, not part of the Candidate row.
- **The vote class is not the claim.** The merge vote's evidence class
  (`assign_batch._evidence_class`, docs/MERGE.md §3 step 4) ranks on the same
  two level sets (`EVIDENCE_CLASS_GOOD` = `CLAIM_IDENTIFIED`,
  `EVIDENCE_CLASS_MID` = `CLAIM_ION`) and adds the `corroborated` axis on top:
  a corroborated reading votes with the good class at any level, so a
  corroborated 5b votes class 2 and claims `tentative`. The vote reads
  `evidence_level` / `evidence_axes`, never `claim`.
- **Tier and claim are separate verdicts.** The tier says whether the engine
  prints a formula or only offers it; the claim says what the evidence lets a
  reader say of it. Both are read from the same ledger columns and they are
  not nested: on the three same-air regression runs 16 / 5 / 52 Candidate rows
  are identified and 112 / 11 / 103 Assigned rows are tentative. The outputs
  show the two side by side (the workbook, the PDF report and the scorecard
  list the rows where they part); neither is read off, corrected from or
  routed into the other.
- **Nothing upstream reads it.** Not the tiers, not the merge vote, not the
  cross set (§6.4), not a predicate: the claim changes no ion, tier or level.

## 2. The columns a level reads

All from the ledger `peaky.assignment.ledger` already writes. Every read is
null-safe: NaN, `pd.NA`, `None`, `''` and `'false'` are False;
`bool(numpy.nan)` is True and once inflated 5b by 127 rows.

| column | rows | read as |
|---|---|---|
| `role` | all | `M0` are levelled; `iso_child` supply satellites; everything else is ignored |
| `peak_id`, `parent_peak_id` | M0, iso | a child hangs off the M0 with `peak_id == parent_peak_id` **in the same file** |
| `iso_label` | iso | the satellite's tag before any `+` (`13C+1` → `13C`; `81Br`, `34S`, `37Cl`, `29Si`, `30Si`, `18O`, `15N`) |
| `height` | M0, iso | the ratio child/parent, tested against natural abundance |
| `ion_formula` | M0 | carbon count for the 13C expectation |
| `neutral_formula`, `adduct` | M0 | the pair the level is about |
| `method` | M0 | `known:<family>` marks a pass-0 curated identity |
| `confidence` | M0 | its first word: `High`, `Good`, `Low`, `Suspect` |
| `tied` | M0 | the arbiter broke a near-tie |
| `below_assignability` | M0 | the assignment **argues with itself** — set by `tiers.flag_below_assignability` (O ≥ 11 and mass-saturated); the `plausibility` O-monster (`demote_oxygen_monsters`: O/C > 1.3 on a mass-degenerate window) and carbon-cluster (`demote_carbon_clusters`: integer DBE/C ≥ 1) demotes; `cleanup.demote_unconfirmed_fluorine` (F ≥ 4, no confirmed Cl/Br/S isotope anchor, not a PFCA); `cleanup.demote_implausible_carbon` ((H+F)/C < 0.35); `cleanup.demote_implausible_ionization` (a hydrocarbon or an N-only neutral read through an anion channel, docs/ASSIGNMENT_DETAIL.md §5.6c); `cleanup.demote_speculative_residual` for an **off-calibration** residual fit (\|z\| > `cal_z_accept`: the mass disagrees with the calibration). A corroborated radical-anion relabel (`cleanup.relabel_radical_anions`) clears it; an ion-only row never carries it. Created False with `tentative_lead` by the tier stage; a commit, a clear and a displacement reset both (`ledger.reset_flags`) |
| `tentative_lead` | M0 | (C19(c), 2026-09-27) the proposal is **unsupported, not contradicted** — the other half of what `below_assignability` said before the split: `reflists.rescue_unexplained_by_reflist`'s dim branch (a reference-list match too dim to show its ¹³C line, "tentative lead, not confirmed"); `plausibility.demote_off_budget` (a commit outside the run context's element budget that no curated list names, docs/ASSIGNMENT_DETAIL.md §8.5); `cleanup.demote_speculative_residual` for **N ≥ 3 with no isotope corroboration**, a **series gap-fill with no supporting anchors** and a **sole minor channel**; `cleanup.relabel_radical_anions` when the radical anion is **uncorroborated** (a corroborated one clears both flags); `cleanup.relabel_reagent_n_adducts` (the reagent-N re-read; per file only on a single sample, and the merged ledger a batch re-reads carries no flag, so nothing is written there). A row both flags mark keeps `below_assignability`: precedence is hard. A ledger written before the split has no such column: its leads sit in `below_assignability` and the missing column reads False. On the pooled batch a halogen lock lifts a lead it answers (rule H, C11+b: §3 `lead_lift`) |
| `lead_by` | M0 | (C11+b, 2026-09-28) which setter made the row a lead: the pipe-joined, sorted codes `reflist_dim`, `off_budget`, `spec_n3` / `spec_gapfill` / `spec_minor` (the three speculative-residual reasons), `radical_anion`, `reagent_n` (`ledger.LEAD_SETTERS`, written by `ledger.mark_lead`); `''` on every other row. Created and reset with the flags. Not a flag and never read per file: rule H reads it on the pooled batch, where the commentary that also names the setter is not among the columns kept (`evidence.PREDICATE_COLUMNS`). A ledger written before it has no such column: its lead rows read "any setter", the element budget's included (§3 `lead_lift`) |
| `degeneracy_density` | M0 | plausible ions sharing the mass in the calibrated window (`degeneracy.apply_degeneracy`) |
| `degeneracy_note` | M0 | contains `MASS-SATURATED` when the audit capped |
| `resolvability` | M0 | `resolved` / `isolated` / `blended` / `unresolvable`; stamped by the `resolvability` stage on every run that has a width model (a batch measures one from the raw profile of a middling spectrum; `--resolving-power` declares or declines it); NA on an offline run without one |
| `series_unit`, `anchor_peak_id` | M0 | a homologous-series or anchor tie |
| `isotopologues` | M0 | the server-attributed satellite list (a repr'd list; non-empty counts as isotope support) |
| `tier` | M0 | reported beside the level; never read by a predicate |

Optional, reported only: `mz`, `ppm_error_cal`, `occurrence`, `n_files`.

## 3. The evidence record — one per (neutral, adduct) per source

Computed exactly as `level_ledger.measure_source` does.

| fact | definition |
|---|---|
| `iso` | some child of some row of the pair has a height ratio within **0.5–2.0×** of natural abundance (13C: 0.0107 per carbon of `ion_formula`; 15N: 0.00369 per ¹⁴N atom of `ion_formula` — a caret `^N` atom is already ¹⁵N; 18O: 0.00205 per O atom, except on an ion carrying Br or Cl, whose 81Br / 37Cl line owns the M+2 region (no 18O line is measured there); 34S 0.0443; 37Cl 0.3196; 81Br 0.9728; 29Si 0.0508; 30Si 0.0335) **or** any row's `isotopologues` list is non-empty. 15N and 18O joined the band with C17 (2026-09-27). On a pair rule H lifts (`lead_lift`, batch only) the lock is its isotope axis |
| `multiline` | (C17, 2026-09-27) the pair's children carry in-band lines (the `iso` test, per line) of **≥ 2 distinct elements the neutral supplies**. A child's tag names the atoms it measures (`13C` / `13C2` → C, `81Br` / `2x81Br` / `81Br2` / `81Br(pair)` → Br, `37Cl…` → Cl, `15N` → N (the ¹⁴N atoms), `14N` → `^N` (a ¹⁵N label's atoms), `18O` → O, …); a generic `M+n` child names none. The neutral **supplies** them when it holds more than half of the ion's atoms of that key, counted unfolded (a labelled adduct's `^N` is not the neutral's N): the ¹⁵N line of a urea adduct of an N-free neutral, the ⁸¹Br line of a bromide adduct and the ¹⁸O line of formic acid's nitrate cluster (2 of 5 O) measure the reagent; the ¹⁸O line of C₁₀H₁₆O₄'s urea adduct (4 of 5 O) measures the neutral. Before C17 it was "≥ 2 distinct satellite tags", so 13C + 13C2 (one element twice), the reagent's own 15N / 81Br and `M+n` children all counted (output audit K01). The elements are written as `multiline_elements` (`C|O`) |
| `carbon_ev` | a `13C…` tag is among them |
| `chan2` | the **neutral** is committed under ≥ 2 distinct adducts in this source — on a labelled-nitrate batch (rule K, batch only) the 14N `[M+NO3]-` and the 15N `[M+^NO3]-` count as **one** channel, and neither an alien 14N line (`label_alien`) nor a pair an isotope check refutes (`iso_veto`, C11+, batch only) gives or takes it. On a pair rule H lifts (`lead_lift`) it is read over its own adduct and the adducts its neutral commits on an **unflagged** regular row of a pair that is not alien (rule K's fold applies): a second channel only flagged rows gave goes with the flag |
| `anchor` | any row has `anchor_peak_id` or `series_unit` — on a pair rule H lifts, any **unflagged** row |
| `branch` | the neutral's adduct set meets both `{[M-H]-}` and `{[M+NO3]-, [M+15NO3]-, [M+^NO3]-, [M+Br]-, [M+HBr+Br]-, [M+CO3]-}` — deprotonated **and** clustered: the gas-phase-acidity branch; an alien 14N line (rule K) or a pair an isotope check refutes (`iso_veto`) is not in the set and takes no branch; a pair rule H lifts reads it over the adduct set of its `chan2` (unflagged rows only) |
| `reagent_only_iso` | the channel has a reagent halogen (§3.1), the pair has satellites, none is 13C, every tag starts with that halogen's heavy isotope (`81Br` / `37Cl`; iodine is monoisotopic, so an iodide channel never sets it), and (C11+a, 2026-09-27) the **ion carries more of that halogen than the neutral** — the ion's composition read off its `ion_formula` when that carries a charge sign, else neutral + adduct (a ledger row can hold the neutral there), **or carries none of it** (2026-09-27 hold). On a brominated neutral's `[M-H]-` or `[M+NO3]-` the 81Br line is the neutral's own and the flag clears (4d on the flag before); an ion with no Br at all keeps the flag: a 1:1 +2 Da line on a Br-free ion is no isotope line of that formula -- it argues against it (the batch's HIGH check refutes the reading) -- and is held at the old 4d until C11+c's count-aware band judges it per file. False on a pair rule H lifts: its lock is a line of its own |
| `tied` | **all** rows of the pair are tied — unless rule K's untie clears it (batch only, `label_untie` below) |
| `below` | **any** row is below assignability |
| `lead` | (C19(c)) **any** row is a tentative lead. Hard like `below` (derived `hard` below) and worded like it: a lead-only pair reads `5b: below assignability`, exactly as the same pair did before the split, so the split moved no level and no reason. False on the pooled batch where a halogen lock lifts it (`lead_lift`, rule H) |
| `lowconf` | **all** rows are `Low` or `Suspect` |
| `degeneracy` | median `degeneracy_density` over rows that have one; NaN when none has. The audit counts the ions the run could have committed in the calibrated window -- its channels, its element space, no box (`degeneracy.py`); a row whose commit lies outside that space with fewer than three ions carries NaN (`not measured`), so `unique` never holds for it |
| `saturated` | any `degeneracy_note` contains `MASS-SATURATED` |
| `res_ok` | no row carries a `resolvability` value, **or** at least one is `resolved` / `isolated` — a source that never measured it is not penalised |
| `corroborated` | the neutral is in the **cross set** (§6.4): a source — the other reagent channel, the other instrument on the same air, a `--corroborate` run — that holds it at **4b or better by its own evidence**; never for an ion-only pair |
| `known_fam` | the family of the first `known:` method among the rows, else `''` |
| `upair` | (rule U, C17 + U, 2026-09-27) **batch only**: the neutral is in the batch's **neutral-pair set** — the profile declares a `(bare, cluster)` pair (`ReagentProfile.neutral_pair`; bundled: the uronium profile's `([M+H]+, [M+(CH4N2O)H]+)`, 60.0324 Da apart) and `batch/neutral_pairs.py` finds, on the stamped batch time series, both ions of the neutral committed, the neutral C/H/O only (N-free: the NH4 alias `[X+NH4]+` and the urea ladder both need N; no S/Si/P/halogen), each ion present within 2 ppm of its exact m/z in ≥ 50 % of the spectra with \|median ppm\| ≤ 1 (the signed median), r(log h_bare, log h_cluster) ≥ 0.5 over ≥ 30 spectra with both ≥ 150 counts, neither matched peak stamped as an isotope child, an artifact or another reading's M0 in > 50 % of its spectra, and no ¹³C carbon count (area, the scale calibrated on both ions of every committed-both neutral) contradicting either ion by more than max(1, 0.2 n). A fact about the neutral: **never an axis, never in `cross`**, never on an ion-only pair, never per file and never on a profile without a pair (a leaked fact would move the TOF and nitrate goldens, §8); a composed profile keeps the pair when exactly one pair is declared among its components (`Ur+EasyIC` does; two different pairs give none), and the fact then lifts the neutral's rows on every adduct of that profile. Written to `tables/neutral_pairs.csv` |
| `label_untie` | (rule K, C18, 2026-09-27) **batch only**, on a 15N-labelled nitrate channel (the profile's label is `^N` and it clusters on `[M+^NO3]-`: `NO3_15N`, or composed `NO3+NO3_15N`): the pair is a tied `[X+NO3]-` whose 14N line **tracks** X's 15N cluster and whose tie `batch/label_twins.py` therefore breaks in the cluster's favour. The 14N line ties by construction with the organonitrate `[X'-H]-` (X' = X + HNO3, the same ion); it tracks when its 14N/15N height ratio q(t) follows the batch's cluster ratio k_cl(t) — the per-spectrum median of q over the bright CHO acid clusters committed on `[M+^NO3]-` (the 15N line's median ≥ 1000 counts, both lines within 2 ppm in ≥ 100 spectra), the pair under test left out: median(q / k_cl) in 0.5–2, sd(log q / k_cl) ≤ 0.25, r(log q, log k_cl) ≥ 0.8 over ≥ 50 spectra — AND every file's tie on the reading is with same-ion decomposition aliases alone (the tier engine's own test). The pair's `tied` then reads False and the level falls through to the rows below (on a real acid, the branch: 3b). Written to `tables/label_twins.csv` |
| `label_alien` | (rule K) **batch only**, same scope: a committed `[X+NO3]-` whose 14N line is not established as X's cluster — `absent` (the 15N partner in ≤ 20 % of the ≥ 10 spectra of the 14N line), `untestable`, or `excess` (tested, the ratio above 2 × k_cl: cluster or organonitrate) with its 15N partner in < 80 % of its spectra; an excess line whose partner is present in ≥ 80 % still shows X's cluster through the partner and stays in the pools (the 2026-09-27 decision: the aged-SOA tracer C8H12O6 runs at 2.1 × with its partner in 98 %), though it is refuted itself (`label_veto`). An alien line is kept out of X's per-neutral pools **in both directions**, like an ion-only row: it gives X no `chan2` and no acid `branch`, and takes neither. A `consistent` line — tested, not above the cluster ratio (a noisy cluster, or one dominated by the reagent's own 14N, as trifluoroacetic acid's at 0.12 × k_cl; an organonitrate can only add 14N intensity) — stays X's cluster, untied |
| `label_veto` | (rule K) **batch only**, same scope: the pair is an `[Y+^NO3]-` reading whose **14N twin** — the line the reagent's own 14N impurity puts 0.99703 Da below every real cluster, at f = (1 − purity) / purity of its height (`ReagentProfile.purity`: 0.98 → 0.0204; the labelled reagent's nitrate dimer reads 0.0186 and trifluoroacetic acid's cluster 0.0206 on the regression batch) — is refuted, or a committed `[X+NO3]-` whose 14N line is `excess` or `absent` (above; `label_note` says which): for the 15N reading, over the spectra where the batch's own detection curve (the 13C lines of every committed M0 of ≥ 4 carbons, binned by expected height; a bin with no line takes the populated bin below it, 0 below the first) gives the twin p ≥ 0.5 and the twin's m/z is not below the spectrum's lowest peak (a twin under the scan start was never measured), E = Σ p ≥ 3 and the twin (within 2 ppm) is seen in ≤ 0.2 E of them. A hard input (§4 row 1); `label_note` carries the numbers. `passes` (≥ 0.5 E), `unclear` and `untestable` (E < 3) set nothing |
| `iso_veto` | (C11+a, 2026-09-27) **batch only**: `batch/iso_checks.py` finds, on the stamped batch time series, that the formula's own isotope claim is false. The instrument class is read off the batch's width model: **Orbitrap-class** when it resolves ≥ 50 000 at m/z 200 (`batch_summary.resolution.r_at_200`), else **TOF-class**; no width model is TOF-class. Three checks, any one refutes: **rule C** (Orbitrap-class only) — per stamped pair, not ion-only: in each spectrum the brightest M0 stamp and the nearest peak to it + 1.0033548 within 3 ppm; a spectrum counts when h0 × C × 0.010816 ≥ 5 × its noise edge (its 1st-percentile height), ≥ 8 spectra; the pooled area ratio (corrected for the width, ((m+1)/m)^1.5, and 17O) and the pooled height ratio each read a carbon count, divided by (1 + bias), the level-free median misread over every testable heteroatom-free pair; refuted when area **and** height both miss the ion's carbons by more than max(1.5, 0.25 C, 3 se) — a 'too many' reading whose M+1 slot is another named pair's line in > 20 % of the spectra is untestable; an ion below the scan start + 1 Da (the batch's lowest m/z) and a 14N `[M+NO3]-` beside the profile's 15N `[M+^NO3]-` (its M+1 sits 6.32 mDa above the taller 15N line) are not tested, nor an ion whose +1 line 13C does not dominate (13C < 50 % of it beside 29Si, 33S, 15N, 2H and 17O: a Si-rich ion's +1 line is mostly 29Si, 3.8 mDa lower — the 2026-09-27 decision). **REQ** — per pooled pair whose ion carries Br / Cl (TOF-class) or Br / Cl / S / Si (Orbitrap-class): the ion's count-aware fine structure, components closer than one FWHM of the width model merged into observable lines, placed relative to the **stamped** line (a Br2 ion is committed on 79Br81Br); required are every halogen group ≥ 0.25 × the stamped line (81Br per Br, 37Cl per Cl, the Br2 1:2:1 and Cl2 patterns) and, on an Orbitrap-class ion without Br or Cl, 34S / 29Si / 30Si where the component is ≥ 50 % of its line; a line is detectable in a spectrum when h0 × its ratio × the element's **line efficiency** ≥ 3 × the spectrum's height gate (noise edge × the batch's `height_cutoff_x_edge`) — the efficiency is the batch's own: per element, the median seen/theory height over the pairs whose line is present in ≥ half of ≥ 3 theory-detectable spectra, from ≥ 3 pairs, read within [0.25, 1] (1 without a measurement; a Br + Cl line takes the smaller; the 2026-09-27 decision: the uronium batch's Si lines run 0.58 × theory, their peaks 1.3–1.5 × wider), present when a peak sits within max(1 ppm, 4σ) of the blend centroid or the pure component (TOF-class: the stamp window **and** 20 ppm, absent only when absent at both); refuted when ≥ 3 spectra are detectable and the line is present in ≤ 20 % of them (a pair with no stamp takes the tallest peak within the stamp window of its pooled m/z). **HIGH** (variant V4) — per pooled pair with ≥ 8 spectra: at a heavy offset (81Br, 37Cl, 34S, 30Si, 18O, 13C2) the nearest peak within 1 ppm (TOF-class 10 ppm) co-varies with the M0 (r(log area) ≥ 0.8, present in ≥ 60 % of its spectra) at a pooled area **and** height ratio ≥ 3 × the ion's count-aware expected M+2 and ≥ 0.2 — unless the line is another committed M0 or a plain `13C` child in > 20 % of its spectra, sits nearer a non-isotopic +2 alias (F↔OH, C3↔F2, N2↔CH2O) than the heavy spacing (Orbitrap-class), or runs above expected + 3, or names an element the formula lacks that no ion at that mass can carry in the number the line implies (round(min(area, height ratio) / the per-atom height), ≥ 1): some CcHhNnOoSs rest (N ≤ 4, S ≤ 3, 0 ≤ H ≤ 2C + N + 4) must make up the rest of the M0 mass within 5 ppm (TOF-class 20 ppm) — the 2026-09-27 decision; an O2 ↔ H2S analogue sits 0.06 mDa from 81Br. A line several heavy spacings reach is the tallest one's, named by the spacing it sits nearest. A hard input (§4 row 1) that also keeps the pair out of its neutral's `chan2` / `branch` pools in both directions, as an alien 14N line; `iso_note` joins the refuting checks' notes with their numbers (`rule C: 13C reads 4.1 C for 6 …; REQ: the M+2 (81Br) line (0.97x the stamped line) absent in 12 of 12 detectable spectra …`). Never an axis, never in `cross`, never per file; it reaches an ion-only pair too, and a curated row: hard outranks curated. Written to `tables/iso_checks.csv` |
| `lead_lift` | (rule H, C11+b, 2026-09-28) **batch only**: a halogen lock lifts the pair's tentative lead. The lock is check `H` of `batch/iso_checks.py`, a POSITIVE fact on the stamped batch time series (never a veto): per pooled pair whose **ion** carries Cl or Br, in each spectrum the brightest M0 stamp and the nearest peak to it + the exact 37Cl − 35Cl (1.9970499) / 81Br − 79Br (1.9979521) spacing within **1 ppm** of the M0's m/z; it locks when that line is present in ≥ 60 % of ≥ 20 M0 spectra, r(log area) ≥ 0.8, the pooled area ratio in 0.65–1.45 × n × 0.3198 (Cl) / 0.9728 (Br) for the ion's count n, and the M0 is not itself a heavy line (`heavy`: a line one Cl or Br spacing below it, co-varying the same way, at the count window of 1–4 Cl or 1–2 Br). The **silicon test** (the 2026-09-28 decision, "the ²⁹Si line decides", its blended reading revised 2026-09-29; `si_rich`, never a veto): from `LOCK_SI_MZ` (≈ 206.3, where the ³⁰Si spacing, 0.206 mDa below ³⁷Cl, is 1 ppm of the M0's m/z — derived from the constants: the AME2020 ³⁰Si − ²⁸Si spacing 1.9968436; the isotope table REQ and HIGH read keeps 1.9968442, 0.6 µDa high: re-rounding it there would move HIGH `offset_mda` on 45 / 11 / 73 rows of the regression batches, two offset labels (the labelled-nitrate C5H8Cl2O2 `[M-H]-`, a rule H lock whose HIGH row names its Cl2 M+2 a ³⁰Si line seen in 8 % of its spectra, 30Si → 37Cl; the TOF's C11H10N2O14 `[M+NO3]-`, 37Cl → 30Si), `n_used` / ratio / r / presence on 10 rows, `other_m0` on 3 TOF HIGH rows and the note on 9 rows, no verdict — left for a card) a Cl lock is refused where the ion's M+1 region shows the ²⁹Si line a Si-rich ion making the partner from ³⁰Si must carry: n = ratio / 0.0335 silicons, ²⁹Si n × 0.0508 at +0.99957 Da (the isotope table's 0.9995683, 0.2 µDa above AME2020's 0.9995681), 3.79 mDa below ¹³C; a chlorinated ion carries none. Where the width model parts the two at the +1 m/z (FWHM < 3.79 mDa) the ²⁹Si line itself is looked for as the partner is (1 ppm, present in ≥ 60 %, r(log area) ≥ 0.8) at ≥ half its expected area — and where no such line is present (in < 60 % of the stamps) the peak picker may not have parted it from ¹³C after all (two lines at a ¹³C / ²⁹Si height ratio of ~0.4 part only from ~1.15 FWHM), so the +1 region decides as below (`si29_mode` `unparted`, 2026-09-29). Where they blend (and without a width model) the +1 region — every peak from 1 ppm below ²⁹Si to 1 ppm above ¹³C — must be present in ≥ 60 % and co-vary with the M0 like the partner (r(log region area, log M0 area) ≥ 0.8 over the spectra it is present in: a Si-rich ion's ²⁹Si line is its own isotope line, and an anti-correlated region is not), and then its **position** decides (the 2026-09-29 decision): refused where the region's area-weighted position, the median over the spectra, sits ≥ half-way from the reading's own +1 position (¹³C, ²H, ¹⁵N, ¹⁷O, ³³S and the reading's own ²⁹Si, height-weighted) toward the blend its own +1 line and the implied ²⁹Si line make. A reading that itself carries 1–2 Si is refused on its **excess** too (the Si-reading guard): the region reading ≥ half the implied ²⁹Si area above that reading's own +1 line (pooled, sum over sum) — its own ²⁹Si pulls the half-way mark toward ²⁹Si, so its position can fall short (the D7 siloxane's urea cluster read as its N8 S Si2 Cl formula misses the mark by 0.06 mDa and reads 1.01 × the half ²⁹Si). The guard is an OR — a Si reading is refused on the position **or** its excess: the decision's "a Si reading must ALSO show the excess", read as OR and confirmed by the user 2026-09-29. Read as an AND it would ask a Si reading for more than a Si-free one and reopen the Si2 corner it closes: 393 of 852 synthetic blended scenario-2 / D7-urea cases no longer refused, every one a Si1 / Si2 reading — the dim scenario-2 Si1 and Si2 readings rule C cannot read (their excess 0.87 × / 0.72 × of the need) and the Si2 corner among them — and on the siloxane control the 73 Si readings the position refuses no longer refused by the silicon test (115 → 42; rule C still refutes 64 of them, so 9 more misreads go unrefuted: 61 → 70). Where both criteria hold the `si_rich` note names the position. Why the position and not the height, in plain words: the first build refused on the excess AND the position, on the premise that rule C refutes the carbon-rich readings whose own ¹³C line absorbs the excess, leaving only an organosilicon above m/z ~600 read with more than ~40 C. Refute round 2 showed the premise wrong twice over: rule C reads no ion too dim for its ¹³C line (h0 × C × 0.0108 ≥ 5 × the noise edge in ≥ 8 spectra), so a dim siloxane read as a carbon-rich Cl formula locked with nothing refuting it at any m/z from 206.3; and the regression batches' peak picker reports a siloxane's +1 blend as ONE peak near its apex (0.3–1.0 mDa above ²⁹Si) carrying 72–84 % of its area, so the excess comes out short even where the reading adds no carbon. The position survives both: whatever the picker reports, a Si-rich ion's +1 peak sits toward ²⁹Si and a chlorinated ion's at its own ¹³C position. Measured (the re-measure after refute round 2): the excess test let 48 of 52 dim readings of the uronium batch's real D8 / D9 urea clusters through (lock gates forced) and up to 173 of 263 synthetic Si7–Si12 readings, the position test none but one Si2 reading, which the guard refuses; neither refuses any of the 12 real Cl ions of the batches above m/z 206.3 that show a ³⁷Cl line, and neither moves a level on the three regression batches; placed on ~3 900 real +1 regions, a real Cl ion is refused 1.0–1.4 times per 1000 by the position (0.5–0.9 by the excess test). What the test cannot refuse: a misread whose +1 region is absent from more than 40 % of its spectra or does not co-vary with its M0; what it can wrongly refuse: a real Cl ion into whose dim +1 region another ion's line near ²⁹Si merges while its own ¹³C line supplies the co-variation; and, through the Si-reading guard, which asks no position, a real Si1–2 Cl reading with any excess in its +1 region, a neighbour's line at the ¹³C position included (0 / 0 / 1 of the 30 / 12 / 37 real Si1–2 ions of the regression batches from m/z 206.3, forced blended with a one-Cl partner — the one a TOF C5H7NO2Si `[M+Br]-`: its region reads 4.1 × the excess the guard needs while it sits 2.22 mDa above ²⁹Si, short of its mark at 1.17 — a Br adduct, whose Cl reading would carry both halogens and which rule H therefore never tests; on the readings it would test, 0 / 0 / 0). On a TOF-class batch the position is no silicon signature: its peak list reads an ion's +1 region ~1.3 mDa below the ion's own +1 position (the regression TOF's median over its 69 present regions; ~1.4 below the ¹³C spacing), and the seven Si-free ions only the position fires on there 1.4–2.6 mDa below theirs (the area-weighted region, the median over the spectra; 1.7–2.5 mDa below the ¹³C spacing, 1.2–2.1 mDa above ²⁹Si) — harmless while no TOF Cl lock forms above m/z 206.3 (the batch's one TOF lock, C2H3ClO2 `[M+NO3]-` at m/z 156, sits below the silicon test's threshold); a TOF Cl lock above it would need its own margin. Areas, not heights, for the excess and the position's weights: the partner's ratio and the silicons it implies are area ratios, and a blend's height is not the sum of its lines' either. A real Cl lock keeps the full window at every mass; the same plain window covers the other +2 look-alikes a line-position guard once weighed — the non-isotopic C3 ↔ F2 swap (+1.996806, 0.244 mDa below ³⁷Cl, inside 1 ppm from m/z ~244) and ³⁴S (from m/z ~1250) — with no position test (the 2026-09-29 decision): a line there must still pass the lock's co-variation and count-window gates, which stand in for it, while HIGH keeps its alias guard for its own purpose (on the regression batches no pair from m/z 244 has a gate-passing line at the ³⁷Cl spacing nearer the C3 ↔ F2 alias: the only gate-passing lines there, three Cl-free 15N-twin readings rule H does not test, sit 0.03–0.09 mDa below ³⁷Cl); ⁸¹Br needs no test (³⁰Si sits 1.11 mDa below it, outside 1 ppm below m/z ~1100). `si_n`, `si29_expected`, `si29_seen` (blended / unparted: the region's excess over the reading's own +1 line) and `si29_mode` (`resolved` / `unparted` / `blended`) record it; a `si_rich` note gives the region's position and the half-way mark, or says a Si reading was refused on its excess. Never locked (`untestable`): an ion with Si ≥ 3, one carrying both Cl and Br (its M+2 is a 37Cl / 81Br blend), a pair with no M0 stamp, one stamped on a heavy isotopologue (> 0.5 Da above its all-light m/z), one stamped in < 20 spectra. The halogen must be the sample's, decided by the **batch's reagent**, not the adduct label (2026-09-28): a line counts only where the ion carries more of the element than one reagent ion supplies — the most any adduct of the profile adds (a bromide profile's `[M+HBr+Br]-`: 2 Br; a nitrate or uronium profile: none); otherwise `reagent`. So on a bromide batch no ion carrying two Br locks, whatever its adduct — a Br2 ion may be `[M+HBr+Br]-` of a Br-free neutral — while a Br3 ion can. The ion BrHNO3− reads HNO3 `[M+Br]-` or HBr `[M+NO3]-`: on a bromide batch neither locks, on the labelled-nitrate batch HBr `[M+^NO3]-` does. The **lift**: a locked pair whose lead rows were **all** set by a setter the lock answers (`lead_by`: the three speculative-residual reasons, the reference-list dim rescue — the lock is the isotope confirmation it lacked — and the element budget **only** where the locked halogen is the neutral's sole violation: `budget_ok`, the neutral against the batch context's budget with that halogen's cap lifted, the CF2 exemption's construction; a lead row naming no setter — a ledger written before `lead_by` — may be the budget's and lifts only where `budget_ok`), no row below assignability, not ion-only or alien, and neither an isotope check nor rule K refutes the reading. The radical anion and the reagent-N re-read are never lifted. On a lifted pair `lead` reads False, `iso` True, `reagent_only_iso` False, and `anchor` / `chan2` / `branch` are read over unflagged rows only; `multiline`, `carbon_ev`, `known_fam`, `tied` and `lowconf` stay (a tie or a Low score still holds, the reason losing "below assignability"). Every other pair's facts are unchanged — the lifted pair's own flagged rows still give their siblings what they gave. Rule C is the only 13C hold. Never an axis, never in `cross`, never per file; `lock_note` carries the lock's numbers. Written to `tables/iso_checks.csv` (the `H` rows) |
| `ion_only` | the pair was written by the **ion-only stage** (`ion_only`, C7): adduct `[M]-.` with method `ion_only:*` (a merged ledger: an `ion_only_of` link) — the +1.0078 Da electron-attachment line beside a committed `[M-H]-` acid, carrying the acid's composition. An ion-only pair is levelled on its **own** satellite alone (§4 row 2b′) and is kept **out of the per-neutral pools in both directions**: `chan2`, `branch`, `anchor` and `reagent_only_iso` are computed over the regular rows only, so the row never gives its parent a second channel and never takes an axis from it; it is never `corroborated` and its neutral never enters a cross set (`corroborating_neutrals` skips it) |

Derived:

```
n_axes         = iso + chan2 + anchor + corroborated            # 0..4
cross          = corroborated or multiline or known_fam != ''    # an axis outside this channel's ionization chemistry
neutral_backed = corroborated or chan2 or anchor or known_fam != '' or carbon_ev
degenerate     = saturated or degeneracy >= 3
unique         = degeneracy <= 1                                  # NaN is neither
hard           = tied or below or lead or lowconf or label_veto or iso_veto
                 # lead: C19(c), False where rule H lifts it (batch only);
                 # label_veto: rule K, batch only; iso_veto: C11+, batch only
```

Two of these terms can never decide a level today and are kept because the
reference script states both (verified by mutation, B4): `carbon_ev` inside
`neutral_backed`, and the `not carbon_ev` guard of `reagent_only_iso` —
`reagent_only_iso` already requires every satellite tag to be the reagent
halogen's, so a pair with a ¹³C line never reaches the 4d test at all.

`n_files` (files of the source that carry the pair) is **recorded** in
`evidence_axes` but is **not an axis**: the golden vectors were measured
without it, and whether persistence across files should count is a
calibration question for B3, decided on roster hit rate and decoy rate.

### 3.1 The reagent halogen

The halogen of the source's **commonest cluster adduct** (`[M+Br]-` → Br;
`[M+NO3]-` → none), never of "any halogen adduct present": a nitrate channel
with two stray `[M+Br]-` rows against 346 `[M+NO3]-` is a nitrate channel.

## 4. The decision table — first predicate that holds wins

| order | level | predicate | meaning |
|---:|---|---|---|
| 1 | **5b** | `hard` | the assignment argues with itself: a near-tie the arbiter broke, a row below assignability or a tentative lead (both read "below assignability" in `level_reason`; a lead a halogen lock lifts is not hard, rule H, batch only), a score the engine calls Low/Suspect, (rule K, batch only) the reagent's two isotopologues refute the cluster reading, or (C11+, batch only) an isotope check of the batch's time series refutes the formula (`5b: an isotope check refutes the formula (<note>)`) |
| 1′ | **4d** / **5a** | `ion_only` | an ion-only row: **4d** when its own satellite passes the band (`iso`) — the composition is pinned by exact mass and ¹³C, the ionization process and the neutral are open — else **5a**, exact mass only. The same rung the reagent-halogen case (row 8) reaches by the other route: 4d = *ion pinned, neutral not*, by either route |
| 2 | **5b** | `degenerate and n_axes == 0` | mass-degenerate with nothing to break the tie |
| 3 | **2b** | `known_fam != ''` and `scope(known_fam) == "compound"` and `n_plausible_structures == 1` | a curated **identity** on a formula that admits one structure |
| 4 | **3a** | `known_fam != ''` (any other curated commit) | a named class, isomers open |
| 5 | **3b** | `branch` | a substituent only: the same neutral deprotonated and clustered means an acidic hydrogen, nothing more |
| 6 | **4c** | `n_axes == 0 and unique and res_ok` | formula unopposed — one plausible ion in the calibrated window on a separable peak — but nothing corroborates it |
| 7 | **5a** | `n_axes == 0` | exact mass only; no discriminating test was possible |
| 8 | **4d** | `not neutral_backed and reagent_only_iso` | **ion** formula only: the sole isotope support is the reagent halogen, which pins the ion and says nothing about the neutral (CIMS-specific; no Schymanski analogue). With row 1′, 4d reads *ion pinned, neutral not* whichever route reached it |
| 9 | **4a** | `n_axes >= 2 and cross` | formula confirmed **and the neutral established**: two orthogonal axes, at least one from outside this channel's ionization chemistry |
| 9′ | **4a** | `upair and (iso or (unique and res_ok))` | (rule U) the neutral established by the channel's **neutral pair** — the bare and the reagent-cluster ion of one neutral, seen together at exact mass and co-varying, the way the acid branch (row 5) establishes it on the anion channels — on a formula with its own support: an isotope line, or one plausible ion on a resolved peak. Placed after row 9 so no existing 4a changes its reason; 4a, not 3b: the pair makes no functional-group claim |
| 10 | **4b** | else | formula confirmed, one corroboration |

Levels **1** and **2a** are in `LEVEL_ORDER` and never assigned.

### 4.1 Level 2b needs both an identity claim and a unique structure

`scope(family)` is a property of the pass-0 registry, not of the level: a
family whose entries are hand-listed compounds (`atmospheric`,
`reactive_iodine`, `ambient_inorganic`, `nitroaromatic`, `cyclosiloxane`,
`indoor_sulfur`, `organophosphate`, `organothiophosphate`, `easyic_hydride`)
asserts a **compound**; a family generated from a formula loop
(`perfluoroacid` CₙHF₂ₙ₋₁O₂, `chlorinated_paraffin` CₙH₂ₙ₊₂₋ₓClₓ,
`contaminant:silanediol`) asserts a **class**. `n_plausible_structures` is
read from `peaky/data/isomer_space.csv` (§5); a formula absent from it has
`NA` and can never be 2b.

The boundary case is trifluoroacetic acid: CF₃COOH is the only structure of
C₂HF₃O₂, so the isomer space says 1 — and it stays **3a**, because the
perfluoroacid family's commit says "a PFCA of this formula", not "TFA". A
per-compound entry (or a standard, level 1) would lift it. This is why 2b
is "from isomer space, never a whitelist": the level rule reads a structure
count and a registry scope, and no list of formulas lives in `evidence.py`.

Nitrophenol (C₆H₅NO₃) is the other direction: a compound-scope family, but
the isomer space says 3 (2-, 3-, 4-nitrophenol), so 3a.

### 4.2 Settled decisions

- **One rating channel.** A level rates a `(neutral, adduct)` pair as seen
  through one channel; a second channel is an *axis* (`chan2`), not a second
  rating. The batch stamps the same level on every merged row of the pair.
- **Cross-reagent (and cross-instrument) agreement is formula evidence, never
  more.** It can carry a row to 4a; it cannot make 3b, 3a or 2b, which need
  a structural claim the mass cannot give.
- **Reagent ions are excluded by role.** `role == reagent` rows never carry
  a level; a reagent cluster that a pass mis-committed as an analyte is the
  reagent stage's bug, not a level.
- **The isomer ceiling is stated per row.** `n_plausible_structures` is
  written on every M0 row whose formula is in the isomer space (`NA`
  otherwise). It caps 2b and is reported so a reader sees that a 4a
  C₁₀H₁₆O₃ is one formula and, by a chemist's count, dozens of structures.
- **Resolvability binds only where it was measured.** Every run with a width
  model (measured from the raw profile, or declared) produces it; a run without
  one (offline, no `--resolving-power`) is not denied 4c for lacking it.
- **Isotope evidence is judged on physics.** A satellite whose ratio is
  outside 0.5–2× natural abundance is not evidence; a satellite the server
  attributed (`isotopologues`) is.
- **Two isotope lines speak for the neutral only as two of its elements
  (C17).** `multiline` is the only outside term a single channel has, so it
  must say something the other axes cannot: the ion's composition in two
  independent elements, each supplied mostly by the neutral, each at a height
  its natural abundance allows. One element seen twice (13C and 13C2), the
  reagent's own isotope (the urea's 15N, the bromide's 81Br, nitrate's 18O on a
  C1 neutral) or a line with no element (`M+n`) is not a second line, and an
  ion carrying Br or Cl has no measured 18O line (the halogen's 81Br / 37Cl line
  owns its M+2 region; on a TOF the two are one peak). On the regression runs it
  moved 1 row (labelled nitrate), 10 (uronium) and 15 (TOF) from 4a to 4b; rule U
  re-establishes 8 of the 10 uronium rows (C6H8O3 and C9H16O2 urea adducts stay
  4b: a 13C contradiction and no co-variation).
- **The neutral pair establishes the neutral where the chemistry declares
  one (rule U).** A uronium channel has no acid branch; its equivalent is the
  protonated and the urea-clustered ion of one neutral at exact mass,
  co-varying through the batch. Its specificity comes from exact mass and the
  formula support (round 3: 0.25 neutrals per decoy offset before the stamp
  clause; as built, 0 at each of 12 offsets on the uronium run), not from the
  co-variation, which is a weak veto (wrong partners pass r >= 0.5 in most
  cases). Two batch-specific vetoes of
  the round-3 design -- line proportionality through hand-dated steady states and
  a bright-parent guard read at those steps -- are not built: no batch-generic
  form reproduced them (the user's decision, 2026-09-27). The neutrals whose
  protonated line also carries an in-source fragment of a brighter parent
  (C₁₀H₁₆O from a hydroperoxide, C₁₄H₂₈O₅ from TEG-EH) stay 4a; the fragment is
  an intensity note, not a level.
- **On a labelled-nitrate channel the reagent's two isotopologues arbitrate
  each other's reading (rule K, C18).** The 14N and the 15N line of one
  cluster are one ion seen twice: one channel, never two, and each tests the
  other's reading in its own direction. A committed 14N `[X+NO3]-` ties by
  construction with the organonitrate of the same ion. It is X's cluster when
  its 14N/15N ratio follows the batch's cluster ratio through the run (it
  tracks: the tie is cleared where it is with that alias alone) or at least
  does not exceed it (consistent: an organonitrate can only add 14N, so a
  noisy or reagent-dominated cluster stays X's). A line above twice the
  cluster ratio, or with no 15N partner at all, is cluster or organonitrate
  undecided: it is refuted (5b); an absent or untestable line gives X nothing
  -- no second channel, no acid branch -- and so does an excess line unless its
  15N partner is present in >= 80 % of its spectra (the cluster is then shown
  by the partner, as for the aged-SOA tracer C8H12O6). On the regression batch: 14 of 40
  co-detected pairs track (8 tied rows untie 5b → 3b, 0.179 % of the signal;
  mismatched X14 / Y15 pairs pass 42 of 600, organonitrate lines 3 of 150); of
  the 291 committed 14N lines 18 are consistent, 13 excess, 220 have no 15N
  partner and 26 are untestable; the C11–C12 `[M-H]-` acids whose only
  cluster was a 14N line with no 15N partner in any spectrum leave the acid
  branch (3b → 4b, "ion pinned, neutral open"; identified signal 48.3 → 45.1 %
  with the rest of the rule). A labelled `[Y+^NO3]-` without the 14N twin its
  reagent's impurity must carry, where the batch would have seen it, is refuted
  (2 readings, 4b → 5b: 0 of 42 / 53 expected detections; the same two for f
  from 0.0150 to 0.0236, none of the 16 13C-backed clusters of identified
  acids, and every testable N-free `[M-H]-` control is refuted by the same
  test). The twin fraction is the profile's reagent purity, not a reagent-ion
  scan (the batch's scan starts above the reagent dimer). All of it is a fact
  of the pooled batch series, never an axis, never in `cross`, never per file:
  the per-file level, the tier and the merge vote keep the arbiter's tie, and
  the merged row reads the pooled level (§6.2). A Cl-free labelled reading
  that carries a locked 37Cl line is left to the isotope checks (C11+), not
  to this rule.
- **A formula's own isotope lines are tested on the batch series (C11+a).**
  The per-file isotope test only asks whether a child it found sits in the
  band; three batch checks ask what the formula claims and refute it where the
  series says otherwise (§3 `iso_veto`). Numbers against the trunk with rule K
  (C18 + K, C19(c); the engine replay equals the runs row for row), identified /
  ion signal in % of the batch series:
  - **rule C** (the 13C carbon count, Orbitrap-class only). Labelled-nitrate
    Orbitrap: 19 pairs refuted; with the pool exclusion 11 merged rows move
    (4 identified → 5b, 3 ion → 5b, 4 acid-branch siblings 3b → 4b: brominated
    readings whose sibling adduct was their only branch), identified 45.148 →
    44.713 %, ion 33.224 → 32.711. Uronium Orbitrap: 2 refuted, 1 move
    (C24H40N2O3 urea cluster 5a → 5b, reads 11.6 C for 25). The TOF batch is
    TOF-class: no rule C. 0 of 62 roster or well-known species tested are
    refuted (pinic acid, MBTCA, pinonic acid, TFA, DEHP read their own carbon
    counts); decoys read at C − 2 are refuted in ~50 %, at C + 2 in ≤ 7.5 % — a
    veto for a gross carbon over-claim, never a confirmation. A Si-rich ion is
    not read: three Si10–Si12 siloxane readings of the uronium batch read
    "−0.2 C" on a +1 line 13C makes 25 % of, the rest 29Si (all already 5b).
  - **REQ** (a required heavy line absent). Labelled nitrate: 152 pooled / 104
    merged pairs refuted (Br 62, Cl 39, S 3), 71 merged rows move, identified
    −0.774, ion −0.843; genuine Br / Cl lines there read 1.01–1.13 × theory
    (efficiency 1), their offsets scatter 0.04–0.36 ppm, and 55 of the 63
    directly refuted pairs stay absent out to 3 ppm. Uronium: 3 refuted, 2
    moves (C12H14N2O4Si `[M+H]+` 4a → 5b: no 29Si in 319 of 319 spectra at
    6.7 × the floor, and C13H15N2O3S+ fits at +0.05 ppm with its 34S line;
    identified −0.257); its Si lines run 0.58 × theory, so the cyclosiloxanes
    D7·urea (curated 2b) and D5 `[M+H]+` (4b), whose 29Si line would sit near
    1.5 × the floor, are untestable, not refuted. The TOF: 94 pooled / 27
    merged refuted at both windows (Br efficiency 0.96, Cl 0.70), identified
    0.000, ion −0.126 — its blends displace real 81Br lines 10–19 ppm, so the
    stamp window alone would refute 39 merged, three identified among them
    (measured before the efficiency); hence the two windows. REQ tests presence only: a line far below its
    required height, or another pair's M0, passes — the yields are a lower
    bound (count-aware heights: C11+c).
  - **HIGH** (a heavy line too high for the formula, variant V4). Labelled
    nitrate: 11 refuted, 5 merged moves (4b → 5b 4, 5a → 5b 1), ion −0.202 —
    aromatic `[M+^NO3]-` readings whose 37Cl line (0.26–0.46 ×, the exact
    spacing, r 0.87–0.98) says they are chloride adducts of α-pinene products
    (`[C10H18O7+Cl]-` …; C2 + 15N ↔ H4 + Cl is a 0.044 mDa alias), diethyl
    phthalate's `[M+^NO3]-` reading among them. Uronium: none (C4H6O·urea's
    0.68 × line 1.998 Da up cannot be bromine at m/z 131.08). TOF: 64 / 23
    refuted, 6 merged moves (4b → 5b 4, 4d → 5b 2: C6H11NO6S and C7H19N3O6Si
    `[M+NO3]-`, Br-free ions whose own 1:1 +2 Da doublet refutes them; three
    more such `[M-H]-` pairs are refuted, not merged). Identified 0 on all three; under V4, 0
    decoy flags per ±4 / 8 mDa offset on both Orbitraps.
  - all three together: identified / ion / tentative 45.148 / 33.224 / 10.072
    → 44.374 / 32.045 / 12.024 (labelled nitrate, 76 merged moves), 73.822 /
    12.552 / 0.826 → 73.565 / 12.549 / 1.086 (uronium, 3), 11.207 / 4.054 /
    3.862 → 11.207 / 3.907 / 4.009 (TOF, 25). Measured first on the trunk
    before rule K (C17 + U): 48.266 / 30.879 → 47.319 / 28.967 on the
    labelled nitrate; the 15 merged rows that differ are rows rule K already
    refutes (a pair is refuted once, by whichever rule comes first). The
    merge vote is not re-run: a refuted reading is not replaced by its
    runner-up here.
- **A reagent satellite needs the reagent in the ion (C11+a).** 4d reads
  "the ion is pinned by the reagent's own halogen"; that holds only where the
  reagent put the halogen there. A brominated neutral's `[M-H]-` or
  `[M+NO3]-` carries its OWN Br, and its 81Br line is the neutral's: the flag
  clears. An ion carrying NONE of the reagent halogen keeps the flag (the
  2026-09-27 hold): a 1:1 +2 Da line on a Br-free ion is no isotope line of
  that formula — it says the ion carries a Br the formula lacks (C35H32O4S
  `[M-H]-` at m/z 547.195: 1.00 × its M0 in 39 of 39 spectra, where the
  formula makes 0.12 ×) — and would read 4b on a line that refutes it; HIGH
  refutes such a reading on the batch, and C11+c's count-aware band will
  judge it per file. Per file, 23 M0 rows (15 pairs) of the bromide/nitrate
  TOF batch move 4d → 4b, all brominated neutrals' `[M+NO3]-` (e.g.
  hypobromous acid, HBrO·NO3- at m/z 157.909); pooled 13 pairs (8 merged);
  the Orbitrap batches do not move.
- **A halogen lock lifts a tentative lead (C11+b, rule H, 2026-09-28).** A
  lead is unsupported, not contradicted; where the batch's series shows the
  ion's exact ³⁷Cl / ⁸¹Br line at its count, co-varying, the formula has the
  isotope support the lead lacked (§3 `lead_lift`). The lift is pooled and
  batch only: the per-file rows keep their flag and their own level, the merged
  row reads the lifted level (as rule K's untie). Measured on the three
  regression batches (the C11+a final runs, the H rows recomputed from each
  run's series: the C / REQ / HIGH rows reproduce byte for byte; three
  implementations agree pair for pair):
  - **the check**: labelled nitrate 282 pairs tested, 16 locked (173 no lock,
    92 untestable, 1 heavy); uronium 8 / 1; the TOF 2620 / 1 — its bromide
    reagent could have put the Br into 659 ions (`reagent`), and 114 carry both
    halogens. The TOF's own 81Br partners of 609 `[M+Br]-` pairs sit 6.8 / 11.4
    / 21.8 ppm off the exact spacing (50th / 68th / 90th percentile): at 1 ppm
    rule H is **Orbitrap-only in practice**. Widening finds no real TOF lock at
    3, 5 or 10 ppm, and at 10 ppm a +8 mDa decoy locks, so the window stays
    1 ppm on both classes. The silicon test (the 2026-09-28 decision, "the
    ²⁹Si line decides", which replaced a line-position guard; its blended
    reading the position since 2026-09-29) reads the four Cl locks above m/z
    206.3 — C7H11ClO5, C6H10Cl2O4, C10H19ClO3 `[M-H]-` (m/z 209.02, 214.99,
    221.10, labelled nitrate) and C7H11ClO2 `[M+(CH4N2O)H]+` (223.08,
    uronium), all where the width model parts ²⁹Si from ¹³C — and finds no
    ²⁹Si line at its own position on any; their +1 region, which then decides
    (`unparted`), sits at the ions' own ¹³C position, 1.5–1.8 mDa short of
    the half-way mark (present in 57 / 2 / 75 / 1 % of their spectra), and
    reads −0.04 to +0.03 × above their own +1 line against the 0.26–0.54 × a
    Si reading would need: the 16 / 1 / 1 locks and the five lifts stand, and
    the three tables are identical under the excess test and the position
    test. On the uronium batch's own siloxanes no Cl lock can form: none
    passes the one-Cl gates at 1 ppm over its spectra (the D7 urea cluster
    and D5 `[M+H]+` are too dim for their ³⁰Si line to show, in 0 % of their
    spectra; the D5 urea cluster's M+2 reads 0.17 ×, below the one-Cl
    window's 0.21, at r 0.65–0.70; the Si10 and Si11 urea clusters' ³⁰Si +
    ²⁹Si₂ blend sits 0.5–0.9 ppm off, present at the ³⁷Cl spacing in 46 % /
    1 % of their spectra; the D3 lines do not co-vary, their M+2 at r
    0.26–0.28). As a control of the test alone, each siloxane line's M+2 was
    handed over as the Cl partner (within 1 ppm of the ³⁷Cl or the ³⁰Si
    spacing where present in ≥ 60 % of the spectra, else taken within 3 ppm
    of the ³⁰Si spacing where 1 ppm finds it in < 60 % of the spectra — Si10
    24 %, Si11 0 % at the ³⁰Si spacing, 46 % / 1 % at the ³⁷Cl one) and read
    as every CHNOS(+Si ≤ 2)+Cl formula
    within 1 ppm, over all, the bright and the dim half of the spectra: of 224
    such cases the position test refuses 115 and rule C refutes 48 more; the
    other 61 pass, every one with a +1 region the gates cannot read (the D7
    urea cluster's in 26 % of its spectra, the D5 urea one's dim half at r
    0.59). The excess test refused 2 of the 224, left rule C 150 and let 72
    pass — dim readings rule C cannot test among them. Where the real lock
    gates pass (the Si12 line's bright half, 11 readings) rule C refutes every
    one. Over the full series (and its bright half) rule C refutes every one
    of the 37 Cl1 / Cl2 readings (23 one-Cl) of the D5, Si10 and Si11 urea
    clusters within 1 ppm, and Si12's 19 (11): their blend holds no ¹³C line
    at the exact spacing in any used spectrum; in the dim half rule C is
    untestable on the D5 urea cluster's 3, 2 of Si10's and all 19 of
    Si12's. The D7 urea cluster's
    readings are rule-C-untestable, and only the lock's 1 ppm gate stops that
    cluster (its M+2 line is at the ³⁷Cl spacing in 0 % of its spectra, its
    +1 region present in 26 %). On the real Si-free ions of the two Orbitrap
    batches, forced into the blended regime with a one-Cl partner at the
    bottom of the window, the position test fires on 0 and 1, as the excess
    test did — the co-variation gate's work: without it the excess test fired
    on 23 and 14 from m/z 206.3 (37 → 1; the 24th labelled-nitrate fire of
    the first count, 38 → 1, sat at m/z 206.04 and left with the threshold
    once the AME2020 ³⁰Si spacing moved it), the position test on 43 of the
    277 with a +1 region. On the TOF, blended at every mass, the position
    test fires on 24 (the excess test 17): the seven more are Si-free ions
    whose +1 region sits 1.4–2.6 mDa below the ion's own +1 position (the
    area-weighted region, the median over the spectra; 1.7–2.5 mDa below the
    ¹³C spacing, 1.2–2.1 mDa above ²⁹Si); the rest read centroid scatter and
    other ions' committed lines. The pooled-only ³⁷Cl
    reading at m/z 197.022 (C6H11ClO5 `[M-H]-`, whose line the batch stamps as
    C8H6O2 `[M+^NO3]-`) has no M0 stamp and is `untestable`.
  - **the lift**: labelled nitrate C6H9ClO3, C7H11ClO5, C6H10Cl2O4 `[M-H]-`
    (speculative series gap-fills; ³⁵Cl + 4 H − 2 C sits within 0.044 mDa of
    ¹⁵N, so each has a 15N-nitrate twin with one Cl fewer — C9H6O2
    `[M+^NO3]-` for C7H11ClO5, the one-Cl C8H5ClO `[M+^NO3]-` for
    C6H10Cl2O4 — and the ³⁷Cl line and its count are the only evidence that
    tells them apart) and HBr `[M+^NO3]-` (a gap in
    a C2H2O series built down from C4H5BrO2; its series anchor, on the flagged
    rows only, is dropped), 5b → 4b "one corroboration (iso)": ion 32.045 →
    32.158 %, tentative 12.024 → 11.912; uronium C7H11ClO2 `[M+(CH4N2O)H]+`
    (outside the uronium element budget, Cl its only violation; its `[M+H]+`
    second channel, too dim to show the ³⁷Cl line, is dropped) 5b → 4b, ion
    12.549 → 12.576; the TOF none. Identified 0 on all three; every other pair
    identical in level, reason and axes. The four chlorinated ones were rule C
    `untestable` (no spectrum bright enough; HBr carries no carbon and has no
    rule C row): rule C is the only 13C hold, so C7H11ClO5, whose pooled-area
    13C reading of 10.4 C blends another ion's line, lifts.
  - **controls**: a lock at ±4 / 8 / 12 / 16 mDa around both spacings moves
    nothing on the three batches; the per-file isotope axis as the gate
    instead would move pairs with no batch lock (C5H9ClO4 `[M-H]-`, whose
    37Cl line is in 13 % of its spectra at r 0.37). Relaxed gates (2 ppm,
    presence 0.5, r 0.7, window 0.5–2) add nothing.
  - **per setter** (lead pairs, not below, with Cl / Br in the neutral):
    speculative gap-fill 40 / 0 / 12 (labelled nitrate / uronium / TOF; the 4
    labelled-nitrate lifts), N ≥ 3 with no isotope 2 / 0 / 6, off-budget 10 /
    8 / 134 (the uronium lift; the TOF's perfect 1:1 lines on C4H8N2S3 and
    C6H15Cl2N2PS2 `[M+Br]-` are the reagent's Br and a mixed-halogen ion, not
    locks), reference-list dim 0 / 0 / 0; the radical anion and the reagent-N
    re-read carry no halogen by construction.
  A lead on one file of many still makes the pair hard (the pooled `lead` is
  any row): two of the labelled-nitrate lifts are leads on 1 of 8 and 1 of 7
  files. A majority rule for that is its own card, not rule H.
- **Predicate order is the design.** `hard` outranks a curated identity: a
  known species the arbiter had to tie-break is 5b, and that is a defect in
  the pass-0 lock to be fixed there, not hidden here.
- **The tier is not an input.** Levels and tiers are computed from the same
  columns and may disagree; B3 measures where.
- **A formula outside the run's element budget is not evidence of the neutral
  (C9).** A pass that widens the search -- a multi-channel certificate, a series
  extrapolation, a chain-opened contaminant family -- can commit a formula the
  context's element budget excludes (P, F or I in ambient air, S > 1, ...). The
  channels or the series step that proposed it are exactly the `chan2` / acid
  branch / `anchor` facts a level counts, so read back they would rate the
  formula 3b or 4b on its own proposal. The `plausibility` stage flags such a
  commit a `tentative_lead` unless a curated list names it (C19(c); before the
  split it was `below_assignability`): nothing tests the composition either way,
  so it is unsupported, not contradicted. The level reads a lead as 5b through
  `hard`, as it read the old flag -- no predicate here changes, and the golden
  fixtures (written before the stage) do not move. Fluorine alone has an
  exemption: a formula whose only violation is F keeps its level when the ledger
  commits its CF2 neighbour on the same adduct -- the CF2 step is the one piece
  of element-specific evidence a monoisotopic element can have. Chlorine and
  bromine have theirs in the isotope line: on the pooled batch a halogen lock
  lifts such a lead where the locked halogen is the formula's only violation
  (rule H, C11+b, above; docs/ASSIGNMENT_DETAIL.md §8.5).

## 5. `peaky/data/isomer_space.csv`

`formula, n_plausible_structures, family, name, rationale`. One row per
formula a pass-0 family can commit, with the count of structures a chemist
would accept for that formula in this chemistry and the reason, in words.
Seeded in B1 with the 2b and 3a formulas of the three golden sets and the
positive-mode families; B2 loads it through `peaky.paths.pkg_data`. Adding a
row needs a rationale; a row without one is a whitelist entry and is refused
by `tests/test_evidence.py::test_isomer_space_rows_carry_a_rationale`.

## 6. Where it runs

### 6.1 Single sample — a stage in `assign.run`

A new `_Stage("evidence", …)` in `peaky/assignment/assign.py::_STAGES`,
placed **after `iso_env_final` and before `timeseries`**: after every tier
and demote stage (the last of which is `plausibility`), after
`reflist_rescue` (which commits M0 rows post-tier), after `ion_only` (C7:
the electron-attachment rows, post-tier like the rescue, placed before the
final sweep so that sweep claims each new row's own ¹³C — the satellite row
1′ reads), and after the final isotope sweep (which adds the satellites the
`iso` axis reads).
`timeseries` writes only `ts_*` dispositions and is not an input.
`safe=False` (a level that cannot be computed is a bug, not a lost stage),
`store=True` (the stage summary is kept in the run's `summaries` and written
to the single-sample `<prefix>_manifest.json`; a batch reports the pooled
recompute of §6.2 in `batch_summary.json` instead).

The stage calls `evidence.apply_levels(ledger, cfg=cfg, cross=cross)` which
writes the five columns of §7 in place (`claim` read off the level, §1.1; `NA`
on every non-M0 row like the other four) and returns the summary: `levels`
(`{level: n}`), `claims` (`{identified, ion, tentative: n}` over the M0 rows,
zeros kept), `n_levelled`, `n_pairs`, `n_corroborate` and `axes`. The run's
log line ends `; claims identified N | ion N | tentative N`, and `peaky
assign` prints the same tally as a `claims:` line. `cross` is
the cross set of §6.4: the neutral formulas the `--corroborate` sources hold
at 4b or better by their own evidence (repeatable CLI option on `peaky
assign` and `peaky batch`; a run dir, an out-dir holding one run, or a ledger
CSV; resolved as `level_ledger.resolve_source` does); empty for a bare
single-sample run.

### 6.2 Batch — recomputed on the pooled files, stamped on the merged ledger

The merged ledger carries none of the predicate columns, so the batch level
is **not** read off merged rows. `assign_batch.run` pools the per-file
ledgers of the assigned files (cover **and** residual stages) as one source
and calls `evidence.level_pooled(per_file, cross=cross)`: one evidence record
per `(neutral, adduct)` over all files, so `chan2` sees a second adduct in
*any* file, `iso` any file's satellite, `tied`/`lowconf` require *all* rows
(across files) and `below` and `lead` any. The result is joined onto the merged ledger
by `(neutral_formula, adduct)` — each merged row is one ion, so the join is
one-to-one — and the five columns are written there too (`evidence.stamp_merged`:
`claim` is re-read off the joined level, so every merged row carries one and a
row with no per-file reading reads `tentative`). The per-file ledgers keep
their own per-file levels (computed at 6.1).

Two batch runs named together (`peaky batch … --corroborate <other run>`)
corroborate this run by the neutrals the other holds at 4b or better on its
own evidence (§6.4). The trace-first path is a one-file batch and follows the
same rule.

Rule U's fact is measured here and nowhere else: `batch/neutral_pairs.measure`
reads the stamped batch time series (the table `per_file/_batch_ts.parquet`
holds) and the pooled per-file ledgers, `tables/neutral_pairs.csv` records every
clause per neutral (written on every run, empty without a declared pair or a
time series), and `level_pooled(..., upair=...)` reads the set.

Rule K's facts likewise (C18): `batch/label_twins.measure` reads the same
series, the pooled per-file ledgers and, per file, whether each tied
`[X+NO3]-` row's tie is with same-ion aliases alone (`label_twins.alias_only_ties`,
taken while the parent collects the file); `tables/label_twins.csv` holds one
row per 14N line of every neutral committed on either nitrate adduct (the
cluster-k test, the line's verdict, the untie, `alien`, the veto) and one per
committed `[Y+^NO3]-` reading (E, the twin's sightings, the verdict, the veto)
— written on every run, empty out of scope or without a time series — and
`level_pooled(..., label=label_twins.facts(table))` reads the untie, veto and
alien sets; given them, it also counts the two nitrate clusters of one
neutral as one channel.
The untie changes only the pooled level: the per-file levels, the tiers, the
confidence labels and the vote keep the arbiter's tie, so a merged row can read
3b where its per-file rows read 5b.

The isotope checks likewise (C11+a): `batch/iso_checks.measure` reads the same
series and the pooled per-file ledgers, with the batch's width model (the
instrument class and REQ's observable lines), its mass scale (σ and the stamp
window) and its `height_cutoff_x_edge` (each spectrum's floor, the noise edge
times it — the per-file `height_gate_cps` where a file is also a ledger);
`tables/iso_checks.csv` holds one row per tested pooled pair and check (the
numbers, the verdict, `veto` and its `note`) — written on every run, empty
without a time series — and `level_pooled(..., iso=iso_checks.facts(table))`
reads the veto set: hard 5b with the joined note, and the pair leaves its
neutral's pools in both directions. As rule K's, the facts change only the
pooled level; the per-file levels, the tiers and the vote are untouched.
Rule H (C11+b) is the table's fourth check: its `H` rows carry a `lock` (never
a veto), judged against the batch's reagent profile (the reagent supply) and
its `context` (`budget_ok`, the element budget each file's plausibility stage
demoted against), and `iso_checks.facts(table)` returns the locks beside the
vetoes; `level_pooled` lifts the leads they answer (§3 `lead_lift`). A merged
row can read 4b where its per-file rows carry `tentative_lead` and read 5b.

`batch_summary["evidence_levels"]` = `{"pooled": {level: n}` (one count per
pair), `"merged": {level: n}` (per merged row), `"per_stage": {cover: {…},
residual: {…}}`, `"n_pairs"`, `"n_unstamped"` (merged rows whose reading no
per-file ledger holds), `"n_corroborate"`, `"cross_source": [...]`,
`"neutral_pairs": {pair, neutrals, committed_both, cho, present, covary, clean,
upair}` (the funnel; `{pair: [], neutrals: 0, upair: 0}` without a pair),
`"label_twins": {in_scope, f, lines, codetected, references, cluster_k,
committed_lines, lines_tracks, lines_consistent, lines_excess, lines_absent,
lines_untestable, alien, untie, readings, testable, passes, refuted,
lines_refuted}` (rule K's funnel; `{in_scope: false,
lines: 0, untie: 0, readings: 0, refuted: 0}` out of scope),
`"iso_checks": {instrument, tested, vetoed_pairs, locked_pairs, C: {tested,
vetoed, agree, ambiguous, contradict, untestable, scan_edge, exempt_14N}, REQ:
{tested, vetoed, present, absent, untestable}, HIGH: {tested, vetoed,
consistent, guarded, too_high}, H: {tested, locked, lock, no_lock,
si_rich, heavy, reagent, untestable}}` (the checks' funnel; `tested`
counts the H rows too; `{instrument, tested: 0, vetoed_pairs: 0,
locked_pairs: 0}` without a time series), `"lead_lifted"` (the pooled pairs
whose lead a halogen lock lifted)}; the
pair table with every fact of §3 is written to `tables/evidence_levels.csv`,
with the pair's `claim` beside its level.

`batch_summary["claims"]`, right after `evidence_levels`, tallies the claims
(each a `{identified, ion, tentative: n}` dict, zeros kept): `"merged"` (per
merged row), `"pooled"` (per pair of the pair table), `"per_stage"` (merged
rows by `cover` / `residual`), `"by_tier"` (merged rows per tier, Assigned
first; the ion-only rows under their own `"ion-only"` key, present only when
there are some) and `"n_unlevelled"` (merged rows with no level, whose claim
reads tentative). The batch logs `[assign_batch] claims (merged): identified N
| ion N | tentative N`, `peaky batch` prints the merged tally again after
`[batch] done`, and `run_manifest.json` records it as `counts.merged_claims`
beside `merged_tiers`.

### 6.3 The post-hoc script

`scripts/level_ledger.py` stays as the reference and the tool for runs made
before the column existed. `scripts/scorecard.py` already prefers an in-core
`evidence_level` column when present. Rule U's fact is measured by the batch,
not by the script: `--upair` (no value) reads each batch source's own
`tables/neutral_pairs.csv`, `--upair <csv>` applies one table to every levelled
source, and without the flag row 9′ never fires. Rule K's facts the same way:
`--label-twins` reads each batch source's `tables/label_twins.csv`,
`--label-twins <csv>` applies one table to every levelled source that holds
every pair the table names (another batch's table does not fire there, and
says so), and without the flag (or with an empty table) no part of rule K fires, the
one-channel fold included. The isotope checks the same way: `--iso-checks`
reads each batch source's `tables/iso_checks.csv`, and `--iso-checks <csv>`
applies one table to every levelled source that holds **every pair it
vetoes or locks** — the checks were measured on one batch's series and name its pairs,
so a source missing one of them is another batch; the table does not fire
there, and stderr says so. The script reads rule H's locks from the same table
and lifts in lockstep (`lift_leads`: the setters from `lead_by`, the unflagged
rows' second channel, branch and anchor). A run named as a `--corroborate` source is
re-levelled without its rule K facts and its isotope checks, as without its
rule U pair (§6.4): the cross set is the source's own per-file evidence (a
run's per-file evidence carries no lift; a source that resolves to a merged
ledger is read at its stored level, a lift included, as for rules K and U);
`scorecard.own_levels_for` reads all
three tables as the run's own evidence, rule H's locks included.

### 6.4 The cross set — what a source's sighting is worth (C8)

A source corroborates a neutral only when it **pins it on its own**: at level
**4b or better** when the source is levelled with **no cross set at all**
(`evidence.source_neutrals`; `CORROBORATE_MAX_LEVEL = "4b"`). Below that the
source's grid merely enumerated the same formula at a peak — 4c unopposed but
unconfirmed, 4d the ion only, 5a exact mass alone, 5b arguing with itself —
and two grids agreeing is not a second sighting of the neutral.

- A source with per-file ledgers (a run dir, an out-dir holding one run, a
  per-file or single-sample ledger CSV) is **re-levelled**: its ledgers pooled
  as ONE source, `level_pooled(…, cross=None)` — the batch's own merged-row
  level, §6.2 — and a neutral counts when any of its pairs reaches 4b.
  Stored levels are never read here, so a source that was itself run with
  `--corroborate` cannot hand a run back the agreement it got from it, and
  two sources named together (`level_ledger.py A B`) are each corroborated by
  what the other pins on its own — never by their mutual agreement.
- A merged ledger (no `role` column) carries none of the predicate columns:
  its stored `evidence_level` is read, **without its own `corroborated`
  axis** — a row counts at 4b or better when `corroborated` is not among its
  axes, or when it still holds `iso` / `chan2` / `anchor` once that axis is
  taken away (and is not an `iso`-only `reagent_only_iso` row, which would
  fall to 4d). A level the axis alone produced — a 4b of one corroboration, a
  curated row with no axis of its own — does not count. A source carrying
  neither the predicate columns nor `evidence_level` raises: there is nothing
  to judge the sighting by.
- Ion-only pairs never count, in either direction (§3).

Measured on the same-air pair before the rule (three regression runs, the
per-file levels, the merge vote and the merged levels replayed with the new
cross set; each replay's baseline reproduced its run row for row): the
labelled-nitrate Orbitrap offered the TOF 1521 neutrals, 379 of them pinned on
its own; the TOF offered the Orbitrap 3728, 435 pinned (3437 of its pooled
pairs are 5b — a 10k-resolution TOF can seldom pin a formula); of the TOF's 49
vote winners lifted by the axis alone, 28 rested on an Orbitrap 5a / 5b. The
merge vote (C2b) then changes 36 winners on the TOF (every test-set row and
every roster compound keeps its reading), 12 on the Orbitrap (all between
implausible grid formulas) and none on the uronium channel, where the 4a rungs
built on the TOF's 5b agreement fall to 4b. The curated perfluoroheptanoic
acid, mass-saturated on both instruments with no axis on either, falls from
3a to 5b on both: its one "axis" was the mutual agreement.

## 7. The column contract

| column | type | on | value |
|---|---|---|---|
| `evidence_level` | str | M0 rows; `NA` elsewhere | one of `2b 3a 3b 4a 4b 4c 4d 5a 5b` (never `1` / `2a` today) |
| `evidence_axes` | str | M0 rows | `|`-joined, in this order, of the axes that hold: `iso`, `chan2`, `anchor`, `corroborated`, then the modifiers `multiline`, `carbon`, `branch`, `reagent_only_iso`, `ion_only`, `upair` (batch only, rule U), `label_untie`, `label_veto` (batch only, rule K), `iso_veto` (batch only, C11+), `lead_lift` (batch only, rule H), `known:<family>`, `files:<n>` (batch only); `''` when none |
| `level_reason` | str | M0 rows | one sentence naming the predicate that fired, in the words of §4, with the numbers (`"5b: near-tie broken by the arbiter"`, `"4c: 1 plausible ion in the window, resolved, no axis"`, `"4a: iso + chan2, corroborated by the other source"`) |
| `n_plausible_structures` | Int64 | M0 rows whose formula is in the isomer space; `NA` otherwise | from `isomer_space.csv` |
| `claim` | str | every M0 row of a per-file ledger and every merged row; `NA` on every other per-file row | `identified` / `ion` / `tentative` = `claim_class(evidence_level)` (§1.1); a committed row with no level reads `tentative` |

The two assignability flags the level reads (§2) are per-file ledger columns
too: `below_assignability` and `tentative_lead`, bool, on every row of a tiered
ledger (created False by the tier stage, set on M0 rows, reset by a commit, a
clear or a displacement). The merged ledger carries neither: its level is the
pooled one. `tables/evidence_levels.csv` carries both pooled facts, `below` and
`lead`; the workbook's **Below assignability** sheet lists the M0 rows either flag
marks, with a `tentative_lead` column saying which and `lead_by` which setter
made it a lead (C11+b); `io/publish.py` carries all three in
`engine_provenance`. A ledger written before C19(c) has no `tentative_lead`
column and every reader takes it as False; one written before C11+b has no
`lead_by`, and rule H takes its leads for any setter.

Written to: the per-file `<prefix>_ledger.csv`, `merged_ledger.csv`,
`tables/evidence_levels.csv` (batch: one row per pair with every fact of §3, the
`ion_only` flag, `multiline_elements`, `upair`, `label_untie`, `label_veto`,
`label_note`, `iso_veto`, `iso_note`, `lead`, `lead_lift`, `lock_note` and the pair's `claim` included; an alien 14N line and a pair an
isotope check refutes read `chan2` / `branch` False),
`tables/neutral_pairs.csv` (batch: rule U's clauses per neutral),
`tables/label_twins.csv` (batch: rule K's tests per 14N line and per labelled
reading), `tables/iso_checks.csv` (batch: the isotope checks, one row per tested
pooled pair and check, rule H's `H` rows with `lock`, `element`, `n_halogen`,
`ratio_lo` / `ratio_hi`, `heavy_cl` / `heavy_br`, `budget_ok` / `budget_why`), the
Excel workbook (a column on the ledger sheets, `claim` directly before
`evidence_level`; a **"By evidence level"** sheet: one row per level with
count, share, tier split, the axes histogram and the twenty brightest rows;
and the **"By claim"** sheet the workbook opens on: one row per claim with
count, share, signal, tier split and level histogram, the rows where tier and
claim part, and the twenty brightest rows per claim), an **Evidence levels**
page in the PDF report after the assignment-quality page and a line on its
cover, a **Claims** page right after the cover and a claims line leading the
cover's summary, the published engine provenance (`io/publish.py` carries the
five columns in `engine_provenance`, dropping `NA` values; a batch publish
carries `batch_summary["claims"]` in the run's config), `docs/OUTPUTS.md`,
README, SKILL.md. Every consumer must render a ledger **without** the columns
unchanged (older runs, `peaky report` on them); a ledger that has
`evidence_level` but no `claim` (a run made before the claim existed) gets the
claim read off its level, on the M0 / merged rows only.

## 8. Golden fixtures and what the tests pin

`tests/fixtures/levels/` (see its README): 52 gzipped per-file ledgers from
four real runs, trimmed to the rows and columns of §2, the uronium run's
neutral-pair table, and
`expected_levels.csv` with the level and the axes, flags and degeneracy facts
behind it (not `multiline`, `multiline_elements` or `upair`, which the tests pin
separately) per
`(source, neutral, adduct)`, written by `level_ledger.py` at the commit that
added them. `tests/test_evidence.py`:

- one test per level (2b, 3a, 3b, 4a, 4b, 4c, 4d, 5a, 5b) on a synthetic
  ledger where exactly that level fires, **and a mutant** that flips one
  input and must change the level;
- the four golden count vectors, exact (each source corroborated by what
  the other pins on its own, §6.4; the uronium set alone, with its
  neutral-pair table):
  `tv` 1373 → 21/15/107/16/143/15/10/37/1009,
  `tof` 3364 → 6/15/182/15/258/82/95/138/2573,
  `orbi` 1707 → 0/11/217/9/203/139/0/35/1093,
  `ur` 1161 → 4/4/0/331/376/291/0/82/73 (4/4/0/25/682/291/0/82/73 without
  the pair table)
  (before C8, with any-level membership: 21/15/107/38/162/9/9/33/979,
  6/16/182/38/260/79/91/135/2557 and 0/12/217/44/215/119/0/30/1070 —
  183 of the 6,444 pairs moved, every one to a lower level; before C17 the
  TOF vector was 6/15/182/22/247/82/99/138/2573: seven bromide-adduct 4a
  pairs had counted the reagent's 81Br as a second line; before C11+a it was
  6/15/182/15/254/82/99/138/2573: four brominated neutrals' `[M-H]-` pairs
  read 4d on their own 81Br line, and read 4b once `reagent_only_iso` needs
  the ion to carry more of the reagent halogen than the neutral, or none of
  it -- the two Br-free `[M-H]-` pairs with a 1:1 +2 Da line keep 4d);
- rule U: each clause of the fact vetoes it, the profile scope (only the
  uronium profile declares a pair), row 9′'s formula support and order, the
  fact never an axis nor in `cross` nor per file, and the **leak mutant** —
  the fact computed as "two channels and N-free" on every channel moves the
  TOF and nitrate vectors (`tests/test_neutral_pairs.py`); rule K: each clause
  of the cluster-k test, each 14N line verdict and its boundaries, the twin
  test, the alias-only guard, the detection curve's gate, the untie's, the
  alien line's (both directions), the one-channel fold's and the veto's
  levels, the profile scope,
  the facts never axes nor in `cross` nor per file, the reference script and
  the scorecard reading the table, and the **leak mutants** — untie every tied
  `[M+NO3]-` pair, veto every `[M+^NO3]-` pair, the fold alone, every 14N line
  alien: the nitrate and Orbitrap vectors move (`tests/test_label_twins.py`; no golden source carries a batch series,
  so the vectors stand); the isotope checks (C11+a): each clause of rule C,
  REQ and HIGH at its edge (the tolerance, the area-and-height rule, the
  eight spectra, the level-free bias, the M+1-slot guard, the 14N
  exemption, the scan edge, REQ's floor, window, stamped isotopologue and TOF
  double window, HIGH's co-variation, presence, floor, 3 × on area and height,
  slot, 13C-slot, alias and cap guards), the instrument gate, the veto's level
  and note, the pool exclusion both ways, the facts never axes nor in `cross`
  nor per file, the reference script and the scorecard reading the table, the
  batch end to end, and the **leak mutant** — every `[M-H]-` of the Orbitrap
  set vetoed, or one branch acid alone, moves its vector
  (`tests/test_iso_checks.py`), and the reagent-halogen flag's ion test
  (`tests/test_evidence.py`); C17's element rule
  case by case (`tests/test_multiline_elements.py`), both against the
  reference script;
- row-for-row equality with `expected_levels.csv`;
- non-M0 rows carry `NA`; a reagent-ion row never carries a level;
- missing columns and all-null columns do not raise;
- the pooled batch computation equals the script's pooling on the same files;
- every row of the isomer space carries a rationale; every compound-scope
  2b formula in the fixtures is in the isomer space;
- `evidence_axes` order and `level_reason` are stable strings.

All of it passes since B2; `tests/test_evidence_outputs.py` pins the wiring
(stage order, the batch recompute, every output, `--corroborate`).

The fixtures predate the `tentative_lead` split (C19(c)): they carry no such
column, their leads sit in `below_assignability`, and the four vectors are
unchanged by it. Replaying three regression batches' per-file ledgers with
each lead row's flag moved from `below_assignability` to `tentative_lead` (the
setter read off its note) gives every pooled pair the same level and the same
`level_reason`. `tests/test_tentative_lead.py` pins the split:

- each lead setter writes the lead and not below, and each hard setter writes
  below and not the lead;
- a row both mark keeps below, and a corroborated radical anion clears both;
- the pooled `lead` is any row. A lead-only pair is 5b with the reason
  `5b: below assignability`, alone or beside a tie or a Low score;
- the flag in either column gives the same levels, reasons, pooled facts and
  tiers. The reference script agrees row for row, and a missing column levels
  as an all-False one;
- the ion-only stage takes no lead parent;
- the Below assignability sheet lists leads, and publish carries the column;
- a commit, a clear and a displacement reset both flags and never create one.

Rule H (C11+b, decision D8) needed a lead the lock could lift, so the Orbitrap
and uronium sets carry `tentative_lead` / `lead_by` and, on the four pairs the
live runs lock and hold as leads (HBr `[M+^NO3]-`, C6H9ClO3 and C6H10Cl2O4
`[M-H]-`; C7H11ClO2 `[M+(CH4N2O)H]+`), the flag moved from
`below_assignability` to `tentative_lead`; `orbi_iso_checks.csv` /
`ur_iso_checks.csv` hold the live runs' rule H lock rows for the pairs each set
holds. The golden `orbi` and `ur` vectors are levelled with them:
0/11/217/9/206/139/0/35/1090 and 4/4/0/331/377/291/0/82/72 (without the lock
table 0/11/217/9/203/139/0/35/1093 and 4/4/0/331/376/291/0/82/73, the vectors
before; tv, tof and rule K's vector do not move). `tests/test_halogen_lock_*.py`
pin the rest: `lead_by` per setter (`_leadby`), every clause of the check at
its edge — the window ends, 1 ppm, presence 12/20, r 0.8, npar 20, the heavy
check at both spacings, Si ≥ 3, the mixed ion, the reagent supply, the
silicon test (the three scenarios of the 2026-09-28 decision: a ³⁷Cl line 0.15 mDa low locks at m/z 193 and, with no ²⁹Si line, at m/z 351; the D7 siloxane read as C22H23N6O7Cl is refused — resolved, blended and unparted — and, bright, refuted by rule C; the 2026-09-29 decision: the D7 siloxane dim, read as each of its six Cl formulas within 1 ppm, refused as a full-area centroid, as an apex-reporting picker's one peak at ²⁹Si and unparted — the four rule C cannot read with nothing else refuting them —, the D7 urea cluster's Si2 reading refused on its excess alone, a Si reading's excess at its edge (Si2 and Si1), a Si-free reading refused on the position alone and not refused on an excess at the ¹³C position, bright and dim with its own ¹³C line, an anti-correlated region, the TOF note — a +1 line read 2 mDa low refuses a real Cl lock; the m/z threshold at 206.3; the blended region's co-variation, median position (its peaks weighed by area, not read off the tallest one or weighed by height), bounds and presence; the note naming the position where a Si reading meets both criteria; the partner at the literal ³⁷Cl / ⁸¹Br spacings and the ²⁹Si line at the isotope table's (0.9995683, 0.2 µDa above AME2020's 0.9995681), the window's two edges below and above the threshold; each blended fixture is a full-area centroid unless it is named the apex), decoys, the veto state of every fixture that locks (`_lock_only`; the blended fixtures rule C refutes say so), `budget_ok` (`_check`), every clause of the lift
and the lockstep with the reference script (`_lift`), and the golden moves
(`_goldens`: exactly the four pairs; the flag back in below or no table moves
nothing).

`tests/test_claims.py` pins the claim (§1.1):

- `claim_class` on every level and on `None`, NaN, `pd.NA`, `''`, `'nan'`,
  an unknown level and a padded one; the two level sets are disjoint and
  cover the scale bar 5a / 5b; `claim` is the fifth of `COLUMNS`;
  `summarize_claims` keeps the zeros, in `CLAIMS` order, and skips `NA`;
- `apply_levels` stamps the claim on M0 rows only (an isotope child and a
  reagent ion carry `NA`), its summary's `claims` equals the tally of the
  column, and the claim follows the level when a cross set lifts it; the
  stage's log line ends with the tally;
- `stamp_merged` gives every merged row a claim — `tentative` for a row with
  no per-file reading, for an empty pair table and for none — and joins a pair
  table that predates the column;
- `batch_summary["claims"]`: the ion-only rows under their own key, a
  Candidate identified and an Assigned tentative standing side by side, zeros
  on an empty ledger; a batch run writes the column on the merged ledger and
  the pair table, the block right after `evidence_levels` and its log line,
  never on the `DONE` line the progress panel parses;
- the claim changes nothing upstream: with `claim_class` replaced by a
  constant the batch writes the same merged ledger, pair table and jitter
  table, claim aside; the vote's class sets equal the claim sets while a
  corroborated 5b votes class 2 and claims tentative; `claim` is neither a
  predicate column nor a vote column, and a per-file claim column that says
  the opposite of the levels moves no winner, tier or note;
- the entry points pass the tallies on: the publish batch config, both CLI
  lines, the manifest's `merged_claims`, and the MCP tools' `claims` and
  per-species `claim`.

`tests/test_claims_outputs.py` pins the workbook and the PDF (docs/OUTPUTS.md):
the By claim sheet first, `claim` directly before `evidence_level` on every
sheet that shows the level, `NA` off the committed rows, the claim read off the
level for a ledger without the column, a ledger without levels rendered
unchanged, the Claims page second in the report with its `unmatched` signal
bucket and both kinds of disagreement, and the cover's claims line.

## 9. Worked rows (real rows of `expected_levels.csv`)

| source | neutral | adduct | facts | level |
|---|---|---|---|---|
| tv_bromide | HO2 | [M+Br]- | known:atmospheric, one 81Br satellite | **2b** — identity outranks the reagent-only isotope reading |
| tv_nitrate | C2HF3O2 | [M-H]- | known:perfluoroacid, chan2, corroborated by the bromide channel | **3a** — class scope: TFA stays 3a |
| orbi | C9H8O3 | [M-H]- and [M+^NO3]- | branch (bare and clustered), iso, chan2, anchor | **3b** |
| tv_bromide | C5H8O3 | [M-H]- | iso, corroborated by the nitrate channel | **4a** |
| orbi | C7H8O6 | [M-H]- | anchor only, 12 of 12 files | **4b** |
| orbi | C4H8O6 | [M-H]- | no axis, one plausible ion in the window, resolvability not measured | **4c** |
| tof | H2O2 | [M+Br]- | one 81Br satellite, no 13C, no other axis, 23 files | **4d** — the ion HBr·O₂H⁻ is pinned, the neutral is not |
| orbi | C8H6O5 | [M-H]- | no axis, two plausible ions | **5a** |
| tof | HO2 | [M-H]- | known:atmospheric **but** Low confidence | **5b** — `hard` outranks the identity (a defect for the pass-0 lock) |
| orbi | HBr | [M+^NO3]- | iso, anchor, below assignability | **5b** |

Across the three field sets 4,675 of 6,444 pairs are 5b; the top two levels hold
68 (4,606 and 70 before §6.4). The uronium set (§8) adds 1,161 pairs, 73 of them
5b. That distribution is the point of the scale,
not a problem with it.

## 10. Open for B2/B3 (not decided here)

- Whether `n_files ≥ 2` becomes an axis (B3, on calibration).
- Whether a level above 350 m/z with one axis should be capped at 4b (the A3
  decoy split: above m/z 350 the mass-defect gap no longer protects the
  Orbitrap) — B3 measures, the predicate table changes only if the
  calibration order breaks.
- One reading difference between the code and the reference script, to be
  adjudicated in B3 on real runs: the ledger writes `isotopologues` with
  `json.dumps`, so a satellite without a per-line score (a ³⁷Cl envelope
  confirmed against the ledger) carries `null`; the script's `ast.literal_eval`
  cannot read that cell and counts the list as empty, the in-core `as_list`
  reads JSON first. No fixture row moves (the goldens and the row-for-row test
  hold), but a real run's chlorinated-paraffin rows may differ on the `iso`
  fact between the two. **B3 adjudicated (2026-09-22): no row of the three
  working-set channels moved on it, plain or corroborated — the commits that
  carry a `null` score are `known:` rows, levelled 3a before the `iso` fact is
  read.** The script keeps its reader; aligning it would move no golden.
- Whether `neutral_backed` should count `anchor` and `carbon_ev` — B3's
  calibration (the wrong-adduct decoy) shows both are facts about the **ion**:
  a ¹³C line and a series tie survive a wrong adduct set, so 64 wrong-adduct
  rows sit at 4b and 30 at 4c on the labelled-nitrate channel and 339 at 4c on
  the uronium channel. Option (a): drop them from `neutral_backed` and send rows
  whose only axes are ion-level to a generalised 4d ("ion pinned, neutral not";
  the goldens move, `level_ledger.py` and `evidence.py` change together, R1–R3
  re-run and re-validated row for row). Option (b): keep, and document that
  4b / 4c rate the ion formula. The user's decision; not changed here.
- A detectability term for heteroatom satellites (a 34S/81Br line that would
  sit below the noise edge cannot count as *missing*) — C3's topic; the level
  today only counts satellites that were seen.
