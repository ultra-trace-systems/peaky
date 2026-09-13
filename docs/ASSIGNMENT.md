# Peaky — Assignment explained

For the design/internals see [ARCHITECTURE.md](ARCHITECTURE.md); this page explains
*what assignment does and what the results mean*, for a scientist reading a run.

## What "assignment" means

**Assignment turns each observed peak into a chemical formula** — a confident
mapping from a measured m/z to a neutral molecular formula and the ionization
adduct it was detected through (e.g. peak at 339.001 → `C10H16O4` detected as
`[M+Br]⁻`). Every peak in the spectrum ends up in exactly one **role**:
`M0` (the monoisotopic owner of a formula), `iso_child` (an isotopologue of an
M0), `reagent` (a labeled reagent-ion cluster), `artifact` (instrumental
ringing/sidelobe), or `unexplained`.

## The one-ledger model

All state lives in **a single ledger DataFrame — one row per physical peak.**
Every stage is a `ledger → ledger` function that only *fills or annotates*
columns; **nothing drops rows.** A commit API enforces the invariant "every peak
is in exactly one role and M0 ownership is unique," so no pass can corrupt the
table. Because state is one auditable table, you can see exactly which pass
claimed each peak and why.

## Mascope is the only scorer

Peaky **never invents a mass or isotope score.** It owns *candidate generation*
(which formulas are worth asking about), *chemistry plausibility*, *series
logic*, and *arbitration* (which answer wins a peak). For scoring it uses
Mascope's own maths — by default **in-process** via `mascope_tools` (IsoSpec
envelope + `score_pattern`), with the network `match_compounds` as an opt-in
fallback (`PEAKY_LOCAL_SCORING=0`). Either way it gets, per candidate ion, a
per-isotopologue verdict (each isotopologue's `match_score` + the attributed
peak), which Peaky reads. See `docs/MASCOPE_TOOLS_INTEGRATION.md`.
**No LLM is anywhere in this loop** — which is exactly why a run is reproducible
and auditable.

## Sample selection (batch)

`match_compounds` scores against one real server sample's peaks, so a whole-batch
assignment assigns a **subset of samples** and merges by m/z. The subset is a
greedy **presence set-cover** over the batch's m/z bins (bins present in ≥ 2
samples, no height floor; each pick covers the most not-yet-covered bins; stop
when the next pick adds < 0.5 %, after at least 6 picks; `--k-max` 30 is a
flagged budget). Because the merge keeps whatever any assigned sample contained,
this selection is what decides recall — see [`SAMPLING.md`](SAMPLING.md) for the
measurements. Same merge, same outputs; `batch_summary.json['selection']` records
the achieved coverage and why the selection stopped.

## Admission: which peaks are eligible (persistence OR brightness)

Eight formula-hunting sites draw their candidate peaks through one gate
(`assignment/admission.py`): the **pass-1 grid**, **pass-2 series growth**,
**pass-3 contaminant families** and pass-3's two **cluster resolvers** (HX,
acid·I₂) — those five share `directors._target_peaks` — plus the **pass-6 ladder
gap-fill**, **residual stage B** and the **siloxane ladder** (its work set *and*
its seed test). A peak is **eligible** there if it is bright enough
(`height ≥ height_cutoff`, an edge multiple — see below) **or** persistent:
the batch's spectra hold a peak within tolerance of *its* m/z in at least a
threshold fraction of them (`sampling.BATCH_TOL_PPM` = 6 ppm, the one tolerance
the selection, this table and the merge share, so `batch` and `assign --ts-batch`
read one occurrence for one peak). The occurrence is **per peak, not per bin**
(`batch/traces.py`): one sweep over the m/z-sorted batch gives every peak the
count of distinct spectra with a peak within ±tol of it, and a ledger peak is
looked up by exactly that rule, so construction and lookup cannot disagree. (The
first version binned the batch by gap-clustering and read a peak off its bin's
centre: on a TOF 91 % of the "persistent" bins were wider than the tolerance —
one 134 ppm — so much of their persistence was the chaining, and a peak that had
been binned could fail to find its own bin.)
That threshold is **derived from the batch** (`occurrence_min = "auto"`): the
occurrence distribution is cleanly bimodal on every instrument measured
(transient below 0.1, persistent above 0.9), and Otsu's split of it — taken over
*traces*, each peak weighted by one over its spectrum count so an ion present in
500 spectra and a noise spike seen twice count once each — lands at 0.38–0.55
everywhere; a fixed 0.8 was tried first and discarded two-thirds of the recurring
weak ions the path exists to recover. The derived value is **clamped to
[0.25, 0.75]**, because Otsu returns a split even for a batch with no two modes
to separate: a unimodal-low distribution would derive ~0.06 (admit everything)
and a unimodal-high one ~0.96 (admit nothing). An all-equal distribution has no
split at all and switches the path off rather than defaulting to 0. A number
overrides the derivation, `0` disables the path, and fewer than 10 spectra switch
it off.

**The brightness floor is derived from the same table.** The height gate is a
multiple of each sample's noise edge (the 1st percentile of its picked heights),
and the right multiple belongs to the *peak picker*, not the reagent: a picker
that stops at the noise edge wants 1× (rare real ions sit at 1–3× it), one that
picks *into* the noise admits nearly everything it found at 1×. Two statistics
of the batch read that behaviour off the data, and the floor is raised only when
**both** say so. First, the **transient share of the peaks a floor of x edges
would admit** (the table labels every peak persistent or transient): three clean
Orbitrap modes measured 0.08–0.16 at 1×; a TOF 0.34 at 1× and 0.17 at 5×; an
Orbitrap mode with the reagent ion in range 0.50 at 1× (the reagent ion's noise
skirt and a scan-edge pile-up), 0.14 at 2×. Second, the **picker tail**: the share
of the batch's peaks sitting below 0.75× their own sample's 1st-percentile edge —
a hard-threshold picker leaves essentially nothing under its edge (at most 1 in
10 000 peaks on every Orbitrap mode, the reagent-in-range one included), a TOF
picker that keeps picking into the noise leaves a thin soft tail (31 in 10 000).
With nothing set (`height_cutoff_x_edge = "auto"`, the package policy), a batch
whose tail share exceeds 5 in 10 000 takes the smallest multiple on the grid 1 /
1.5 / 2 / 3 / 5 / 8 / 12 / 20 whose admitted peaks are at most **20 % transient**;
any other batch keeps 1× whatever its transient share. So every Orbitrap mode keeps exactly 1× — on the
reagent-in-range mode a 2× floor (which the transient share alone would have
chosen) removed 45 scan-edge pile-up rows and 12 reagent-skirt rows but also 14
multi-file Assigned ions at 1–2× the edge, a C6H12O `[M+H]+` seen in 9 of 27
files among them: there the "edge" is the picker's relative threshold under a
huge reagent ion, not noise — and the TOF lands at 5× (inside the recall plateau
of a live sweep, 17 of 18 findable target ions from 5× up, though below its
8–12× precision optimum; a site wanting the tighter list pins 8 or 12 in its
reagent profile), with the persistence path keeping the recurring weak ions the
raised floor would otherwise drop. Without a batch (a single sample) the floor is
1×. The chosen multiple, its source, both statistics and the whole transient-share
table are recorded (`batch_summary.json` `gate`); an explicit
`--height-cutoff-x-edge` or a profile value pins a number instead.
Noise does not recur at a fixed m/z across hundreds of spectra; ions do — on a
mixed-reagent TOF batch ~500 bins recur in > 80 % of 230 spectra at a median
3–4 cps, and the exact masses of 21 highly oxygenated molecules an Orbitrap saw
on the same air all sit in that recurrent population while mass-shifted decoys
hit nothing. The persistence path is purely additive (no bright bin is
transient) and **only admits peaks for consideration; it does not lower the bar
for confirming them** — at 2–3 cps the isotopologues are sub-count, so such
peaks normally land as Candidates, and the isotope rules are untouched.
**Persistence gates entry; only corroboration gates the tier**: an
occurrence-admitted peak is a real ion, but nothing constrains *which* formula
it got, so the tier engine caps it at Candidate (`tier_reason`
`persistent-weak`) unless an isotopologue, a second channel or a series anchor
corroborates the formula. A pass-0 known species is exempt: that branch is
tested first, so a curated identity stays Assigned — its evidence is the locked
list, not this peak's height. Each per-file ledger row records `occurrence` (float
in [0, 1], NaN without batch context or when no batch peak lies within tolerance) and
`admitted_by` (`height` / `occurrence` = persistence only / `''` = not eligible
at those eight sites); the merged ledger carries them for the winning row.
Batch runs compute the occurrence table from the batch time series; a
single-sample run without `--ts-batch` has no batch context and is
brightness-only. `--occurrence-min 0` disables the path.

## From the merged ledger to every spectrum (trace stamping)

Only the selected samples are assigned; the other spectra receive their formulas
by **stamping**: every raw peak within a window of a merged ion is labelled with
it (`per_file/_batch_ts.parquet`). The merged m/z is an *anchor* minted from the
few assigned samples, and on a TOF the assignment snaps that anchor to theory (a
formula is only committed where a sample's draw lands near it) while the ion's
trace across the batch genuinely sits several ppm away — measured on one
230-spectrum TOF batch, 22.6 % of the anchors were outside their own trace's
window, and the highly oxygenated monomer C10H16O9 kept 40 % of its spectra from
the anchor and 81 % from the trace centre. Two assigned samples also mint two
competing rows for one ion when their draws differ by more than the merge
tolerance. So, between the merge and the stamp (`timeseries.recentre_ledger`,
`collapse_trace_labels`, `stamp_tolerance`; `docs/TIMESERIES.md` §9):

1. **Re-centre** every merged row on its own trace (a mean shift over the full
   batch peak list, never more than 10 ppm from the anchor; an almost-empty anchor
   that wants to jump more than 6 ppm must be seen in ≥ 2 files or be Assigned).
   The merge anchor stays in `mz_anchor`; the stamp uses `mz_trace`.
2. **Collapse** rows whose trace centres fall within the tolerance of each other:
   one ion, one label — the winner is the row seen in the most assigned files,
   then the higher tier, then the higher ion score (the merge's own ordering; a
   per-file score is blind to reproducibility and, first, handed a one-file
   C14H17NO4S the trace of an Orbitrap-confirmed six-file C10H16O6 — while
   proximity to the trace centre, tried as a tie-break ahead of the score, handed
   C10H16O9's trace to a one-file competitor whose theory-snapped anchor happened
   to land near it). Collapsed rows stay in the ledger flagged
   `trace_role = collapsed`.
3. **Size the stamping window** to the batch's own per-ion scatter (the third
   quartile of the per-trace scatter, ×2.5, between 1× and 2× the merge
   tolerance): an Orbitrap keeps 6 ppm, a TOF gets ~10 ppm.

The identity is still decided by the assignment on the bright samples; the trace
only carries it to the faint ones. Measured on the same TOF batch this lifted the
stamped coverage of the known monomers from a 0.50 mean to 0.57 (C10H16O9 from
0.14 to 0.86) with 38 % of all ions gaining more than 5 points and 3.7 % losing,
and changed two Orbitrap batches by nothing (mean coverage 0.754 → 0.754,
0.542 → 0.542) — the no-op that validates the diagnosis.

**Not yet gated (brightness-only, `height ≥ height_cutoff`)**: residual stage A
(the ~2-Da isotope-doublet scan), the reflist rescue, pass-3's series
*detection* statistics (`series_detect` counts a series' members rather than
proposing formulas for a peak — pass-3's own target peaks *are* gated) and the
isotope-satellite tests in `passes/postprocess`. `admitted_by` says nothing
about those; converting them is a documented follow-up, kept out of this change
so the validated behaviour is unchanged. The gated set is the callers of
`directors._target_peaks` plus the three direct `admissible()` call sites —
re-derive it that way, not by grepping for `admissible(`.

## The pass sequence

Assignment is multi-pass; each pass only adds commitments the previous ones
justify:

| Pass | What it does |
|---|---|
| **Pre** | Detect reagent adducts; prescan the isotope fingerprint; **label reagent-ion clusters** (e.g. Brₙ, BrO/BrO₂/BrO₃ with both ⁷⁹/⁸¹Br) so they are never candidates. |
| **0** | **Known species (committed + locked, runs first):** specific families that the generic grid/gates would otherwise miss — atmospheric acids/radicals, nitroaromatics, **PFCAs**, **chlorinated paraffins** (only with a confirmed ³⁷Cl envelope), silanediol contaminants, and (positive mode) **organophosphates** AND **organothiophosphate/-dithioate insecticides** (malathion family, chlorpyrifos, diazinon, parathion…; P off the grid, S above `max_S`). A P-bearing known species commits with **≥2 ion channels OR** a confirmed diagnostic ³⁴S/³⁷Cl/⁸¹Br isotope envelope (**not ¹³C**) substituting for the 2nd channel. |
| **1** | Lock the high-confidence **CHO/CHON backbone**: grid-enumerate candidates, score with `match_compounds`, arbitrate (complexity-penalised, isotopologue-gated), commit the M0 owners + attach isotopologue children. Pass-1 self-calibration refines the mass offset. |
| **2** | **Iterative GKA series** expansion from the locked anchors (CH₂/O/H₂O/CO/CO₂/C₂H₂O + siloxane/CF₂), chaining each confirmed member as a new anchor. |
| **3** | **Automatic series detection** (the "rotating plot") opens contaminant families on decoy-controlled evidence — organosulfate/nitrate/siloxane/amine, isotope-gated bromo/chloro-organics. |
| **4** | **Residual explainer**: resolves ~1.998-Da isotope doublets, deep 2-step series, ppm-disciplined acceptance. |
| **5** | **Known-neutral completion**: fills cross-channel partners + series gaps of passes 1–4 (no new formula space). |
| **6** | **Anchored ladder gap-fill**: walks +O/+CH₂/+CO₂/−H₂O diagonals out from Assigned anchors, satellite-guarded (Candidate tier). |
| **cleanup** | Isotope-confirmed recovery of molecules the score gate dropped, bromide-cluster relabelling, ringing-artifact flagging, and (positive mode) re-reading uncorroborated `[M+NH4]+` as the `[M+H]+` amine. A further (positive mode) **reagent-N re-read** re-reads a *pure hydrocarbon* that was assigned via an N-carrying reagent cluster (`[M+NH4]+` / urea `[M+(CH4N2O)H]+`) — chemically implausible, since a hydrocarbon has no site to bind it — as the `[M+H]+` of the corresponding N-heterocycle (skipped when the hydrocarbon has its own genuine `[M+H]+` row). Post-tier **plausibility demotes** (carbon-cluster, heteroatom-free-hydrocarbon-via-an-anion-channel, speculative-residual) and a Br-run reagent-halocarbon relabel drop over-eager commits one tier — never deleting a peak. |
| **15N-label rescue** | (labelled profiles only, e.g. `NO3_15N`) Fills/repairs peaks that are covalent heavy-isotope products of the labelled reagent — a ¹⁵N-organonitrate sits *j·*0.997 Da off any grid formula, so it is left unexplained or absorbed by a partially-fluorinated fit. Re-enumerates the CHON grid at the shifted mass, substitutes ¹⁵N, and commits only on a four-gate discipline (on-calibration mass, organonitrate plausibility, isotope corroboration, non-degenerate). See ASSIGNMENT_DETAIL §3.8. |
| **¹⁴NO₃-cluster re-read** | (¹⁵N-nitrate profile only) The exact-isobar arbitration for a NOx run: a covalent organonitrate `[Y−H]⁻` whose cluster parent `X = Y−HNO₃` is independently detected (its own `[X−H]⁻` and/or ¹⁵N cluster `[X+¹⁵NO₃]⁻`) is re-read as the chamber `[X+¹⁴NO₃]⁻` cluster (same ion; tier preserved). ¹⁴NO₃ is kept off the scoring grid so the isobar is never arbitrated by score. See ASSIGNMENT_DETAIL §3.9. |
| **rearbitrate** | (calibrated runs only, before degeneracy + tiers) **Off-calibration re-arbitration**: an over-ranked off-cal degenerate M0 winner is displaced at *selection* — not just tier-demoted later — so an "aromatic-monster" answer can't hold the M0 slot it would only ever be tiered out of. |
| **reflist** | Context-gated **reference peaklists** (literature HOM + common MS contaminants) corroborate near-ties (selection prior) and **rescue** mass-matched unexplained peaks — each rescue re-scored by the server before commit, provenance-tagged, never overriding an isotope-scored Assigned. |
| **tiers** | The final verdict (below). Degeneracy density is measured first so a mass-degenerate commit can't claim Assigned. |

(Passes 2/3 run via the `series_gka` / `series_detect` engines under the `passes`
director. Interleaved sweeps — isotope-envelope completion, **composite detection
(halide-CIMS only — a no-op in positive urea mode)**, the dedicated siloxane
ladder — claim satellites and Si oligomers the CHON-centric heuristics otherwise
mis-read. CLI toggles: `--no-pass2/3/4`; `--no-pass5` disables **both** Pass 5 and
the Pass-6 ladder gap-fill.)

## Reference peaklists (literature corroboration)

Peaky can consult **curated, provenance-tagged reference peaklists** — a catalog of
known neutrals per chemical system, each entry carrying its source, data version, and
literature references. They are used three ways, all **soft** and none ever overriding
an isotope-scored Assigned:

- **Selection prior** — a candidate that sits on an active list wins a near-tie in
  arbitration (a small score nudge, not a free pass).
- **Rescue-verify** — an *unexplained* peak whose mass matches a list entry is handed
  back to Mascope's `match_compounds`; it is committed only if the server confirms it,
  otherwise kept as a tentative low-quality Candidate.
- **Report corroboration** — a dedicated report section + `tables/reflist_matches_*.csv`
  records which assignments a list corroborated and which peaks it rescued.

Lists are **context-gated** by the run's metadata (the common-contaminant list is
always active). Seeded with α-pinene OH-oxidation HOM (Kang et al., 830 neutrals) and
the Keller (2008) MS-contaminant list (59 neutrals); add your own as a self-describing
JSON file under `peaky/data/peaklists/`. A literature match never *invents* confidence —
Mascope still scores every commit, so the honesty principle holds.

## Plausibility hardening

A set of post-tiering gates demote (never delete) commitments that fit by mass but whose
isotope evidence or ionization chemistry does not actually support them: **carbon-rich**
(an F-free formula with implausibly low H/C), **implausible-ionization** (a heteroatom-free
hydrocarbon detected through an anion channel that needs an acidic / H-bond site),
**speculative-residual** (a residual commit resting on off-calibration charge, uncorroborated
multi-nitrogen, a zero-anchor series, or a single minor channel), and a **reagent-halocarbon
relabel** (Br runs) that re-reads bromomethane fragments mis-assigned as a bare element +
Br-cluster on their invariant ion composition. Each is conservative — it lowers a tier or
relabels a role, it never fabricates an assignment.

A second, **hardened** layer (`peaky/plausibility.py`) shares one oracle between the scrutiny
flags and the demotes, so a flagged formula and a demoted formula can never disagree:

- **Oxygen-lattice monster** — `O/C > 1.3` **and** the degeneracy audit flags the mass as
  saturated. It is deliberately *not* niso-gated: a ¹³C satellite confirms the carbon count,
  not the oxygen count, so a real ¹³C twin must not exempt a monster. Real HOMs top out at
  `O/C ≈ 1.14` and are spared by the ratio cut.
- **Carbon cluster** — `DBE/C ≥ 1.0` (equivalently `H ≤ N+2`), F-free, C ≥ 2, with a
  **half-integer-DBE (radical) exemption**. The `≥ 1.0` cutoff spares real aromatics
  (pyridine, coumarin, umbelliferone, furfural, phthalic anhydride all sit below 1.0).

Both are demote-only (Assigned→Candidate + `below_assignability`). Every touch is logged to
`tables/plausibility_audit_<tag>.csv` (one row per peak: before/after tier, the reason, and the
supporting evidence).

> **Deferred:** an automatic *in-source fragment* relabel (adduct-less protonated M0 → fragment
> of a heavier co-varying parent, with a companion time-incoherent-series dissolve) was
> prototyped but removed before release — on real merged data the triangulation over-fired on
> coincidental facile-loss mass matches. The two demotes above are unaffected.

## The structural chemistry gates

Valence facts, applied identically every run — the reason a bare "mass fit" never
wins:

- **Integer DBE + Senior's rule on the neutral.** Half-integer DBE is an ion-only
  artifact (deprotonation), so organic nitrates pass as neutrals.
- **Oxygen cap** `O ≤ 2·(C+N+S+P)+4` — valence, not a Van Krevelen prior. Kills
  `C3H5ClO17`-type mass-fits while real HOMs pass.
- **Isotopologue-gated heteroatoms.** A neutral S/Cl/Br must show its
  Mascope-confirmed ³⁴S/³⁷Cl/⁸¹Br, or its complexity skepticism stands.
- **Reagent-halogen policy (two-sided).** The reagent halogen's *ion* isotope
  can't prove the halogen sits in the *neutral* (covalent `X(Br)[M-H]⁻` is
  degenerate with `Y·HBr·Br⁻`). **Isotope-confirmable halogens (Cl/Br/S)** are
  opened on the grid and tiered on their envelope; **monoisotopic F/P** are off
  the grid except specific known families.

## The tiers

Committed assignments are split into report tiers by **mechanical rules on ledger
columns** (no judgment calls at report time):

- **Assigned** — the formula is unique in the calibrated mass window, *or* it's
  corroborated by independent evidence (confirmed isotopologues / attached
  satellites, the same neutral in a second ionization channel, or series-anchor
  support), and nothing about the chemistry contradicts the validated sample
  profile.
- **Candidate** — a plausible formula, honestly ambiguous: a low/suspect base
  confidence, an effective-score near-tie, undiscriminated close alternatives in
  the window, or a high cross-family mass-degeneracy density.
- **Reagent-N isobar demotion (positive mode).** A CHO neutral seen through an
  N-carrying reagent adduct (`[M+NH4]+` or urea `[M+(CH4N2O)H]+`) forms the *exact
  same ion* as a protonated CHON neutral — same formula, so neither mass nor
  isotopes can separate them. Such a commit is demoted to **Candidate** unless an
  extra-spectral discriminator resolves it: the same neutral in an N-free channel
  (`[M+H]+`/`[M+Na]+`/`[M+K]+`), the jointly-unfakeable NH4+urea pair, or a series
  anchor. The tier_reason says so honestly rather than claiming the old false
  "unique in the calibrated window." Validated: Ur moved −124 Assigned→Candidate
  (all in the reagent-N family); Br is unaffected (halide, negative-only). See
  [ASSIGNMENT_DETAIL](ASSIGNMENT_DETAIL.md) for the mechanics.
- **Below assignability** — the **unexplained** residual, characterized
  peak-by-peak (isotope-partner / has-constraints / isolated). Presented as a
  *constrained mass*, not a confident formula.

Reported mass errors carry a **calibrated `ppm_error_cal`** column alongside the
raw `ppm_error`: an offset-only correction that subtracts the robust median mass
error of the corroborated pure-organic core, so QC reads and provenance sit on a
zero-centred window. It is display/provenance only — tiering was already
calibration-aware (the z-gate centres on that robust median), so no tier changes
from this. See [ASSIGNMENT_DETAIL](ASSIGNMENT_DETAIL.md).

> **The honesty principle:** `% signal explained` is a **coverage** metric, not a
> **quality** metric. A peak is only "Assigned" when the *evidence* — not just
> the mass — supports it.
