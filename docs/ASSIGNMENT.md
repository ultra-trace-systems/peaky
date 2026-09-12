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
