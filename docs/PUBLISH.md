# Publishing a ledger back into Mascope (`peaky publish`)

peaky reads peaks out of Mascope and writes its ledgers as local files. `publish`
closes that loop: it uploads a finished ledger into Mascope's **peak-assignment
run ledger**, so a peaky run lands in the same store the in-app engine writes to.

```
Mascope peaks --(SDK read)--> peaky computes ledger --(publish)--> Mascope run ledger
                                                                     -> run selector
                                                                     -> peak inspector
                                                                     -> batch Assignments overview
                                                                     -> verification loop
```

Both engines then share one store, one read API, one UI and one retention
policy, which is what makes them comparable on the same sample. The server-side
contract is Mascope's `docs/dev/sdk_peak_assignment.md` §8.2; this page is the
client half.

```bash
peaky publish output/<run>/<sample>_<stamp>_ledger.csv --dry-run   # translate + check, send nothing
peaky publish output/<run>/<sample>_<stamp>_ledger.csv             # publish
```

The sample is read from the ledger's own `sample_item_id`, and the
`*_manifest.json` beside it is picked up automatically as the run's config.

## What it does to your ledger

An import must satisfy Mascope's read model, and a peaky ledger is not
field-for-field identical to it. Four translations happen, and three of them are
places a naive mapping is silently wrong.

### 1. Two tiers, and neither is a copy of the other (the important one)

peaky tiers **mechanically**: window uniqueness, isotopologue corroboration, the
degeneracy audit, oxygen count, F/H coherence (`assignment/tiers.py`). Mascope
tiers by **threshold**: a row's *evidence* — `fit_score` weighted by the chemical
plausibility of its formula — against the run's declared `tier_bands`.

These answer different questions, and the published row carries **both**:

| column | who decided it | used in cross-sample roll-ups |
|---|---|---|
| `tier` | Mascope, from the evidence under the declared bands | yes |
| `engine_tier` | peaky, on its own terms | **never** |

`publish` **does not send `tier` at all.** It is a pure function of the fit, the
formula and the bands — every input already server-side — so the server derives
it. Sending one would mean reproducing Mascope's chemical-plausibility function
exactly; the two implementations then drift, and a disagreement at a band edge
refuses the whole import over a number peaky had no reason to hold.

What peaky *does* send is `engine_tier`: its own verdict, mapped onto Mascope's
vocabulary (`Assigned`/`Identified` → `assigned`, `Candidate` → `candidate`).
Only committed **M0** rows carry one — peaky tiers nothing else, and null is how
that is said. Absence is not agreement: Mascope's `tier_disagrees` filter
excludes untiered rows from both answers.

This is the point of the feature. On a real ledger, 195 of 1096 assigned rows
were rows where peaky demoted a peak that Mascope's banding would have called
`assigned` — disagreements that used to be flattened away and are now a filter
in the app.

The bands still matter, because they set what Mascope's `tier` means: they
default to the in-app engine's own (`assigned >= 0.75`, `candidate >= 0.45`), so
a published tier means the same thing as an in-app tier on the same sample.
Override with `--assigned-band` / `--candidate-band`; the run records whichever
pair it used, and the app shows them on the engine badge.

> **Version note, now cosmetic.** `--dry-run` previews the tiers Mascope will
> derive, and that preview needs `formula_plausibility`, which lives in
> `mascope_tools` but is not in the 2026.6.25 release peaky's dependency
> resolves to. Without it the preview assumes a plausibility of 1.0 and reads
> high for a formula the server weighs down. **Nothing is at risk** — the tier
> is derived server-side and the preview is not published. Upgrade
> `mascope-tools` to a release exporting `formula_plausibility` for an exact
> preview.

### 2. Which intensity, and why it is not your choice

Mascope expects the quantity its own engine supplies for that instrument:
**peak heights for Orbitrap files, peak areas for TOF files**. The value is not
cosmetic — it lands on the batch occurrence that scales this sample's consensus
vote, while the batch peak's declared *unit* is derived independently from the
file's instrument type. Publish the other quantity and the stored number
disagrees with the label above it and shifts the sample's weight against its
peers, silently, because nothing on either side can detect the swap.

`--intensity auto` (the default) reads the instrument off the sample record and
picks for you. It **refuses to guess** when the instrument cannot be classified;
pass `--intensity height|area` to decide explicitly. Naming it also makes the
translation fully offline, which is what `--dry-run` wants.

### 3. Vocabularies

| peaky | Mascope |
|---|---|
| role `unexplained` | role `unassigned` |
| roles `M0` / `iso_child` / `reagent` / `artifact` | unchanged |
| tier `Assigned` / `Candidate` (capitalized, mechanical) | re-derived, lower-case |

An `iso_child` publishes its owner's committed formula (Mascope models the
family that way; peaky's child row carries none of its own) and is scored by its
`iso_match_score`. Its `parent_peak_id` becomes `owner_sample_peak_id`, which
the server resolves to the minted owner id when the import finalizes.

### 3b. The ionization mechanism, and why it is resolved by default

peaky names adducts by notation (`[M+H]+`); Mascope keys them by a per-deployment
`ionization_mechanism_id`. The contract permits sending null, and that is a real
fallback &mdash; but it costs more than an empty column. The mechanism id is part
of an assignment's **verification identity**
(`sample_peak_id|assigned_formula|ionization_mechanism_id`), so a null one makes
an imported row a different identity from the in-app row for the same peak and
formula, and a verdict on one does not carry to the other. The **fit view** also
refuses to open without it.

So publish resolves by default: peaky's adduct labels map to the server's
mechanism names, and those are looked up on the deployment. Only ids that exist
there are sent, so an adduct this deployment does not know publishes as null
rather than as a guess &mdash; a supplied id must exist *and* carry the sample's
polarity or the whole import is refused. An `iso_child` inherits its owner's
mechanism, the way it inherits the owner's formula and the way the in-app engine
writes the family.

### 4. Rows that cannot be published

- **Synthetic sub-peaks** from composite de-blending are dropped: they exist in
  no Mascope peak file, and every `sample_peak_id` must.
- Rows without a usable m/z or intensity are dropped.

Everything dropped is reported in the summary rather than silently.

## What the server owns

An imported ledger is data a workspace editor asserts, not a computation the
server performed. So an import may say what it *found*, but not write the values
Mascope presents as its own calibrated judgement:

- `p_correct`, its provisional flag and the adduct corroboration count stay
  **empty** on imported rows. The keys they are derived from (`p_correct`,
  `calibration`, `corroboration`) are **stripped** from the top level of the
  provenance blob. `publish` drops them before sending and tells you it did.
- `evidence` is **overwritten** with the server's own derived value, so the
  number shown beside a tier is the number that was validated.
- peaky's own figures survive under `provenance.engine_provenance`, which the
  server stores verbatim and never interprets. That is what keeps "whose number
  is this" answerable when the two engines are compared. The *verdict* has a
  column of its own now (`engine_tier`), so the blob is the **explanation** —
  `tier_reason`, `confidence`, the arbitration numbers, the commentary — rather
  than the answer.

`calibration` is **required** and is what replaces the m/z verification gate an
import bypasses. peaky discloses honestly: it calibrates its mass axis from its
own commits, not against Mascope's verified assignments, so the run records
`provisional: true` and `verified_against_mascope: false`. `--calibration-note`
adds free text.

## Uploading

A dense ledger is too large for one request, so an import assembles over several
and `publish` handles the whole protocol:

- chunks are sized by **serialized bytes as well as row count** (1000 rows and
  5 MB per request; the deployment's real row cap comes back on the first
  response and is adopted for the rest);
- `chunk.index` is a **row offset**, not a counter, and it is what makes a
  retry safe — the SDK retries POSTs on timeouts with no way to opt out, so a
  chunk the server applied but never acknowledged is re-sent and recognised as a
  replay instead of duplicating rows;
- `--import-id` covers the same hazard for the request that *creates* the run,
  which has no offset to be idempotent about. A fresh id is generated per run.

**Resuming.** If an upload dies part-way it leaves a non-terminal `importing`
run, and that run refuses every later import *and* in-app assignment for the
sample until it is cleared. Re-run with the same `--import-id` to continue it, or
delete the run to release the sample. Re-running a *finished* import id is not a
resume: the server returns the run it already made and nothing is duplicated.

**Republishing** creates a new run; imports are append-only and never touch an
existing run. Mind retention, though: Mascope keeps the newest few completed runs
per sample, so republishing repeatedly can evict older runs.

## Partial ledgers

An imported run may cover a subset of the sample's peaks, but the batch view does
not merely tolerate that: folding a sample in **replaces** all of its prior batch
occurrences. Publishing 20 rows of interest on a 30,000-peak sample withdraws
that sample's other peaks from the batch overview and can delete anchors it alone
supported. Publish the whole ledger — including the unexplained residual, which
becomes `unassigned` rows — unless you specifically want otherwise.

## Options

| flag | meaning |
|---|---|
| `--sample-id` | target sample (default: the ledger's own `sample_item_id`) |
| `--manifest` | run config to publish (default: the `*_manifest.json` beside the ledger) |
| `--intensity auto\|height\|area` | which intensity to publish; `auto` reads the instrument |
| `--assigned-band` / `--candidate-band` | evidence-scale bands to tier and declare |
| `--no-resolve-mechanisms` | stop resolving adduct notation to this deployment's ionization-mechanism ids. Resolving is the default |
| `--engine-version` | version string stamped on the run |
| `--calibration-note` | extra text for the calibration disclosure |
| `--import-id` | idempotency key; re-use to resume an interrupted upload |
| `--max-rows` | rows per request (the server lowers it if the deployment caps tighter) |
| `--dry-run` / `-o` | translate and validate only; optionally write the payload out |

## Failure modes worth recognising

- **422 on tier coherence** — you should not see this, because `publish` sends
  no `tier` and the server derives it. If it appears, something is sending one:
  the message names the evidence and the fit it came from.
- **422 on peak existence** — a `sample_peak_id` is not in the sample. Usually
  the ledger was computed against a different sample or server.
- **409** — the sample already has a run in flight (an in-app assignment, or an
  import that never finished). The message says how to recover.
- **413** — the request body exceeded the proxy's `client_max_body_size`, which
  is enforced before the API sees it. Publish with a smaller `--max-rows`.

A dry run catches everything except peak existence and the two conflict cases,
which need the server.
