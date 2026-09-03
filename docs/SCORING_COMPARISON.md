# Scoring: Mascope backend vs. `mascope_tools` (peaky local)

> Unstaged analysis note. Background: peaky's default scorer is now in-process
> `mascope_tools.score_pattern`; the server's `match_compounds` uses a *different*
> formula. This documents the difference, its effect, and the consolidation path.

## The two formulas

**Mascope backend** — `libraries/match/src/mascope_match/compute/isotopes.py`, per
isotopologue *i*, then aggregated per ion in `.../match/lib/match_aggregate.py`:

```
mz_termᵢ    = max(0, 1 − 0.01·|ppmᵢ|)                 # gentle: 0 only at 100 ppm
abund_termᵢ = 1 − min(1, |observed_relᵢ/predicted_relᵢ − 1|)
scoreᵢ      = abund_termᵢ × mz_termᵢ                  # MULTIPLIED; unmatched isotopologue → 0
ion_score   = Σᵢ scoreᵢ · predicted_relᵢ              # abundance-weighted over the FULL predicted envelope
```

**mascope_tools / peaky** — `libraries/tools/src/mascope_tools/composition/heuristic_filter.py::score_pattern`:

```
mass      = max(0, 1 − mean|ppm|/5)                   # steep: 0 at 5 ppm (the tolerance)
intensity = max(0, 1 − mean|abund_err|/0.4)
pattern   = cosine_similarity(predicted_rel, observed_rel)   # ← no backend equivalent
ion_score = 0.6·mass + 0.2·pattern + 0.2·intensity    # averaged over MATCHED peaks only
```

(`ISOTOPE_MATCHING_MZ_TOLERANCE_PPM = 5`, `ISOTOPE_MATCHING_INTENSITY_TOLERANCE = 0.4`.)

## Four structural differences

1. **Missing isotopologues: punished vs. ignored.** Backend sums over the *whole*
   predicted envelope, so an undetected M+1 contributes a hard 0 and pulls the ion
   score down. `mascope_tools` averages over *matched* peaks only — a missing M+1
   simply doesn't count.
2. **Mass-error scale differs.** Backend `1 − 0.01·ppm` (zero at 100 ppm); local
   `1 − ppm/5` (zero at 5 ppm). At 1 ppm: backend 0.99, local 0.80.
3. **Cosine pattern term exists only locally** (0.2 weight). Backend encodes pattern
   only implicitly (per-peak abundance error × abundance weighting).
4. **Combination shape: multiply vs. linear blend.** Backend multiplies mass×abundance
   per peak (one bad dimension zeroes that peak). Local linearly blends three *averaged*
   terms, so the score has a ~0.4 floor (pattern+intensity) even with a poor mass fit.

## Effect on scores — regime-dependent, not a constant offset

Computed on identical inputs (`mass_tol=5`, `int_tol=0.4`):

| case | local | backend | local − backend |
|---|---|---|---|
| Full envelope, 0.3 ppm, good abundance | 0.932 | 0.989 | **−0.057** |
| **M0 only detected** (M+1 below noise) | 0.963 | 0.897 | **+0.065** |
| Br₂ M0+M2, 0.4 ppm | 0.939 | 0.972 | −0.032 |
| 1 ppm, abundance off by 0.2 | 0.818 | 0.966 | −0.148 |

- **Full envelope seen → local lower** (its steep 5-ppm mass term at 0.6 weight can't reach ~1.0).
- **M0-only seen → local higher** (averaging + floor forgive missing peaks; the backend's envelope-weighted sum penalizes them).
- Real spectra are isotopologue-sparse, so the M0-only regime dominates → the measured
  **net +0.05–0.07** (local higher) on the Bromide/Uronium parity eval. The **±0.06 spread**
  *is* this regime-dependence. Ranking is mostly preserved (both monotonic in mass/abundance
  fit), hence 0.91–0.98 argmax agreement; disagreements cluster on near-ties.
- **Where they disagree most is the weak/sparse peaks — i.e. the Assigned/Candidate
  boundary.** So the discrepancy is not cosmetic; it nudges the hardest calls.

## Consolidation (recommended)

Two implementations of one concept, diverging exactly where it matters, is a real defect.

- **One shared scorer in `mascope_tools`.** Have the backend `match_compounds` call
  `mascope_tools.score_pattern` (or both call one shared function) so server and local
  scores are identical *by construction*. This is the **same direction as the filed
  "retire the molmass fork / converge backend onto mascope_tools" issue** — scoring
  convergence rides along with it.
- **Calibrated once:** peaky's tier thresholds stop needing the offset caveat; server
  and local runs agree; the scoring science lives in one place.
- **Science decision to make:** the backend's "penalize missing isotopologues" is
  chemically principled but harsh when M+1 is genuinely below noise; local's "average
  over matched + explicit cosine" is noise-robust but credits incomplete envelopes.
  Best of both → keep cosine + mass precision, add an **envelope-completeness penalty
  weighted by detectability** (don't punish an M+1 predicted below the noise floor).
- **Caveat:** changing the backend formula re-scores history — existing `match_score`s
  shift and downstream thresholds need re-checking. A coordinated, versioned change.
