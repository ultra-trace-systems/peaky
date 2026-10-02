# Certified-neutral discovery (pass 7)

*Module:* `peaky/assignment/certified_neutral.py` (pure) + `run_pass_certified`
in `passes/directors.py` · *Script:* `scripts/certify_neutrals.py` · *Since:*
2026-07 (branch `feat/certified-neutral`).

## The problem it solves

A single peak's neutral interpretation is **mass-degenerate**: at m/z 300+ with
a ±3 ppm window dozens of formulas fit, and monoisotopic elements make it
worse — which is exactly why the per-peak grid keeps **P** closed and the
positive-mode grid stays CHON. The cost: whole compound families (the
organophosphate pesticides, sulfonamide plasticizers) are structurally
invisible, and the only prior remedy was a hand-curated known-species
whitelist (pass 0) — a gatekeeper that can only find what someone listed.

## The mechanism

0. **Interrogation scope**: the pass reads the UNEXPLAINED residual **plus
   weak M0 owners** — unlocked commits at Low/Suspect confidence or flagged as
   arbitration near-ties. A single-channel mass fit (e.g. a `[M+Na]+` reading
   on an instrument with no Na source) is weaker evidence than a multi-channel
   certificate, so it must not shield its peak from re-reading. Locked and
   Good/High-confidence owners are never touched. A weak incumbent is
   **displaced** (via `clear_assignment`, audit-trailed in the commentary)
   only by a *strong* certificate (isotope-confirmed or ≥3 channels); when the
   incumbent already matches the certified neutral it is counted as
   corroborated instead.
1. **Group** the interrogable peaks by known channel offsets in a *single
   spectrum*: different adducts (`[M+H]+` vs `[M+NH4]+`) and **reagent-cluster
   ladder rungs** (`[M+H]+`, `[M+urea+H]+`, `[M+2urea+H]+`; step = neutral urea
   60.03236). `channel_offsets` builds the offset set per profile;
   aliased offsets (proton + 1 urea ≡ the registered urea adduct) are
   de-duplicated so one physical ion can never fake a second channel.
2. **Certify**: a group whose back-calculated neutral masses converge within
   3 mDa across ≥2 *distinct* channels imposes N independent mass constraints
   on one unknown — a `Certificate` (`find_certificates`). A k-rung ladder also
   converges at cores shifted by whole repeat units (each peak re-read with
   ±n ureas); `_select_certificates` resolves this by parsimony: most
   channels, then fewest total assumed cluster units.
3. **License**: only for a certified core is the **expanded element box**
   enumerated (`enumerate_certified`: P/S/Cl opened past the per-peak caps).
   Safe because the certificate supplies 2–3 mass constraints, not one: at the
   NBBS core, the box collapses to a handful of candidates. CHON-only cores
   are skipped (pass-1 territory). A core with >24 candidates is
   mass-saturated → skipped (degeneracy guard).
4. **Score + gate**: candidates go through the standard oracle
   (`score_candidates`); the winner must be server-anchored on **≥2 member
   channels** with on-calibration ppm; S/Cl/Br winners want their diagnostic
   heavy-isotope envelope (³⁴S/³⁷Cl/⁸¹Br — ¹³C never counts): a line under an
   ion the certificate commits, a member's (C11+c). A line whose only
   diagnostic isotope is the bromide reagent's ⁸¹Br (`81Br`, `13C+81Br`,
   `81Br2`; ¹³C, ¹⁸O, ¹⁵N or ³³S beside it adds none), under a committed
   bromide cluster of a winner without Br, counts only on a certificate of
   ≥3 channels — on two, one file, two ions and the reagent's own line are not
   enough (the user, 2026-10-02) — while the winner's own ³⁴S / ³⁷Cl, alone or
   in a combined line (`13C+37Cl`, `81Br+37Cl`), and a Br winner's ⁸¹Br count
   on two channels as on three ([ASSIGNMENT_DETAIL.md](ASSIGNMENT_DETAIL.md)
   §3.0).
5. **Commit** (`method=certified:multi-channel`): the same certified neutral
   is committed onto *every* member peak under its own channel label — so the
   tier engine's cross-channel corroboration sees the certificate. Ladder
   rungs above the oracle's registered channels (e.g. `[M+2R+H]+`) commit as
   `certified:ladder-rung` with the rung's own ppm vs the certified core (so
   the calibrated mass-gate audit judges them like any commit).
6. **Optional TS layer**: when a batch `ts_peaks` is available, member-channel
   time co-variation (`ts_covariation`, minimum pairwise log-r) is annotated as
   further corroboration; anti-correlated members (r<0.3) veto the commit.
   Strictly optional — a batch whose mass range excludes the reagent ions
   (e.g. positive urea-CIMS starting at m/z 122) still gets the full
   mass-domain mechanism.

## Relationship to the other passes

This pass **inverts pass 5**: completion walks from *known* neutrals to their
missing cross-channel partners; certified-neutral walks from *unknown* peak
groups to a licensed neutral mass. It runs after pass 5 / before the audits, so
its commits face the same calibrated mass gate as everything else. It also
shrinks pass-0's role back toward naming/priors: off-grid families no longer
*require* a whitelist entry to be discoverable (the organothiophosphate list
remains as a fast path + naming layer).

## Ground truth (validated by hand before implementation, 2026-07-07)

* **NBBS urea ladder** — 214.0896 `[M+H]+` / 274.1220 `[M+urea+H]+` / 334.1544
  `[M+2urea+H]+` all back-calculate to core 213.0823 within **0.1 mDa** →
  C10H15NO2S (N-butylbenzenesulfonamide, nylon-plasticizer instrument
  background; ¹³C→C₁₀ and ³⁴S→S₁ confirmed). The mono-urea adduct was the
  brightest ion in the whole spectrum.
* **Malathion** — 331.0433 `[M+H]+` / 391.0756 `[M+urea+H]+` / 348.0699
  `[M+NH4]+` → core 330.0360 (C10H19O6PS2), invisible to the P-free grid.
* First offline run on a real per-file ledger (339 unexplained peaks)
  certified 37 cores and **blind-rediscovered benzothiazole** (core 135.0143,
  spread 0.014 mDa, C7H5NS) — a compound independently confirmed via the
  reagent-zeroing analysis.

## Standalone use

```bash
# offline (no credentials): certificate table + off-grid candidates
python3 scripts/certify_neutrals.py RUN_DIR/per_file/<SID>_ledger.csv --reagent Ur

# with time-series corroboration
python3 scripts/certify_neutrals.py LEDGER.csv --reagent Ur --ts RUN_DIR/data/Ur_ts.parquet

# full: score + commit through the live oracle
python3 scripts/certify_neutrals.py LEDGER.csv --reagent Ur --sample-id <SID> -o certified.csv
```

## Knobs

| name | default | meaning |
|---|---|---|
| `CORE_TOL_MDA` | 3.0 | channel-convergence tolerance (mass domain) |
| `_CERT_ENUM_TOL_MDA` | 2.0 | candidate-enumeration window around the core |
| `_CERT_MAX_CANDIDATES` | 24 | mass-saturation guard (skip the certificate) |
| `min_channels` | 2 | channels required to certify |
| `max_cluster_order` | 2 | ladder rungs beyond the bare adduct |

## Future (not built)

* **Urea-order / zeroing-lift corroboration**: on batches that retain zeroing
  periods, an ion's zeroing lift grows with its cluster order (measured: NBBS
  n=0/1/2 → +0.01/+0.10/+0.74) — an independent adduct-stoichiometry check.
* **Fragment-cation modeling** (the `(CH3O)2P=S+` class) still needs a ledger
  schema extension (a fragment/charge role); certificates cannot represent
  ions with no intact neutral.
