# Peaky — Reagents & profiles (the single mode switch)

This document explains **how one reagent choice configures the whole pipeline** —
the analyte adduct channels, the grid element box, the time-series normaliser, and
the labeled reagent-ion clusters — and **how reagent-cluster m/z are enumerated
and matched**. It is a module deep-dive companion to
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline, §6 *Reagent profiles*),
[`CHEMISTRY.md`](CHEMISTRY.md) (the adduct shifts + grid the profile drives), and
[`ASSIGNMENT.md`](ASSIGNMENT.md) (which consumes the reagent labels).

**Code:** `peaky/chem/profiles.py` (`ReagentProfile` + the registry + `resolve`)
and `peaky/chem/reagents.py` (the cluster-ion library + `label_reagents`).

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `reagents.py` / `profiles.py`; if you change one there, change it here.

---

## 1. What this stage does

In chemical-ionization MS the reagent anion/cation forms **bright cluster ions
that are not sample chemistry** (bare Rₙ, R·water, R·HBr, protonated-urea
clusters). Left alone they dominate the unexplained residual by signal. This layer
does two things:

1. **Profile selection** — pick (or auto-detect) one `ReagentProfile`, which is
   *everything the pipeline needs to treat the reagent correctly*: polarity, the
   analyte adduct channels, the grid ranges, the normaliser (`reagent`/`tic`), the
   reagent-ion regex, the assignment context, and any isotopic purity. **The
   reagent is the single switch that flips the same pipeline negative- or
   positive-mode.**
2. **Reagent labeling** — enumerate the cluster m/z (with halogen isotopologue
   combinations) and mark matching ledger peaks `role='reagent'`, recording the
   *known* cluster ion formula as the assignment.

```
peaks  ──► resolve('auto', peaks)              name/alias ──► resolve(name)
              │ detect_adduct in sample? → profile      │
              │ else polarity → profile                 ▼
              ▼                              ReagentProfile {polarity, adducts,
        ReagentProfile  ────────────────────► ranges, normaliser, reagent_ion_re,
              │                                 detect_adduct, context, purity}
              ▼  reagent_for_adducts → library key ("Br"/"I"/"Cl"/"urea")
        build_library:
          halide  → bare Rₙ⁻ (odd closed-shell, even radical) · Rₙ·neutralₖ · ROₙ⁻
                    (all halogen isotopologue combos)
          urea    → [Rₙ+H]⁺ protonated series (n=1..6)
              ▼  label_reagents (±ppm 15, nearest label, known ion_formula)
        ledger peaks marked role='reagent'
```

---

## 2. Inputs

- A **reagent selector**: a name/alias (`"Br"`, `"uronium"`, `"15no3"`), or
  `"auto"` + a loaded peak table.
- Optional **`--reagent-config`** JSON/TOML registering extra/override profiles.
- For labeling: the **ledger** (peaks with `role`/`mz`/`peak_id`).

---

## 3. The transformation, stage by stage

1. **Resolve a profile** (`profiles.resolve`). A name/alias hits `_BY_ALIAS`
   directly. `"auto"` detects from the sample: the **diagnostic `detect_adduct`**
   among the server's own adduct mechanisms (`io_mascope.detect_adducts`) wins
   first; failing that, **polarity** (`_detect_polarity`) picks the first
   profile of that sign. A `config` path is loaded (registered) before resolving.

2. **The profile configures everything else.** `ReagentProfile` (frozen) carries:
   `polarity`, `adducts` (analyte channels), `ranges` (the grid box string fed to
   [`CHEMISTRY.md`](CHEMISTRY.md)), `normaliser` (`reagent` or `tic`, for the
   TS/correlation layer), `reagent_ion_re` (regex on `ion_formula` picking reagent
   ions), `detect_adduct`, `context` (the assign-time mode + VK priors + caps), and
   `purity` (a labelled reagent's isotopic purity, threaded to
   `predict_isotopes`), and — for labelled reagents — `label_isotope` /
   `label_max` (the caret heavy isotope a *covalent product* can carry and its max
   count; drives the heavy-isotope rescue in `labeled.py`). Built-ins: **`BR`**
   (Br⁻, neg, normalise on reagent), **`UR`** (urea/uronium, pos, normalise on
   TIC), **`NO3`** (nitrate, neg, reagent), **`NO3_15N`** (¹⁵N nitrate, neg, TIC,
   `purity 0.98`, `label_isotope='^N'`, `label_max=2`), **`IODIDE`**
   (I⁻, neg, normalise on reagent; adducts `[M+I]⁻`/`[M-H]⁻`/`[M+I2]⁻`/
   `[M-H+I2]⁻`).

   > **Iodide is a soft adduct source with a monoisotopic reagent.** Most analytes
   > appear as the `[M+I]⁻` cluster; strong acids also deprotonate to `[M-H]⁻` (both
   > server-confirmed on the 2026-07-21 batch — HNO₃ as `[HNO3+I]⁻` *and*
   > `NO3⁻`, formic/acetic acid as `[M-H]⁻`). Because ¹²⁷I is the **only** iodine
   > isotope, covalent iodine cannot be isotope-confirmed and is kept **OFF the
   > neutral grid** (no `I` in `IODIDE.ranges`, like F/P) — iodine reaches a neutral
   > only via the adduct or the pass-0 `reactive_iodine` known-species family (HOI,
   > HIO₂, HIO₃, OIO, INO₂, INO₃, ICN, INCO, ICl, IBr — the canonical iodide-CIMS
   > analytes; 8 committed at <0.6 ppm on the 2026-07-21 batch; ICl/IBr carry
   > their ³⁷Cl/⁸¹Br envelope). The oxyacids also own their `[M-H]⁻` lines
   > (IO⁻/IO₂⁻/IO₃⁻ are **not** reagent-oxide labels under iodide). `[M+I2]⁻` is
   > kept as a secondary channel (server mechanism `+I2-`); the time-stable
   > pure-iodine oxides (I₂O⁻/I₃O⁻) are pre-labelled reagent background, so that
   > channel does not run wild on them.
   >
   > **`[M-H+I2]⁻` — the deprotonated-acid · I₂ cluster.** Acids the run already
   > believes on `[M+I]⁻`/`[M-H]⁻` ALSO appear as the conjugate base bound to I₂
   > (I₂⁻· is the brightest reagent ion): `[HCOOH-H+I2]⁻` @298.807, acetic
   > @312.823, carbonic @314.802, glycolic @328.818, HNO₄ @331.792. A
   > relabel-only decomposition alias like the Br `[M+HBr+Br]⁻` — **no server
   > mechanism**; the pass-3 resolver (`_resolve_acid_i2_clusters`) scores the
   > covalent alias `(A-H+I) [M+I]⁻` (the identical ion — the `CHIO2`-style
   > reading the series passes once invented) and commits the acid. It claims
   > UNEXPLAINED peaks only and skips iodine-bearing / O-free anchors, so the
   > pass-0 reactive-iodine species keep their I₂X⁻ lines (`HOI2⁻`/`I2NO2⁻`
   > were ruled ambient analytes on TIME behaviour, not acid clusters).

   > **¹⁵N-nitrate ¹⁴NO₃-cluster hazard.** In a NOx-oxidation run the chamber holds
   > abundant *unlabelled* ¹⁴NO₃⁻, so a highly-oxygenated analyte X forms
   > `[X+¹⁴NO₃]⁻` — the **exact isobar** of the deprotonated covalent organonitrate
   > `[Y−H]⁻` (Y = X + HNO₃). `[M+NO3]-` (¹⁴N) is therefore **deliberately kept OFF
   > `NO3_15N.adducts`** (adding it would let the scorer arbitrate the isobar
   > arbitrarily); instead the post-tier `relabel_nitrate_clusters` pass re-reads a
   > covalent organonitrate as the cluster only when the parent X is independently
   > corroborated (ASSIGNMENT_DETAIL §3.9). The ¹⁵N cluster channel is the ordinary
   > `[M+^NO3]-`.

3. **Pick the cluster-library key** (`reagent_for_adducts`). From the analyte
   adducts: `CH4N2O` → `"urea"`; `Br` → `"Br"`; `I` → `"I"`; `Cl` → `"Cl"`. This
   is the **cluster-library key, not** the arbitration `reagent_element` (a
   molecular reagent puts no halogen in the neutral, so `assign.run` sets
   `reagent_element` only for the halogen keys).

4. **Build the cluster library** (`build_library`). For a **halide** (Br/Cl/I):
   - **bare Rₙ⁻**, `n = 1..max_n (4)`, every isotopologue combination
     (`combinations_with_replacement`); **odd n** are closed-shell anions (R⁻,
     R₃⁻), **even n** are radical anions (R₂⁻·, R₄⁻·); each anion adds `M_E`.
   - **Rₙ⁻·(neutral)ₖ**, `k = 1..max_neutral (1)`, over `_CLUSTER_NEUTRALS`
     (`H2O`, `HF`) plus the reagent's **own** shed hydride (`HBr`/`HCl`/`HI` —
     an iodide library carries `[I+HI]⁻`, never a phantom `[I+HBr]⁻`).
   - **reagent-oxide anions** RO⁻/RO₂⁻/RO₃⁻, *both* halogen isotopologues —
     **Br/Cl only**: for iodine the IOₓ⁻ anions are ion-identical to the `[M-H]⁻`
     deprotonation of the iodine oxyacids (IO₃⁻ **is** iodic acid's dominant
     channel — the NPF tracer), so they are left for pass-0 `reactive_iodine`
     (the HNO₃/NO₃⁻ ruling, applied to iodine oxides).
   - **iodine only:** the pure-iodine-oxide background clusters `_IODINE_BACKGROUND`
     (`I₂O⁻` ~2M cps, `I₃O⁻`) — bright, time-stable source ions. The time-VARYING
     poly-iodide ions (HOI₂⁻ 55×, I₂NO₂⁻ 2.3× over the 2026-07-21 batch) are
     deliberately NOT here: they are the `[M+I]⁻` reading of the ambient
     reactive-iodine analytes HOI / INO₂ (pass-0 `reactive_iodine` family).

   For a **molecular reagent** (urea): the protonated series **[Rₙ+H]⁺**,
   `n = 1..max_n (6)` (`_build_positive_library`) — a cation (lose an electron),
   repeat unit `CH4N2O`, no isotope branching.

5. **Label matching peaks** (`label_reagents`, `ppm 15`, `only_unexplained=True`).
   `bisect` the sorted library masses within `±(mz·ppm·1e-6)`, take the **nearest**
   label, and `L.mark_reagent` records the **known cluster ion formula** as the
   assignment (a reagent cluster has a known formula → it is *assigned*, just a
   different class — never left blank/red in the residual). Returns the count.

6. **The positive cluster adducts carry N/O into every analyte ion** (uronium/
   ammonium mode). The analyte channels of a positive N-reagent are not clean
   protonation: the reagent contributes atoms to the observed ion. `[M+NH4]⁺` adds
   `NH3` over `[M+H]⁺`, and the uronium/urea cluster `[M+(CH4N2O)H]⁺` adds a whole
   `CH4N2O`. Two downstream consequences fall out of that extra N/O and are handled
   in assignment, not here:
   - **CHO-via-N-cluster is isobaric with protonated CHON.** A CHO neutral seen as
     `[M+NH4]⁺` or `[M+(CH4N2O)H]⁺` yields the *same* ion formula as a CHON neutral
     seen as `[M+H]⁺` (e.g. `C12H14O4[M+NH4]⁺` and `C12H17NO4[M+H]⁺` are both
     `C12H18NO4⁺`). Mass and isotopes cannot separate them — the tier layer demotes
     such a commit to Candidate unless an extra-spectral discriminator corroborates
     it. See the reagent-N isobar gate in
     [`ASSIGNMENT_DETAIL.md`](ASSIGNMENT_DETAIL.md) (tiers).
   - **A pure hydrocarbon "seen via an N-cluster" is re-read as an N-heterocycle.**
     A hydrocarbon has no site to bind `NH4⁺`/uronium and would show `[M+H]⁺`, so an
     `[M+NH4]⁺` / `[M+(CH4N2O)H]⁺` reading of a bare CₓHᵧ is implausible; assignment
     re-reads it as `[M+H]⁺` of the N-heterocycle `M' = M + (cluster − H)` (see
     [`ASSIGNMENT_DETAIL.md`](ASSIGNMENT_DETAIL.md), reagent-N re-read).

---

## 4. Constants reference

`reagents.py` (library) + `profiles.py` (profiles).

| constant | value | role |
| --- | --- | --- |
| `_HALOGEN_ISO` | Br: ⁷⁹/⁸¹ · Cl: ³⁵/³⁷ · I: ¹²⁷ | reagent-halogen isotope masses/labels |
| `_CLUSTER_NEUTRALS` | `H2O`, `HF` + the reagent's own hydride (`HBr`/`HCl`/`HI`, added per reagent in `build_library`) | neutrals that cluster on a halide core (organic acids deliberately removed) |
| `_IODINE_BACKGROUND` | `I2O`, `I3O` | iodine-only pure-oxide source-background clusters (added to the `"I"` library; HOI₂⁻/I₂NO₂⁻ are pass-0 analytes, not background) |
| `_POSITIVE_REAGENTS` | `{urea: CH4N2O}` | molecular positive reagents (protonated series) |
| reagent-N cluster `[M+NH4]⁺` | `{N:1, H:3}` over `[M+H]⁺` | ammonium cluster adduct N/O added to the ion (isobar gate / re-read) |
| reagent-N cluster `[M+(CH4N2O)H]⁺` | `{C:1, H:4, N:2, O:1}` | uronium/urea cluster adduct atoms added to the ion (isobar gate / re-read) |
| `build_library` `max_n` | 4 (halide) / 6 (positive) | largest Rₙ cluster enumerated |
| `build_library` `max_neutral` | 1 | max neutral adducts per halide core |
| `label_reagents` `ppm` | 15.0 | reagent-cluster mass-match window |
| `label_reagents` `only_unexplained` | True | only relabels still-unexplained peaks |
| `ReagentProfile.normaliser` | `reagent` / `tic` | TS/correlation normalisation basis |
| `BR.ranges` | `C0-40 H0-80 N0-3 O0-18 S0-2 Cl0-2 Br0-2` | bromide grid box |
| `UR.ranges` | `C0-40 H0-90 N0-8 O0-15 S0-2` | uronium grid box |
| `NO3.ranges` / `NO3_15N.ranges` | `C0-40 H0-60 N0-3 O0-25 S0-2` | nitrate grid box |
| `NO3_15N.purity` | 0.98 | ~98 % ¹⁵N reagent (→ `predict_isotopes`) |
| `IODIDE.ranges` | `C0-40 H0-80 N0-3 O0-20 S0-2 Cl0-1` | iodide grid box (**no I** — covalent iodine is monoisotopic, off-grid) |

---

## 5. Metrics, defined

- **cluster ion m/z** — built from `_HALOGEN_ISO` masses ± `M_E` (anion +e,
  cation −e), summed over the isotopologue combination + any clustered neutral.
- **reagent ppm match** — `(mz − ion_mz)/ion_mz · 1e6`; within ±15 ppm of the
  *nearest* library entry the peak is labeled reagent (the ppm is recorded in the
  note).
- **normaliser** — `reagent` divides traces by a reagent-ion signal; `tic` divides
  by total ion current (used when the reagent ions sit below the acquisition
  window, e.g. ¹⁵N-nitrate / uronium).
  - **UR normalises on TIC** because a positive urea-CIMS spectrum starts at ~m/z 122
    and therefore *excludes* the 61/121 uronium reagent ions — there is no in-window
    reagent signal to divide by.
  - **The uronium reagent is essentially flat over the diurnal cycle** (measured on a
    one ambient urea-CIMS batch, 2026-07-07: m/z 61 monomer ~5% amplitude, m/z 121 dimer ~2%,
    and the dimer even weakly *anti*-correlates with the flat-panel common-mode
    ~15:00 afternoon wave). Consequence: even if the reagent were in-window,
    reagent-normalisation would **not** remove that afternoon common-mode wave — a flat
    divisor can't cancel a varying signal. That wave is therefore **real ambient/
    environmental signal**, not a reagent-flow or detection-sensitivity artifact.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `ReagentProfile` | the run's mode config: polarity, adducts, ranges, normaliser, reagent_ion_re, detect_adduct, context, purity |
| `build_library` | `[(label, ion_mz, ion_formula)]` — the enumerated reagent-cluster ions |
| `label_reagents` | count of ledger peaks set to `role='reagent'` (each with a known `ion_formula`) |
| `reagent_for_adducts` | the cluster-library key (`"Br"`/`"I"`/`"Cl"`/`"urea"`/`None`) |

---

## 7. Properties, invariants & gotchas

- **Organic acids are NOT reagent neutrals.** A `[Br+acid]⁻` ion *is* the
  `[acid+Br]⁻` primary `[M+Br]⁻` analyte channel — the labeler used to steal real
  ambient acids (formic acid's 232k-cps line among them) and bury them as reagent.
  `_CLUSTER_NEUTRALS` is now water + the reagent's own HBr + background HF only.
- **Both Rₙ parities are real** reagent ions: odd = closed-shell anion, even =
  radical anion (e.g. the Br₂⁻· di-bromide, the I₂⁻· di-iodide — the brightest ion
  in an iodide source). All are pure reagent → labeled, not left in the residual.
- **Iodide reagent-acid clusters are the analyte channel, not reagent.** Exactly
  like the Br organic-acid ruling: `[HNO3+I]⁻` (189.90), `[H2O2+I]⁻` (160.91),
  `[HCOOH+I]⁻` (172.91) are the `[M+I]⁻` analyte reading of HNO₃/H₂O₂/formic acid
  (server-confirmed), so they are **left for assignment**, not stolen into the
  cluster library. Only `H2O` clusters onto the iodide core (`_CLUSTER_NEUTRALS`).
  What *is* pre-labelled is the bare Iₙ⁻ ladder, IOₓ⁻, and the pure-iodine
  `_IODINE_BACKGROUND` clusters.
- **Reagent-vs-analyte for poly-iodide ions is decided by TIME behavior.** On the
  2026-07-21 batch the bare ladder (I⁻/I₂⁻·/I₃⁻) and I₂O⁻/I₃O⁻ are stable (<±10%)
  → source background; HOI₂⁻ swings 55× (photochemical daytime HOI) and I₂NO₂⁻
  2.3× → ambient analytes, committed by pass 0 as HOI/INO₂ `[M+I]⁻`. Caveat:
  ambient I₂ is detected as `[I2+I]⁻` = **the same I₃⁻ the source makes** — a
  bright, stable I₃⁻ hides any ambient-I₂ contribution; if a campaign targets I₂,
  check I₃⁻/I₂⁻ ratio drift before trusting the reagent label. The same blind
  spot applies to the **IO radical**: `[IO+I]⁻` is composition-identical to the
  locked `I₂O⁻` source cluster — check I₂O⁻/I₂⁻ ratio drift if IO matters. (OIO
  is covered: `[OIO+I]⁻` = I₂O₂⁻ at 285.799 is a `reactive_iodine` known species.)
- **Iodine is monoisotopic → off-grid, and its Br-specific isotope machinery is
  inert.** `_DIAG` (the diagnostic-isotope map) has no `I`, so the complexity prior
  on a covalent I is *never* iso-waived; the Br-doublet clear-both, the
  `relabel_reagent_halocarbons` relabel, and `_prefer_adduct_reading` (needs `HI` in
  `REPEAT_UNITS`) are all `Br`-gated and go inert with `reagent_element='I'`. The
  composite `M+1` test still runs (`has_halogen_adduct` is true for I) and is in fact
  *cleaner* under iodide — a monoisotopic reagent adds nothing to the `M+1` region.
- **A molecular reagent puts no halogen in the neutral.** `reagent_for_adducts`
  returns the *library key*; `reagent_element` (the arbitration complexity element)
  is set only for halogen reagents — never for urea.
- **Charge bookkeeping:** halide clusters *gain* an electron (`+M_E`); the
  protonated positive series *loses* one (`−M_E`).
- **Positive N-reagents make CHO readings isobaric with CHON.** Because `[M+NH4]⁺`
  and `[M+(CH4N2O)H]⁺` fold reagent N/O into the observed ion, a CHO neutral via one
  of these clusters is the *same ion formula* as a protonated CHON neutral — neither
  mass nor isotopes distinguish them. This is a positive-mode-only hazard (the
  halide anions add no N); the tier layer resolves it (isobar gate,
  [`ASSIGNMENT_DETAIL.md`](ASSIGNMENT_DETAIL.md)). It is why `UR.ranges` opens N to
  `N0-8` — the analyte space genuinely spans both readings.
- **A bare hydrocarbon has no N-cluster site.** `NH4⁺`/uronium bind at a polar/basic
  site a pure CₓHᵧ lacks; a hydrocarbon ionises as `[M+H]⁺`. An `[M+NH4]⁺` /
  `[M+(CH4N2O)H]⁺` reading of a hydrocarbon is therefore re-read as the N-heterocycle
  `[M+(cluster−H)+H]⁺` (assignment, not this layer) — *unless* that hydrocarbon also
  has a genuine `[M+H]⁺` row (a real terpene forming `[M+NH4]⁺`), which is left
  alone.
- **`detect_adduct` disambiguates isotopic twins.** `NO3` vs `NO3_15N` differ only
  by their diagnostic adduct (`[M+NO3]⁻` vs `[M+^NO3]⁻`), so auto-detect picks the
  right one; `purity` then flows to the labelled-reagent envelope predictor.
- **A reagent cluster is an assignment, not a blank.** Its formula is known, so it
  is committed with that `ion_formula` (a distinct class), never red in the report.
- **New reagent = a `ReagentProfile`, no fork.** `register` / `from_dict` /
  `load_config` add reagents from JSON/TOML (`--reagent-config`); a top-level list,
  a `{"reagents": [...]}` wrapper, or a `{name: {fields}}` mapping all parse.
- **Auto-detect needs peaks**; with no diagnostic adduct it falls back to polarity,
  and a sparse positive sample can mis-detect as negative — pass `--reagent` to
  force it.

---

## 8. Code map

| function | role |
| --- | --- |
| `profiles.ReagentProfile` | the frozen per-reagent config dataclass |
| `profiles.resolve` | name/alias or `auto` (detect_adduct → polarity) → a profile |
| `profiles.register` / `from_dict` / `load_config` | registry + JSON/TOML reagent loading |
| `profiles._detect_polarity` | infer `+`/`−` from the peak table |
| `reagents.reagent_for_adducts` | analyte adducts → cluster-library key |
| `reagents.build_library` | enumerate the reagent-cluster ions (halide + positive) |
| `reagents._build_positive_library` | the `[Rₙ+H]⁺` protonated-reagent series |
| `reagents.label_reagents` | mass-match + mark ledger peaks `role='reagent'` |
