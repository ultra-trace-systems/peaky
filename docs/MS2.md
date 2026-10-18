# Peaky — MS2 fragment assignment (sub-formulas of the window ions)

This document explains **how an MS2 peak becomes a list of formula readings**:
every sub-formula of every ion in the isolation window that fits the peak's m/z,
kept only when the neutral loss it implies splits into real molecules. It is a
module deep-dive companion to [`CHEMISTRY.md`](CHEMISTRY.md) (exact masses, DBE,
Senior's rules — the arithmetic this module is built on).

**Code:** `peaky/ms2/subformula.py` (public API re-exported by `peaky.ms2`), plus
`chemistry.ion_mass`. Pure arithmetic: no Mascope, no I/O, no plotting. The input
is an MS2 peak list and the window's ion formulas; the output is a long table.
Where the peak lists come from — Mascope's MS2 endpoints read into energy-resolved
tables by `peaky/ms2/sources.py` and `peaky ms2 list | extract` — is §10; what is
built from them across collision energies — breakdown curves, E50s, fragment
onsets and attributions, `peaky ms2 analyze` — is §11; the scans behind the
spectra, and a raw file's MS2 rebuilt from its per-scan metadata (`ms2 extract
--per-scan | --file`, `ms2 files`), are §12.

> Keep this in sync with the code. Every threshold below is a named constant, a
> default argument or a literal in `subformula.py`; if you change one there,
> change it here.

---

## 1. What this stage does

A fragment of a singly-charged precursor is an **element-wise sub-formula of the
precursor ION** — the charge stays on the fragment — and the difference is the
neutral the precursor lost. A plain sub-formula search accepts any difference,
including ones no molecule has (a bare `C4`). This module accepts a reading only
when its loss **decomposes into ≤ `k_max` molecules**, and ranks the readings of
one peak against one parent.

It is a **composition test**, applied to every ion of the isolation window (the
target and its co-isolates) with the same rules. It never decides *which* window
ion a fragment came from: a peak two window ions can both explain keeps both
readings (`n_parents` 2).

```
MS2 peaks (m/z [+ intensity])          window ions {label: ion formula}
        │  apply_offset: mz_cal = mz / (1 + offset_ppm·1e-6)
        ▼                                         │ per parent, once (cached):
   for each peak × parent ◀──────────────────────┘ all sub-formulas → fragment gate
        │  bisect the parent's sorted sub-formula masses
        ▼  within max(tol_ppm·mz_cal·1e-6, tol_mda·1e-3)
   candidate fragment F  →  loss L = parent − F  (element-wise)
        │  decompose L into ≤ k_max molecules (library + ≤ 1 unnamed piece)
        ▼  no decomposition → not a reading
   ranked readings per (peak, parent)  →  long table (every peak appears)
```

---

## 2. Inputs

- **Peaks** — a DataFrame with `mz` (and optionally `intensity`), or a **1-D**
  array-like of m/z. Every m/z must be finite and positive. A 2-D input — an
  `(n, 2)` centroid array, `(mz, intensity)` pairs — raises: flattened, every
  intensity would be read as one more m/z.
- **Parents** — one ion formula string (`'C10H17+'`), or an ordered mapping
  `label → ion formula` (the target plus the co-isolated window ions), or a
  sequence of ion formulas (labels = formulas). Only strings: every parent goes
  through `parse_ion`'s checks, and a `(counts, charge)` tuple raises. Labels must
  be unique after `str()`. All parents must share one polarity.
- **`offset_ppm`** (default `0.0`) — the constant-ppm axis offset, usually from
  `mass_offset` on the surviving precursor (and optionally the reagent ion).
- **`tol_ppm`** `5.0`, **`tol_mda`** `0.3`, **`k_max`** `3`,
  **`allow_radical`** `True`.

Ion formulas are peaky's Hill strings with exactly one sign: `'C9H11+'`,
`'C10H17O5-'`, `'C4H10^NO4+'`. `'^N'` is ¹⁵N and a **distinct element**
everywhere — in masses and in counts — so a labelled-ammonium adduct's
`H3^N` loss (18.0236) is never confused with `H2O` (18.0106) or `H3N` (17.0265).

---

## 3. The transformation, stage by stage

1. **Parse the ions** (`parse_ion`, `format_ion`). Exactly one `+` or `-` is
   required: a formula without one is a neutral passed by mistake, and `2+` / `++`
   would be a doubly charged ion, which nothing here models. A trailing radical
   mark (`'C7H8+.'`, or a bullet / middle dot) is accepted and **checked** — on an
   even-electron formula it is a typo and raises. Formula bodies — of ions, and of
   loss strings given to `decompose_loss` — are parsed **strictly**: the whole
   string must be element tokens and every symbol must be known.
   `chemistry.parse_formula` skips what it cannot read (`'C2H4o2'` → C2H4,
   `'H2O-H2'` → H4O); here both raise.

2. **Recalibrate** (`mass_offset`, `apply_offset`). Anchors are
   `(measured_mz, ion_formula)` pairs. `offset_ppm` = the **median** anchor error
   `(measured − theoretical)/theoretical·1e6` (positive = reads high, negative
   = reads low; with the usual two anchors, precursor + reagent ion, the median
   is their mean). The theoretical m/z is `ion_mass` of the anchor's own formula
   and charge, so an anion anchor (`'^NO3-'`, 62.98540) sits two electron
   masses above the cation of the same formula. Each
   anchor's residual is the same error re-read on the corrected axis,
   `(apply_offset(measured) − theoretical)/theoretical·1e6`, so an outlier keeps
   its sign. Anchors are reported by their normalised ion formula (`'C7H8+.'` →
   `'C7H8+'`). `apply_offset(mz, offset_ppm) = mz / (1 + offset_ppm·1e-6)`.
   A **constant-ppm** model only: one or two anchors cannot fit a low-mass
   absolute term — that is what the `tol_mda` floor absorbs.

3. **Sub-formula grid per parent** (`_parent_grid`, cached for 32 parents, keyed
   by composition AND charge — one formula as a cation and as an anion is two
   grids).
   All element-wise sub-formulas (one numpy meshgrid over `0..count` per element),
   then the **fragment gate** — one vectorised function (`_gate`) that both the
   grid and `fragment_ok` call, its Senior cap read off `chemistry.seniors_cap`:
   at least one non-H atom;
   DBE ≥ −0.5 when even-electron (H3O⁺, NH4⁺ and every protonated saturated
   molecule sit at −0.5), DBE ≥ 0 when odd-electron; DBE ≤ Senior cap + 0.5
   (`seniors_cap` = `C + Si + (N + ^N)/2 + 1`). Masses are
   `ion_mass(F, charge) = neutral_mass(F) − charge·M_E`, sorted ascending. The
   **parent itself is never gated**: its composition is an input, and an ammonium
   cluster of a saturated molecule sits at ion DBE −1.5 (`C6H18^NO2+` =
   C6H14O2 + ¹⁵NH4⁺, a co-isolate of the 2026-09-10 exploratory file's 137.057
   window) yet is its own precursor peak. A grid above **5,000,000** rows
   (`_GRID_MAX_ROWS`) is refused — by `assign_fragments` when it reads the
   parents, before any peak (an empty peak list does not slip one through); a
   C30H51O15⁺ parent is ~16k rows after the gate.

4. **The window.** `tol = max(tol_ppm · mz_cal · 1e-6, tol_mda · 1e-3)` Da;
   every grid row with `|mass − mz_cal| ≤ tol` is a candidate.
   `ppm = (mz_cal − mass)/mass · 1e6`. Why the floor: an Orbitrap's low-mass error
   is an absolute offset — the 2026-09-10 file's sub-80 MS1 ions sat at −0.12 mDa
   (`assignment/masscal.py`), and after the precursor recalibration its C2H3O⁺
   fragment read −3.8 / −4.1 ppm (−0.16 / −0.18 mDa). A pure 5 ppm window is
   0.21 mDa at m/z 43; the 0.3 mDa floor makes it 7 ppm there and is inert above
   m/z 60.

5. **The loss and its parity.** `L = parent − F`, element-wise. DBE is affine in
   the counts, so **`DBE(L) = DBE(parent) − DBE(F) + 1`**: a loss between two ions
   of the same electron parity has an integer DBE (closed-shell); between ions of
   opposite parity a half-integer one (a radical). `F = parent` is the precursor
   reading (`is_precursor`, loss `''`, 0 pieces).

6. **Decompose the loss** (`decompose_loss`, memoised per `(loss, k_max)`).
   - **Necessary bound.** k closed-shell molecules of DBE ≥ 0 sum to
     `DBE(L) + k − 1`, so `k ≥ max(1, ⌈1 − DBE(L)⌉)`; one radical piece (DBE ≥ ½)
     plus closed ones gives `k ≥ max(1, ⌈1.5 − DBE(L)⌉)`. A loss whose bound
     exceeds `k_max` is rejected before any search: H4O (DBE −1) needs 2 (H2O + H2),
     H5^NO needs 2, H7^NO2 (DBE −2) needs 3, H8 (DBE −3) needs 4.
   - **Pieces.** Library formulas (`CLOSED_SHELL_LOSSES`, and for a radical loss
     exactly one of `RADICAL_LOSSES` or one unnamed radical remainder) plus **at
     most one unnamed remainder**. A closed-shell loss holds no radical piece
     and a radical loss never two: ^NO + ^NO2 (^N2O3) is no reading at any
     `k_max`, nor is ^NO + ^NO2 + ^NO3 (^N3O6) at 3 or 4. The 14N formulas do
     read, through the named N2O: N2O3 as `O2 + N2O`, N3O6 as `O2 + N2O + NO3` —
     a composition, not a claim that N2O left.
   - **Unnamed closed-shell remainder** is valid when: integer DBE ≥ 0; Senior's
     second rule — sum of valences ≥ 2 × the largest valence present (valences
     C, Si 4; N, ^N, P 3; O, S 2; H, F, Cl, Br, I 1); and **at least one H or
     halogen**. The last clause rejects bare-skeleton pieces (`C4`, `C`, `O`, `C2O`)
     — `CO`, `CO2`, `N2`, `O2`, `O2S`, `CS2`, `COS` and `N2O` are library pieces,
     so they stay possible.
     A halogen-only piece passes: `I2` and `Br2` are not in the library but read
     as one unnamed piece, which is the iodide reagent's declustering (I3⁻ → I⁻,
     an [M−H+I2]⁻ adduct → [M−H]⁻).
   - **Unnamed radical remainder**: half-integer DBE ≥ ½; sum of valences ≥
     2 × largest − 1 (CH3: 7 ≥ 7, HO: 3 ≥ 3); H or halogen.
   - **Choice** (deterministic): fewest pieces, then fewer unnamed pieces, then
     library priority (the ascending list of the pieces' library indices,
     compared lexicographically). The name lists the pieces in that order, the
     unnamed remainder last: `H5^NO → 'H2O + H3^N'`, `CH6 → 'H2 + CH4'`,
     `H4O2 → 'H2O + H2O'`, `C5H14O → 'H2O + C5H12'`.
   - **Search** (`_decompose`). By total size k = 1, 2, … `k_max`. A *frontier*
     holds every multiset of k − 1 library pieces that fits inside the loss
     (ascending library indices, so no permutations; one radical piece at most,
     and only for a radical loss), in lexicographic order. Size k reads as one
     more library piece that completes the loss exactly (all named — tried
     first), else as a frontier multiset plus one valid unnamed remainder. The
     first hit is the answer, which is exactly the choice rule above. A frontier
     multiset whose rest needs more pieces than `k_max` leaves (the same k-bound,
     applied to the rest, with the closed-shell bound once the multiset holds
     the radical piece) is dropped before it grows, so the search only ever
     holds splits that could still read; it never drops one that can — the tests
     compare it with an unpruned enumeration of every multiset. Size 1 is a
     dictionary look-up plus one remainder test.

   Why a decomposition and not ONE neutral: MS2 fragments are often
   second-generation. On the 2026-09-10 exploratory 15N-ammonium file the
   labelled-ammonium adducts decluster and then dehydrate — C11H18^NO2⁺ →
   C11H13O⁺ (loss H5^NO = H2O + H3^N), C12H22^NO2⁺ → C12H15⁺ (loss H7^NO2,
   DBE −2) — and no single molecule has those formulas. Why the H-or-halogen
   clause: on the same file 88.0215 (C3H6NS⁺) fits C7H6NS⁺ minus C4 at
   −0.5 ppm, but the 0.4 Da window also held C4H10NS2⁺ (136.0248, 13 % of the
   window signal), and C4H10NS2⁺ minus CH4S is the reading.

7. **Rank** within one (peak, parent) (`FragmentCandidate.sort_key`):
   (a) closed-shell losses before radical losses; (b) fewer loss pieces;
   (c) fewer unnamed pieces; (d) smaller |ppm|; (e) the formula string. Every
   candidate is reported (`rank`, `n_candidates`) — ties are ordered, never hidden.
   With `allow_radical=False` radical losses are not readings at all.

8. **The long table** (`assign_fragments`). One row per (peak, parent,
   candidate), ordered by input peak, then parent order as given, then rank.
   `n_parents` = how many parents have ≥ 1 candidate for the peak. A peak no
   parent explains gets **one** row — every input peak is in the output. That
   row keeps its `peak` / `mz` / `mz_cal` / `intensity` and carries `''` in every
   other text column (`parent`, `parent_ion`, `formula`, `parity`, `loss`,
   `loss_name`), 0 in `loss_pieces`, `unnamed_pieces`, `rank`, `n_candidates`
   and `n_parents`, `False` in `radical_loss` and `is_precursor`, and NaN in
   `dbe` and `ppm` — so a filter on a flag or a piece count never picks it up
   as a reading. `best_readings(df)` keeps the rank-1 rows plus the unexplained
   rows.

---

## 4. Constants reference

All in `peaky/ms2/subformula.py` unless noted.

| constant | value | role |
| --- | --- | --- |
| `tol_ppm` (default) | 5.0 ppm | relative window |
| `tol_mda` (default) | 0.3 mDa | absolute floor of the window (low-mass offset) |
| `k_max` (default) | 3 | most molecules a loss may split into |
| `K_MAX_LIMIT` | 6 | `k_max` above this raises. The search holds only multisets that fit inside the loss and stops at the first size that reads; the cap bounds a pathological loss (random large halogenated losses at `k_max` 6: a few ms, about 0.1 s and < 20 MB at worst, once — results are cached; without the pruning the deep ones take 10–50× longer). 6 still reads ^NH3 + 5 H2O (H13^NO5, DBE −5) |
| `allow_radical` (default) | `True` | radical (parity-flipping) losses are readings, ranked after closed-shell ones |
| fragment gate | EE DBE ≥ −0.5, OE DBE ≥ 0, DBE ≤ Senior cap + 0.5, ≥ 1 non-H atom | `fragment_ok`; the parent is exempt |
| `_GRID_MAX_ROWS` | 5,000,000 | largest sub-formula grid built per parent |
| `_parent_grid` cache | 32 parents | `functools.lru_cache`, per (composition, charge) |
| `_decompose` cache | 65,536 losses | `functools.lru_cache`, per `(loss, k_max)` |
| `CLOSED_SHELL_LOSSES` | 43 formulas | see §5 |
| `RADICAL_LOSSES` | 16 formulas | see §5 |
| `chemistry.ion_mass` | `neutral_mass − charge·M_E` | charge must be +1 or −1 |

---

## 5. The loss libraries (priority order)

The library only **names** pieces and breaks ties between decompositions of equal
size; a loss that is not a library formula is still accepted as one unnamed piece
when it passes the remainder rules. Names are formulas, never trivial names and
never a structure: `C3H6O` is a composition, whatever the reader thinks left.

**Closed-shell** (`CLOSED_SHELL_LOSSES`): H2O, H3N, H3^N, CO, CO2, H2, CH4, C2H4,
CH2O, CH4O, CH2O2, C2H2O, C2H4O2, CHN, CHNO, CH4N2O, H2O2, HNO2, H^NO2, HNO3,
H^NO3, H2S, CH4S, O2S, O3S, HF, HCl, HBr, HI, N2, O2, C2H2, C3H6, C4H8, C3H6O,
C2H4O, C5H8, then three added for neutrals peaky's own channels lose often:
C2H6O (ethyl esters / ethers), H2O4S (organosulfates), C2H6OSi
(dimethylsilanone, the ring-contraction loss of cyclic methylsiloxanes); and
last, three real losses with no H or halogen, which the remainder rule would
refuse as unnamed pieces: CS2 and COS (dithiocarbamates, thiocarbamates) and
N2O.

**Radical** (`RADICAL_LOSSES`): H, CH3, HO, HO2, NO, ^NO, NO2, ^NO2, NO3, ^NO3,
CHO, CH3O, C2H5, Cl, Br, I.

---

## 6. Metrics, defined

- **Electron parity of an ION** (`ion_parity`) — even-electron (`EE`) iff the
  ion formula's DBE is **half-integer**: C10H17⁺ 2.5 EE, C10H17O5⁻ 2.5 EE,
  C7H8⁺· 4.0 OE. This is the *opposite* reading of `chemistry.odd_electron()`,
  which is defined on NEUTRALS (half-integer = radical); the module does not
  reuse it.
- **`DBE(L) = DBE(P) − DBE(F) + 1`** — the loss's DBE from its two ions.
- **k_min** — `max(1, ⌈1 − DBE(L)⌉)` closed-shell, `max(1, ⌈1.5 − DBE(L)⌉)` radical.
- **window** — `max(tol_ppm·mz_cal·1e-6, tol_mda·1e-3)` Da.
- **ppm** — `(mz_cal − theoretical)/theoretical · 1e6`.
- **offset_ppm** — median anchor error; `mz_cal = mz / (1 + offset_ppm·1e-6)`.

---

## 7. Outputs

`assign_fragments` returns a DataFrame with fixed column dtypes
(`OUTPUT_COLUMNS`), whatever the content:

| column | dtype | meaning |
| --- | --- | --- |
| `peak` | int64 | 0-based row position of the peak in the input |
| `mz` / `mz_cal` | float64 | input m/z as given / recalibrated m/z every candidate is judged on (on unexplained rows too) |
| `intensity` | float64 | from the input `intensity` column, else NaN |
| `parent` / `parent_ion` | str | the parent's label / its normalised ion formula (`'C7H8+.'` → `'C7H8+'`); `''` when unexplained |
| `formula` | str | fragment ion formula, e.g. `'C6H9+'` |
| `parity` | str | `'EE'` / `'OE'` of the fragment |
| `dbe` | float64 | DBE of the fragment ion formula (NaN when unexplained) |
| `ppm` | float64 | mass error of the reading (NaN when unexplained) |
| `loss` | str | parent − fragment, Hill; `''` for the precursor |
| `loss_pieces` / `unnamed_pieces` | int64 | molecules in the decomposition / how many unnamed (0 or 1) |
| `loss_name` | str | the pieces joined by `' + '` |
| `radical_loss` | bool | half-integer-DBE loss (fragment parity ≠ parent parity) |
| `is_precursor` | bool | the fragment is the parent itself |
| `rank` | int64 | 1 = best within (peak, parent); 0 = unexplained row |
| `n_candidates` | int64 | readings within (peak, parent) |
| `n_parents` | int64 | parents with ≥ 1 reading for this peak; > 1 = composition-ambiguous |

An unexplained row's full set of values is spelled out in §3, step 8.

`fragment_candidates` returns the same readings for one (peak, parent) as
`FragmentCandidate` objects (plus `mz_theo` and the `pieces` tuple);
`decompose_loss` returns a `LossDecomposition` (`pieces`, `unnamed`, `radical`,
`k`, `name`) or `None`; `mass_offset` returns a `MassOffset` (`offset_ppm`,
`anchors`, `to_frame()`).

---

## 8. Properties, invariants & gotchas

- **Composition, not attribution.** `n_parents` > 1 means two window ions can
  both make the peak; nothing here prefers one. On the 2026-09-11 monoterpene
  ammonium-adduct window, 137 / 81 / 95 / 109 are sub-formulas of both
  C10H20^N⁺ and the co-isolated C10H19O⁺.
- **Fewest pieces wins before named pieces.** A loss that is valid as ONE unnamed
  molecule is never split into two named ones: C11H15O2⁺ → C2H3O⁺ reads
  `C9H12O`, not `H2O + C9H10`. A named split appears only where the formula
  cannot be one molecule (negative DBE) or the remainder fails its rules.
- **Piece order is library order**, not mass order: `CH6 → 'H2 + CH4'`.
- **Closed-shell vs radical collisions are rare at Orbitrap tolerance.** Two
  readings of opposite parity at one nominal mass differ by an odd number of
  ¹⁴N atoms. The cheapest trades are ^N for NH (10.8 mDa — possible only when
  the parent holds both ¹⁴N and ¹⁵N, e.g. the ^NH4⁺ adduct of an amide:
  C4H13N^NO⁺ → C3H7^N⁺ by a radical loss has the EE twin C3H8N⁺) and N for CH2
  (12.6 mDa); closer ones need multi-atom trades (^N2 for NO, 2.2 mDa, again
  mixed-label parents only; CH4N3 for C3H6O, 1.3 mDa). CH3's only same-nominal
  closed-shell rival, NH, fails Senior's rule. Ranking rule (a) decides mostly
  in wide windows.
- **The parent is never gated; fragments are.** A fragment below ion DBE −0.5 —
  an ammonium or hydronium cluster of a saturated molecule (¹⁵NH4⁺·H2O,
  DBE −1.5) — is never read, even from a cluster parent. `fragment_ok` is the
  fragment test: do not pre-screen window ions with it (C6H18^NO2⁺ fails it and
  is a real window ion).
- **Long dehydration chains need an explicit `k_max`.** A hexose's [M+^NH4]⁺
  (C6H16^NO6⁺) losing ^NH3 and then water step by step reads at the default 3 up
  to −^NH3 −2 H2O; −3 H2O is 4 molecules, −4 H2O (C6H5O2⁺, loss H11^NO4,
  DBE −4) is 5 and −5 H2O is 6 — no library can do them in fewer. Pass
  `k_max=5` or `6`. The default stays 3 because every extra piece admits more
  losses, and so more wrong formulas, at every peak. `ms2 analyze` reads the
  fourth piece only on a ladder (§11.2): −^NH3 −3 H2O when −^NH3 −2 H2O (or
  −3 H2O) is read at some energy of the same window.
- **`'^N'` is its own element.** A ¹⁵N-labelled reagent's losses (`H3^N`,
  `H^NO2`, `H^NO3`) are library pieces distinct from the ¹⁴N ones.
- **The window is judged on `mz_cal`.** Pass `offset_ppm` from `mass_offset`;
  a 12 ppm axis error left uncorrected is outside the default window.
- **Singly charged only**: `ion_mass`, `parse_ion` and `format_ion` refuse any
  other charge.
- **Deterministic.** Same input, same table — across runs and with cold caches.

---

## 9. Code map

| function | role |
| --- | --- |
| `parse_ion` / `format_ion` | ion formula ↔ (counts, charge); exactly one sign, checked radical mark |
| `ion_parity` | `'EE'` / `'OE'` of an ion formula (half-integer DBE = EE) |
| `fragment_ok` | the fragment gate (DBE floor by parity, Senior cap + 0.5, ≥ 1 non-H atom) |
| `decompose_loss` | a loss → `LossDecomposition` (≤ k_max molecules) or `None` |
| `fragment_candidates` | one (parent, m/z) → ranked `FragmentCandidate` list |
| `mass_offset` / `apply_offset` | constant-ppm offset from anchors; the recalibrated m/z |
| `assign_fragments` | peaks × window ions → the long table |
| `best_readings` | rank-1 rows + unexplained rows |
| `chemistry.ion_mass` | exact m/z of a singly-charged ion from its own formula |

---

## 10. The Mascope source — `peaky ms2 list | extract`

**Code:** `peaky/ms2/sources.py` — pure table builders on the server's payloads
(`parse_activation`, `spectra_table`, `parents_table`, `window_table`,
`inventory_row`, `hidden_tail_s`, `locate_precursor`) and two thin orchestrators
(`list_sample`, `extract_sample`) that call only the fetchers in
`peaky/io/io_mascope.py` (`fetch_ms2_summary`, `fetch_ms2_spectra`,
`fetch_ms2_ms1`, `fetch_sample_record`), which stays the one module that talks to
Mascope. Energy-resolved spectra need **Mascope ≥ 1.8** (mascope-sdk ≥ 2026.9.15,
the first release with `by_activation` and the summary's groups); an older server
is read and labelled energy-averaged. Read-only: nothing is written to the server.

### 10.1 The payload contract

| call (SDK) | what it answers | what the source reads |
| --- | --- | --- |
| `samples.ms2(id).get_summary()` | `parent_peaks`, `groups` — one per (parent, activation): `parent_peak_mz`, `activation` (`'hcd40.00'` = NCE 40), `hcd_energy` (VOLTS, a list, usually one value), `scan_count`, `t_min`, `t_max` — plus `hcd_energy_map`, `isolation_width`, `ms1_scan_count`, `ms2_scan_count`, `parent_peak_tolerance` | scan counts, times, volts, the isolation width, the MS2 count |
| `get_averaged_centroids(by_activation=True, noise_threshold=3)` (`ms2 list`, which reads only the keys: the SDK's default 10) | one averaged centroid spectrum per group, keyed `'<parent>@<activation>'` (`'137.1321@hcd1.00'`): `parent_peak_mz`, `activation`, `mz`, `intensity`, `resolution`, `signal_to_noise` | the spectra |
| `get_ms1_centroids(ppm=1)` | `mz`, `intensity`, `resolution`, `signal_to_noise` over the whole MS1 range | the isolation-window inventory |
| the sample's listing row / `samples.get(id)` | name, `datetime_utc`, polarity, `instrument_type`, `t0`, `t1`, `length` | the hidden tail; a TOF sample is skipped |

`get_timeseries` is read only by `--per-scan` (§12), for the scan times that name
each group's scans: breakdown curves are built from the per-energy spectra (§11),
one point per (parent, activation) group. Parents are always the
**server's keys** (`parent_mz`), never re-derived.

### 10.2 Fallbacks

| the server answers | the source | `status` |
| --- | --- | --- |
| an MS1-only sample: every summary key present, `ms2_scan_count` 0 | asks for nothing else; empty tables with their dtypes | `no MS2` |
| … and the file runs past its last ingested scan | same, with the tail and the remedy in `note` | `MS2 may be hidden` |
| `get_summary` HTTP 400 "Multiple isolation widths found for MS2 scans" (a known server defect; the centroids still work) | `io_mascope.MS2SummaryRefused` (never retried, the URL dropped); spectra from the centroid keys, parents and activations from those keys; `scan_count`, `t_min`, `t_max`, `hcd_v`, `isolation_width` NaN; the reason in every note; the window needs `--iso-width` | `summary refused: <reason>` |
| a server older than 1.8: `by_activation` ignored, per-parent spectra with no `activation` / `parent_peak_mz` fields; a summary without `groups`, whose `hcd_energy_map` holds the scan-weighted MEAN volt of each parent (one per step of a stepped energy: 5.16 V for a parent measured at 4.05–7.41 V) | spectra keyed by parent, `activation` `''`, `nce` NaN, `hcd_v` the server's mean (`hcd_v_values` the step means, when several) with the note that it is a mean and not a voltage applied, `energy_resolved` False, and `server returns energy-averaged MS2 only (Mascope < 1.8): per-energy spectra and breakdown curves need a newer server`; the inventory's `hcd_v_min` / `hcd_v_max` stay NaN (a range of means is not the volts applied) | `energy-averaged server` |
| a current server asked without `by_activation` (not what `peaky ms2` asks; the builders take it): one spectrum per parent, `activation` `''` | one row per parent, `nce` and `hcd_v` NaN, `hcd_v_values` the volts its groups span, `scan_count` / `t_min` / `t_max` over those groups; no per-group rows; `energy_resolved` False and said in the note | `ok` |
| summary groups and centroid keys that disagree | a group no spectrum is keyed to keeps a row (`n_peaks` 0, the group's numbers, note "the summary lists this group but no centroid spectrum is keyed to it …") but is not counted in `n_spectra` / `nce`; a spectrum no group matches keeps its peaks, its summary-only fields NaN (note "no summary group matches …") | `ok` |
| no summary at all (`None`) | the spectra if the centroids answer | `no summary` |
| `ms2 extract` only: no summary, and the centroid call answers HTTP 400 "… No scans found matching the specified filters …" (no MS2 scan in the sample's ingested window: with no summary to count on, the centroid call is made, and the server answers that rather than an empty answer). `ms2 list` asks no centroids here and reads `no summary` | empty tables, no MS1 request, `NO_MS2_SCANS_NOTE`; with a hidden tail the status is `MS2 may be hidden` and the tail's note says why. After a REFUSED summary, with a summary that counts MS2 scans, or for any other error, it is raised as below | `no MS2` |
| a summary that counts MS2 scans but groups none of them (the server groups a scan by parent and activation, and serves a spectrum for it, only when it can read the scan's event `ms2 <m/z>@<activation>`) | no spectrum can exist; the note gives the count and says no noise threshold brings one back | `MS2 unreadable` |
| … groups some of them: the groups' `scan_count`s add up to less than `ms2_scan_count` | the note gives how many scans are in no group (and so in no spectrum) | `ok` |
| groups, but an empty centroid answer | the note says the centroid answer holds no spectrum for the summary's groups (the S/N floor empties a spectrum's lists, never its key) | `ok` |
| a TOF sample (its listing row's / record's `instrument_type`; the server reads MS2 from Orbitrap raw files only and answers anything else HTTP 500) | asks nothing after the record; empty tables | `not an Orbitrap sample` |
| the MS1 centroid call fails for this sample (`is_sample_error`: its HTTP 400 "No scans found matching the specified filters" for a sample window that holds only the MS2 block, its own 504 "MS1 centroid extraction timed out after 120s") | the MS2 tables already read are kept and written; the window is empty, and a note gives the server's reason (`WINDOW_FAILED_NOTE`); `--no-window` skips that request. A server or connection failure on it stops the loop as below | unchanged (e.g. `ok`) |
| one sample's failure (`io_mascope.is_sample_error`): an HTTP 400 / 404 / 409 / 422 / 500; the API's own 403 (`is_api_403`: its JSON answer, not a proxy's page — how the server answers an unknown or stale id); the MS2 endpoints' own 504 "… timed out after 120s for sample" (`is_ms2_compute_timeout`); an unreadable payload | peaky adds no retry of its own — except that the sample record (`samples.get`) retries a 500 with the back-off — but the SDK's transport has already re-sent a 504 up to four times (2 / 4 / 8 s), so a compute timeout costs about four 120-s server computations; that sample's row / console line, its message without the server's address, and a hint chosen by the HTTP status, never by the message's digits (a stale id: re-list the ids); the other samples go on, the `--out` CSV is written, and the command exits 1 naming the failed samples | `error: <exception type>` |
| a 401 (raised at once, no back-off); a proxy's WAF 403, a 429, a gateway 502 / 503 / 504 / 521 / 522 (each retried with peaky's back-off, 2 / 4 / 8 s — and a 429 / 502 / 503 / 504 already re-sent up to four times by the SDK's transport underneath); or a server that cannot be reached (the SDK's `MascopeConnectionError` / `MascopeTimeoutError`, raised after the SDK's own four attempts and not retried again) | the loop stops there (asking on only extends a block, or waits on a server that is down); what was read is written; exit 1 says how many samples were not asked | `error: <exception type>` |
| anything else | — | `ok` |

### 10.3 Rules and constants

All in `peaky/ms2/sources.py`.

| constant | value | role |
| --- | --- | --- |
| `DEFAULT_NOISE_THRESHOLD` (`--noise-threshold`) | 3 | the server's S/N floor for the averaged MS2 centroids `ms2 extract` asks for, below the SDK's default of 10: on the 2026-09-11 monoterpene standard A the [M+H]⁺ window's spectra hold 19–21 peaks at S/N 10 and 25–30 at S/N 3, and ions the raw-file breakdown tracked at 300+ counts per scan (C4H9⁺ at NCE 5, C6H7⁺ at NCE 20) are only in the S/N 3 ones; at 0 the lists fill with noise centroids |
| `ms1_ppm` (`--ms1-ppm`) | 1 | MS1 centroid binning for the window inventory |
| `with_window` (`--no-window` turns it off) | True | ask for the MS1 centroids and build the window inventory |
| `WINDOW_PAD_DA` | 0.005 Da | the inventory reaches ±(isolation_width/2 + pad) around the parent key |
| `PRECURSOR_SEARCH_PPM` / `_MDA` | 10 ppm / 5 mDa | where the precursor is searched for around the key: `max(ppm·m/z·1e-6, mDa·1e-3)` |
| `PRECURSOR_MIN_SHARE` | 0.10 | a candidate holds ≥ 10 % of the strongest peak in that search window |
| `PRECURSOR_TRACK_PPM` / `_MDA` | 5 ppm / 0.5 mDa | how close a spectrum's peak must be to the located precursor |
| `PRECURSOR_LOBE_RATIO` / `_FWHM` | 50× / 5 line widths | only for a raw file's spectra whose m/z axis offset is unknown (§12.4): a candidate this close to a peak this much brighter just outside the search window is that line's side lobe, refused at that energy |
| `KEY_OFFSET_NOTE_MDA` | 1.0 mDa | a precursor this far from its key is noted |
| `HIDDEN_TAIL_MIN_S` | 0.5 s | `length − t1` above it is reported |

**The precursor** (`locate_precursor`): among the peaks within the search window
of the key that hold ≥ `PRECURSOR_MIN_SHARE` of the strongest one, the nearest to
the key (the stronger on a tie). It is located **once per parent**, in its
lowest-NCE spectrum that holds a candidate, and each spectrum's `precursor_mz` /
`precursor_intensity` is its peak within the tracking tolerance of that m/z — NaN
where there is none (spent, or under the noise threshold), never a neighbour. Why
both clauses: a key typed with two decimals (`137.13` on the 2026-09-11
monoterpene standard B) sits 2.3 mDa below its C10H17⁺ precursor, with < 1 %
satellites at −1.3 / −2.2 mDa NEARER to it — nearest-alone picks a satellite; and
at NCE 5 in the 2026-09-10 file's 137.0572 window the co-isolate 2.2 mDa above the
precursor is the stronger peak (19 % vs 13 %) — strongest-alone picks the
co-isolate. The keys of the other 41 spectra in these files sat 0.02–0.35 mDa from
their precursors, a precursor wanders ≤ 0.12 mDa across its energies, and the
nearest other peak sat 1.9 mDa from one. The same rule marks `is_parent` in the
MS1 window (a `--file` folder marks the MS1 centroid at the precursor its MS2
spectra located, §12.4; with its axis offset measured, a finely typed key there is
searched within the tracking tolerance first). Two parents whose located
precursors coincide are said on both rows. **The side-lobe guard** (`lobe_guard`) is for one case only: a raw
file's own spectra (`ms2 extract --file`) whose m/z axis offset could not be
measured (§12.4, "The m/z axis"). An Orbitrap peak list carries weak centroids a
few line widths beside a much brighter line (`chem.sidelobes`); a precursor whose
axis sits off its key has its line beyond the window and only its 0.1–0.7 % upper
lobe inside. With the guard on, a winning candidate within `PRECURSOR_LOBE_FWHM`
line widths of a peak `PRECURSOR_LOBE_RATIO` times brighter just OUTSIDE the
search window is that line's lobe: refused at that energy and the next energy
tried (a lobe keeps its share of its line at every energy); a parent refused at
every energy is **ambiguous**, not located, and its rows say so ("precursor
ambiguous, not located: …"). It is off on the server's spectra (calibrated, keyed
by the isolation centre) and on a file whose offset is measured: there a weak
precursor on its key beside a 50× brighter co-isolate 5 mDa – 5 widths away is
the precursor (a server-style C10H15O4⁺ window with a co-isolate 8 mDa above at
100×: located, F-E50 NCE 7.43). Without the peaks' resolutions the guard is off.

### 10.4 Outputs

`peaky ms2 extract --sample-id ID [ID ...]` writes, per sample, to
`<out-dir>/ms2/<sample id>/` (`--out-dir`, else `$PEAKY_OUTPUT_DIR`, else
`~/peaky-output`; a re-run overwrites the folder):

| file | content |
| --- | --- |
| `ms2_spectra.parquet` + `.csv` | the long spectra table |
| `ms2_parents.csv` | one row per spectrum |
| `ms2_window.csv` | the MS1 isolation-window inventory (empty, with its header, when skipped) |
| `ms2_manifest.json` | sample, `status`, `energy_resolved`, notes, the server's refusal reason, scan counts, `hidden_tail_s`, the parameters (incl. the isolation width used and where it came from), counts (`spectra` = the spectra the centroid answer holds; `parents_rows` = rows of `ms2_parents.csv`), each file's sha1, and the code identity (package / module versions, git, deps — `reporting.provenance`). No credentials, no server URL |

A sample that fails writes nothing (its console line says why) and the others
go on; the command then exits 1 naming it. A failed MS1 call fails only the
window: the sample's MS2 tables are written, `ms2_window.csv` is empty and the
manifest's notes give the server's reason. `--no-window` skips the window and
its MS1 request.

**Spectra** (`SPECTRA_COLUMNS`, one row per centroid; ordered by parent, NCE —
none last — activation, then m/z): `sample_id`, `spectrum` (the server's key),
`parent_mz` (the server's), `activation`, `kind` (`'hcd'`), `nce`, `hcd_v` (mean
of the group's volts; NaN when unknown), `hcd_v_values` (`;`-joined, only when
there is more than one value; for a current server's energy-averaged spectrum
every volt its groups span), `scan_count`, `t_min`, `t_max` (s),
`energy_resolved`, `mz`,
`intensity`, `resolution`, `signal_to_noise`. From a server older than 1.8,
`hcd_v` of the energy-averaged spectrum is the server's mean over all of the
parent's scans (the note says so). One spectrum,
`spectra[spectra.spectrum == key]`, is a peak list `assign_fragments` reads as it
is; `mass_offset([(precursor_mz, ion)])` from the parents table recalibrates it.

**Parents** (`PARENTS_COLUMNS`, one row per spectrum, plus one per summary group
the centroid answer holds no spectrum for): the spectrum's metadata as above,
`isolation_width` (the server's; NaN when refused), `n_peaks`, `tic` (summed
intensity), `precursor_mz`, `precursor_intensity`, `energy_resolved`, `note`. No
survival, E50 or other breakdown numbers — those are `ms2 analyze`'s (§11).

**Window** (`WINDOW_COLUMNS`, per parent then m/z): `sample_id`, `parent_mz`,
`mz`, `offset_da` (from the key), `intensity`, `share` (of the window's summed
intensity), `resolution`, `signal_to_noise`, `is_parent`. Every centroid is kept
whatever its S/N — the MS1 list carries noise centroids at S/N < 1; filter on
`signal_to_noise`. Overlapping windows list a centroid under each parent (the
2026-09-11 standard A's 155.1429 and 155.1559 windows hold each other's
precursor: 155.1559 is 50 % of the 155.1429 window).

`peaky ms2 list (--batch B [--dataset D] | --sample-id ID [ID ...]) [--out CSV]`
prints one inventory row per sample (`INVENTORY_COLUMNS`, the CSV carries all of
them): `sample_id`, `name`, `datetime_utc`, `polarity`, `ms1_scans`,
`ms2_scans`, `n_parents`, `parents` (the server's keys, `;`-joined), `n_spectra`,
`nce` (`;`-joined, of the spectra the row counts), `hcd_v_min` / `hcd_v_max`
(the summary groups' volts; NaN from a server older than 1.8), `isolation_width`,
`energy_resolved`, `hidden_tail_s`, `status`, `note`. It sends one summary
request per sample, sequentially, plus the sample's record with `--sample-id`
(a batch listing supplies the rows) and the centroids of a sample whose summary
is refused (so its row still names parents and NCEs); a TOF sample costs none.
It refuses more than `--max-samples` (100) samples — a long burst is what trips
the server's WAF. A sample that fails gets an `error: <type>` row and the others
go on (§10.2); `--out` is always written.

### 10.5 Gotchas

- **Count, never shape.** An MS1-only sample answers a full scaffold (every key,
  empty lists); "no MS2" is `ms2_scan_count == 0`, and a scaffold-shaped answer
  that counts scans is not "no MS2". With no summary at all, `ms2 extract` reads
  it off the centroid call's HTTP 400 "No scans found" (never with a hidden tail,
  which reads `MS2 may be hidden`).
- **The parent is the server's key**, the method's isolation centre as typed, not
  the measured precursor (`precursor_mz`). Query the server by the key: its
  time-series endpoint returns nothing for 137.1321 on a sample keyed 137.13.
- **Hidden tail.** An ingest that predates the server's MS2-window fix stops at
  the MS1 block; MS2 scans recorded after it are missing until the sample is
  reprocessed on the server. `hidden_tail_s = length − t1` when > 0.5 s (the
  2026-09-11 standard B: 210.5 s); it is reported, never read as "no MS2".
- **NCE and volts are both reported and never converted.** The volts one NCE
  applies depend on the precursor m/z: NCE 1 is 4.05 V at m/z 136 and 4.28 V at
  m/z 315 in the 2026-09-10 file.
- **An older server's volts are means.** Before 1.8 the summary's
  `hcd_energy_map` holds each parent's mean volt over all its scans, a number
  the instrument never applied; it is reported as `hcd_v` with a note, never
  as a range.
- **`noise_threshold` is applied by the server** (S/N; `ms2 extract` asks for 3,
  the SDK's default is 10); it is not a peaky filter, and the manifest records
  the value used. A weak fragment missing from a spectrum may be under the floor:
  re-extract lower before calling it absent.
- **Energy-averaged spectra blend every step** of a stepped-energy acquisition;
  they are labelled (`energy_resolved` False), never passed off as one energy.
- **The window needs an isolation width.** A refused summary has none:
  `--iso-width W` supplies it (and, when given, replaces the server's).
- **The window is optional; the spectra are not.** The MS1 centroids feed the
  window only, so a sample whose MS1 call fails keeps its MS2 tables (§10.2).
- **Retries go by the HTTP status, never by the error's text.** The SDK
  appends the URL, whose sample id can spell `403` or `500`; peaky adds no
  retry to a sample's own failure (§10.2; the sample record retries a 500) and
  retries a WAF block / rate limit / gateway error with back-off
  (`_MS2_RETRY_STATUS`). A 403 is a WAF block only when it is a proxy's page:
  the API's own 403 means this id (unknown, stale, or not the token's to
  read). The console hint goes by the status too: a 500's `(ref: …)` can spell
  `403`.
- **The SDK retries underneath.** Its transport re-sends a 429 / 502 / 503 /
  504 and a connection failure up to four times (2 / 4 / 8 s) before peaky
  sees it, the MS2 endpoints' own compute-timeout 504 included: that answer
  costs about four 120-s computations on the server (≈ 8 min), which peaky
  cannot shorten without bypassing the SDK. A gateway error peaky retries
  can reach the server up to 16 times.
- **No server address in any output.** An error's message is kept without its
  URL and without the host urllib3 names bare ("Failed to resolve '…'",
  "Connection to … timed out") or the configured `MASCOPE_URL`'s
  (`io_mascope.server_error_text`).

### 10.6 Code map

| function | role |
| --- | --- |
| `io_mascope.fetch_ms2_summary` | `get_summary`; the isolation-width 400 → `MS2SummaryRefused(reason)` |
| `io_mascope.fetch_ms2_spectra` / `fetch_ms2_ms1` / `fetch_sample_record` | the centroids (`by_activation`, `noise_threshold`), the MS1 centroids, the sample record |
| `io_mascope.server_error_text` / `is_sample_error` / `is_api_403` / `is_ms2_compute_timeout` | an error's message without the server's address; one sample's failure vs the server's or the connection's; the API's own 403 vs a proxy's; the server's compute timeout vs a gateway's |
| `parse_activation` / `spectrum_key` / `is_energy_averaged` | `'hcd40.00'` → `('hcd', 40.0)`; the server's key; the < 1.8 answer |
| `spectra_table` / `parents_table` / `window_table` | the three tables |
| `locate_precursor` | the precursor rule (§10.3) |
| `inventory_row` / `inventory_frame` / `hidden_tail_s` | the `ms2 list` row |
| `skipped_row` / `error_row` / `is_tof` | the row of a TOF sample (asked nothing) and of a sample that failed |
| `list_sample` / `extract_sample` | the orchestrators (client in, rows / `SampleMS2` out) |
| `manifest` | the `ms2_manifest.json` record |

---

## 11. Across energies — `peaky ms2 analyze`

**Code:** `peaky/ms2/readings.py` (`read_window`, `ladder_gate`, `gate_mode`),
`peaky/ms2/attribute.py` (`peak_roles`, `merge_fragment_peaks`, `fragment_mz`,
`fragment_compositions`, `candidates_across`, `shared_presence`,
`intensity_matrices`, `intact_test`, `share_weights`,
`attribute_fragments`, `branching`), `peaky/ms2/breakdown.py`
(`survival`, `e50`, `e_cm`, `onset`, `ion_budget`) — pure, no I/O — and
`peaky/ms2/analyze.py` (window ions, offsets, the orchestration, the figures and
the manifest), and `peaky/ms2/replicates.py` (the breakdown re-read scan by
scan, the E50 bootstrap: §11.11). **Offline:** it reads one `ms2 extract`
sample folder and asks Mascope nothing; it needs no credentials. A folder with
the per-scan tables (`ms2 extract --per-scan` or `--file`, §12) also gets the
replicate statistics and an interval for every E50 (§11.11); a peak that is an
Orbitrap harmonic ghost of a brighter line is read as no formula (§11.12).

```
ms2_spectra.parquet, ms2_parents.csv, ms2_window.csv, ms2_manifest.json   + the named ions
        │  per parent key: window ions = named ions inside ±(width/2 + 5 mDa);
        ▼  target = the named ion at the located precursor (±10 ppm)
   offset_ppm = mass_offset([(precursor_mz at the lowest located NCE, target)])
        │  every spectrum of the window, one axis
        ▼
   assign_fragments(k_max 4) per energy → ladder gate → ms2_readings.csv
        │  per (spectrum, peak): precursor of X | fragment (candidates) | unexplained
        │  fragment peaks merged across energies (m/z within tolerance, or one formula)
        ▼
   I_X(E), I_fragment(E), TIC(E) → attribution (composition, intact test, guard, weights)
        ▼
   F, F_lo, S per ion and energy → E50 (NCE, V, E_cm) → ms2_breakdown.csv, ms2_e50.csv
   onsets, calls, weights → ms2_fragments.csv;  branching → ms2_branching.csv;  figures
```

### 11.1 Window ions, target, recalibration

- **Named, never guessed.** `--ion MZ=FORMULA` (repeatable) or `--ions CSV` (a
  `formula` column, `mz` optional). The typed m/z must lie within max(10 ppm,
  5 mDa) of the formula's exact m/z (`ION_CHECK_*`, the precursor search window
  of `ms2 extract`): a key typed with two decimals is fine, a formula from
  another window is refused as a typo — in a CSV row as on the command line
  (the row is named). `--ion` and `--ions` add up. A formula named twice
  (also once by each), or both polarities, is refused.
- **Window ions** of a parent key: the named ions whose exact m/z lies within
  `isolation_width/2 + WINDOW_PAD_DA` of the key. The width: `--iso-width`, else
  the one `ms2 extract` built its window inventory with (its manifest), else the
  server's (the parents table). Unknown (a refused summary, no `--iso-width`):
  only the target is read, and that is said. An `--iso-width` wider than the
  extract's reaches co-isolates whose MS1 region `ms2 extract` never
  inventoried (`ms2_window.csv` spans ±(its width/2 + its pad) of each key,
  `inventory_half_width`): their MS1 share is **unknown**, not 0 (§11.5), and
  the run says so.
- **Target:** the named ion nearest the precursor `ms2 extract` located (the
  parents table's `precursor_mz` at the lowest NCE where it was located),
  within max(`TARGET_MATCH_PPM` 10 ppm, `tol_mda`). None: the window is
  **skipped**, with the nearest named ion in the note — how far, in mDa and ppm,
  and, when that is within `sources.AXIS_SEARCH_PPM` (50 ppm, the reach `ms2
  extract` reads an axis offset over), the `--mass-offset-ppm` that would match
  it; a named ion farther away is taken for another ion, and the note just says
  to name this one. Measured: the located
  precursors sat within 1.8 ppm of their formulas (2026-09-10: −0.1 to +0.7 ppm
  over seven parents; 2026-09-11: −1.1 to −1.7 ppm), and the nearest other
  named ion of a window 16 ppm away.
- **The folder's m/z axis** (`folder_axis_offset`): a folder whose spectra are
  a raw file's own centroids (`ms2 extract --file`) can sit tens of ppm off its
  formulas — beyond `TARGET_MATCH_PPM`. Its manifest records the offset the
  extract measured (`parameters.ms2_axis_offset`, §12.4), and the target match
  and the MS1 shares (§11.5) read the located precursor and the MS1 window
  through it (m/z / (1 + offset)); `--mass-offset-ppm` replaces it (a server
  folder records none: the server calibrates its spectra). The window's own
  recalibration then follows as on any folder.
- **Aliases are not both window ions** (`drop_aliases`). Two named ions whose
  exact m/z lie within the reading tolerance of each other (max(`tol_ppm`,
  `tol_mda`) at their m/z: the CHO / N^N or CO3 / ^NO3 aliases of a tied
  ledger row) read the same precursor peak and every fragment both can make;
  the window cannot tell them apart. Keeping both gave the peak to one and
  left the other a phantom with I = 0 (F 0) that still took a share of every
  fragment — on the 2026-09-10 file, naming C10H14NO^N⁺ (0.44 mDa above
  C11H15O2⁺) moved C11H15O2⁺'s E50 from 21.1 to 29.6 (to 23.3 with the MS1
  shares one-to-one, §11.5). The target is kept;
  between two co-isolates, the one named first; the other is left out
  (`left_out` in the manifest, and a note).
- **Recalibration:** one constant-ppm offset per window, `mass_offset` on the
  target's located precursor, applied to all the window's spectra (§3 step 2);
  the target's precursor then reads 0 ppm in that spectrum. Recorded per window
  in the manifest. The anchor is an **energy-resolved** spectrum's whenever one
  located the precursor: an energy-averaged spectrum of the same parent sorts
  after every NCE (on standard A's 137.1321 window its precursor would give
  −1.2138 ppm instead of −1.0651, and every reading would move; tested).

### 11.2 Readings across energies — and the ladder gate on a fourth piece

Each spectrum of the window is read by `assign_fragments` against the window
ions (target first, then the co-isolates by m/z), and stacked: one long table,
`ms2_readings.csv` (`READINGS_COLUMNS`: `spectrum`, `parent_mz`, `nce`, `hcd_v`,
every `OUTPUT_COLUMNS` column, `ladder_support`, `harmonic`). A harmonic ghost
of a brighter line of the spectrum (§11.12) is read as no formula before the
gate — one unexplained-like row whose `harmonic` names the line — so it is
neither a fragment nor a ladder step.

**The ladder gate (default).** The window is read at `k_max` 4, and a reading
whose loss needs a **fourth** piece is kept only when the **same window ion**
shows, at **any energy** of the window, the step before it: a reading that is
rank 1 among its ≤ 3-piece candidates (what `k_max` 3 would rank first) and
whose loss is this loss minus **one named closed-shell library molecule**,
whichever pieces the decomposition prints. `ladder_support` records that loss
(the first printed piece, in the decomposition's order, that is one and finds
one; else the first such step). The printed
decomposition is one of several of the same loss — fewest pieces, then fewest
unnamed, then library order — and need not hold the piece the ladder took:
C9H22NO6⁺ → C8H11O⁺ loses CH11NO5, printed `H3N + CH4O + H2O2 + H2O2` (the
decompositions with water need an unnamed piece), while C8H13O2⁺ (loss CH9NO4,
`H2O + H2O + CH5NO2`) is read one H2O back; testing only the printed pieces
left C8H11O⁺ unexplained whatever was read (tested, synthetic). One molecule
means one named piece with no unnamed remainder and no radical: two waters, an
unnamed remainder or OH• do not make a step — printed or not. A printed piece
once counted whatever it was, so the gate followed what the printer happened
to list: the hexose [M+NH4]⁺'s C6H8O3⁺ (loss H8NO3, printed
`H2O + H2O + H3N + HO`) was kept on C6H9O4⁺ one OH• back, while a loss
printed with HO2• (H8NO4) was refused the same step; and C5H5⁺ (loss CH11NO6,
`H2O + H2O + H2O + CH5NO3`) was kept on C6H10NO3⁺ by its unnamed remainder.
Both are unexplained now, as when alone (tested, synthetic). The losses are compared as
**count vectors** on `subformula`'s fixed
element axis, never as strings: `format_formula` sorts N and ^N by the same
key and keeps their input order, so a formula built from a set followed the
interpreter's hash seed, and for a loss holding both (a 15N-ammonium adduct
of an N-containing molecule) about a third of runs dropped a supported step
(tested across seeds, one process each). Unsupported four-piece readings are
dropped and the table is **re-ranked**: `rank` and `n_candidates` within
(spectrum, peak, parent) — so a peak two window ions read keeps a rank 1 for
each — and `n_parents` within (spectrum, peak); a peak whose every reading
was dropped becomes one unexplained row (§3 step 8). A four-piece reading
never supports another.

| option | mode (`gate`) | read at | what is kept |
| --- | --- | --- | --- |
| (default) | `ladder-k4` | `k_max` 4 | ≤ 3 pieces; 4 pieces on a ladder |
| `--no-ladder-k4` | `strict-k3` | `k_max` 3 | ≤ 3 pieces (§3) |
| `--k-max K` | `ungated` (`strict-k3` for K = 3) | `k_max` K | every reading of ≤ K pieces, no ladder test; the manifest's `gate_note` says so — more chance compositions than the default for K > 3, stricter for K < 3 |

Why a ladder and not a blanket `k_max` 4: every extra piece lets more chance
compositions read at every peak (§8). The fourth piece is what the ammonium
adducts of polyols and oxygenated products need — [M+NH4]⁺ loses NH3 and then
water step by step, and NH3 + 3 H2O (DBE −3) is four molecules whatever the
library — and the gate admits such a chain only when its previous step is
observed. Example: the hexose [M+NH4]⁺ C6H16NO6⁺ → C6H7O3⁺ (loss H9NO3,
`H2O + H2O + H2O + H3N`) is read only when C6H9O4⁺ (loss H7NO2,
`H2O + H2O + H3N`) or C6H10NO3⁺ (loss H6O3) is read in the same window. On our
own 2026-09-10 and 2026-09-11 files no peak needs a fourth piece at all: the
gate, the strict `k_max` 3 and even the ungated `k_max` 4 read exactly the same
tables (tested).

### 11.3 Breakdown — survival per window ion and energy

Only energy-resolved spectra (a finite NCE) enter the breakdown, lowest NCE
first (an energy-averaged spectrum is read but has no energy axis; said). A
spectrum the server returned without peaks (`n_peaks` 0 in the parents table)
measured nothing and is left out, said; a window of nothing else is skipped
(`no spectrum with peaks`).
**One activation kind per breakdown:** an NCE of CID is not an NCE of HCD, so
a parent with spectra of several kinds is broken down on the kind with the
most spectra (HCD on a tie); the others are read (`ms2_readings.csv`) but left
out, said, and the manifest records the window's `activation_kind`.
Every peak of a spectrum gets one **role** (`peak_roles`): the **precursor** of
window ion X (its precursor reading — which wins over any fragment reading of
the same peak; a peak two window ions both read as their precursor goes to
the first in window order, and the window says so), a **fragment** (keyed by
its rank-1 formula; when the window ions that read it read different
formulas, the formulas joined by ` | `; its **candidates** are the window
ions that read it, in window order), **unexplained**, or **harmonic** (a ghost,
§11.12: in no fragment, and out of the TIC and `n_peaks`, counted apart in
`n_harmonic` / `harmonic_intensity`).

**One fragment across energies** (`merge_fragment_peaks`). The key above is
one spectrum's, and the rank-1 formula of one peak can change between
energies: where two compositions lie within the reading tolerance of each other
— the N + ^N / C + O + H alias of 15N data, 0.443 mDa, 5.2 ppm at m/z 85, at
the edge of 5 ppm; Cl against C2 + ^N − 4 H, 0.04 mDa — a sub-ppm jitter of the peak brings
the second one in and the ranking may prefer it. Keyed per spectrum, one
fragment became two rows: one with a late onset and `in_source` false, each
with part of the intensity, and the second split with a co-isolate that could
make it by composition (tested, synthetic: an organonitrate C10H21N^NO5⁺ whose
C4H8N^N⁺ reads C5H9O⁺ at one energy, 0.3 ppm low — its F at that energy read
0.82 instead of 0.70, its E50 17.6 instead of 16.7). So the window's fragment
peaks, every spectrum, are linked when their calibrated m/z lie within the
reading tolerance of each other (single linkage along the m/z axis) or when
they share a rank-1 formula (two peaks of one formula may sit up to twice the
tolerance apart), and each group is **named by every rank-1 formula read in
it**, sorted and joined by ` | ` (`mz_theo` NaN then). Two peaks of one
spectrum in a group are one fragment, summed. An unexplained peak is never a
link. **Two resolved peaks are never one fragment:** two peaks of one spectrum
more than the reading tolerance apart are two ions, and no chain of links joins
them (a cannot-link; the links are tried shortest first, the formula links
before the m/z ones). Single linkage alone let a weak centroid between two
resolved fragments chain them: in a synthetic window at m/z 57, C2H4N^N⁺ (the
target's only) and C3H5O⁺ (either ion's), 0.443 mDa apart and both in the NCE 20
and 30 spectra, became one fragment through a 5-count centroid at NCE 1 midway —
in-source, split 0.5 / 0.5, both E50s moved by 1–2 NCE and the target's
branching flagged. A peak that reaches two groups which cannot be one joins
neither: it keeps its own name (a peak read as one formula only is then that
formula's fragment; one read as two is a fragment of its own), so a few counts
never merge two fragments, and C2H4N^N⁺ stays the target's by composition
(tested). Where the centroid joins, it still counts: read as C3H5O⁺ by both
ions it makes C3H5O⁺ present at NCE 1, and the intact test does not run on a
fragment present at the lowest energy (§11.5) — a presence has no intensity
floor. Placed nearer C2H4N^N⁺, read as that only, it joins that fragment and
every E50 is what it is without it (tested). On our own files no fragment
changed its formula between energies, and no group holds two peaks of one
spectrum beyond the tolerance: the tables are what per-spectrum keys gave
(tested).

- **I_X** — X's own precursor peak(s) (two centroids within tolerance are
  summed; `n_precursor_peaks`).
- **Fragments of X** — the fragments X is a candidate of. One only X can
  make counts fully; a shared one by X's **weight** from the attribution
  (§11.5): its MS2-survivor share, or 1 / 0 after a decisive intact test.
- **F_X = I_X / (I_X + Σ w·I_fragment)** — the survival E50 is read off: a ratio
  within one spectrum, so it holds whatever happened between the spectra of
  two energies — a drift of the precursor supply (source, transmission,
  isolation), space charge, an AGC misprediction scaling a spectrum as a whole.
  Orbitrap intensities are reported **per unit injection time** (AGC fills the
  trap to a target charge, so intensity × injection time is what stays
  constant): the injection time itself moves nothing.
- **F_lo** — the same with every fragment X can make by composition given to X
  whole (w = 1, the intact test ignored): the lower bound over the observed
  products. F_lo..F is the band — F_lo bounds F from below, but F is not an
  upper bound: a survivor split can give an ion more of a shared fragment than
  it made (§11.5, what the survivor split assumes).
- **S_X = I_X(E) / I_X(E_min)** — reported as `s`: it compares spectra recorded
  at different times, so it is a breakdown curve only with a **steady precursor
  supply** across the energies (said in the manifest and on every S row of
  `ms2_e50.csv`) — not a condition on the injection time, which the per-unit-time
  intensities already absorb.
- **Only observed products.** F, F_lo and every TIC fraction see only the peaks
  the spectrum holds. A product **outside the observed m/z range** — below the
  scan's low-mass limit, as the small reagent ion an adduct releases (an
  ammonium adduct's NH4⁺, m/z 18) — is in no sum: F and F_lo read high (an ion
  whose only channel is unobserved keeps F = 1 while S falls, tested), and the
  TIC falls, inflating every fraction that stays (§11.5). S does not depend on
  where the products go: with a steady supply it sees that loss.
- 0/0 (no precursor and no fragment X can make at an energy) is NaN: the ion is
  **unobserved** there — F, F_lo and S alike (an S of 0 there was read as a
  crossing through a hole F calls a `gap`, tested), and its branching, gain and
  loss there are NaN too (§11.5).
- **Ion budget** per spectrum (`ion_budget`): `tic` (every peak's intensity),
  `budget_explained` (precursor + fragment peaks of the window ions) and
  `budget_unexplained` = 1 − explained, with the peak counts.

### 11.4 E50 and the centre-of-mass energy

`e50(nce, value, volts)`: linear interpolation at the **first downward
crossing of 0.5** between two observed energies, on the NCE axis, and with the
same fraction on the volts axis (NaN when the volts are unknown — a refused
summary). `E_cm = V · m_gas / (m_gas + m/z)` with `m_gas` = 28.0061 (N2,
`--gas-mass`): an **assumption** — the collision gas is not in the data Mascope
serves — and a parameter; `m/z` is each window ion's own (a co-isolate's E_cm
is not the target's). E50 is read from F, F_lo and S (`curve` F / F_lo / S).

The volts E50 needs only the two spectra that bracket the crossing, so the
`note` of `ms2_e50.csv` says what the volts were: `volts unknown` when no
spectrum of the breakdown has volts (a refused summary); `volts known for
k of n spectra` when some lack them (a summary group missing for one centroid
key, §10.2) and the E50 on volts still stands; `volts unknown at the crossing
(known for k of n spectra)` when a bracketing spectrum is one of those, and
`e50_v` / `e50_ecm` are NaN. The manifest's `volts_known` is true only when
every spectrum of the breakdown has volts (and the figure's volts axis is drawn
only then).

| `flag` | meaning | E50 |
| --- | --- | --- |
| `interpolated` | crossed between two neighbouring observed energies | the value |
| `below_range` | already < 0.5 at the lowest observed energy (E50 below it) | NaN |
| `above_range` | never below 0.5 (E50 above the highest energy) | NaN |
| `single_energy` | one observed energy | NaN |
| `gap` | the crossing lies across energies where the ion is unobserved: not interpolated across the hole (the 2026-09-10 file's 0.9 % co-isolate C8H10NO⁺ is seen at NCE 1 and next at 81; a line across would read 41) | NaN |
| `no_data` | never observed | NaN |

At the edges: exactly 0.5 is not below 0.5 (a curve starting at 0.5 crosses
at its first energy, `interpolated`); a hole of one unobserved energy is a
`gap` as much as a longer one; a spectrum with no NCE is not a point of the
curve, nor of its span.

`monotonic` False when the curve rises by more than `MONOTONIC_TOL` (0.02)
between two observed energies; the first crossing is still the one reported.
On our own files no window ion's F rose at all; an intact ion's precursor
intensity (S) wanders by about ±2 % between steps.

### 11.5 Fragments and attribution

Per (window, fragment), `ms2_fragments.csv`: its **onset** (lowest NCE where
present; `onset_v`), `in_source` (present at the window's lowest energy), the
energy and value of its largest TIC fraction, its candidates and their losses,
and the call.

Its **candidates** (`candidates_across`) are the window ions that read it as
one of the formulas it **can be where it sits**. Where it sits is its
**signal-weighted m/z** (`fragment_mz`): the mean of its peaks' calibrated m/z
over the energies, each weighted by the share of its spectrum's TIC it holds —
a TIC fraction, not a raw intensity, so a spectrum scaled as a whole moves
nothing (tested). The formulas it can be there (`fragment_compositions`) are
those read in it (rank 1, by any window ion) whose exact m/z lies within the
reading tolerance of that m/z: the tolerance every single peak is read with,
applied to the best estimate of the peak's position. Which ions can make a
fragment is a composition question, and one centroid answers it only as well
as that centroid is placed — a peak of a few counts says little about where
the ion is, and counted spectrum by spectrum it weighed as much as the
strongest:

- In the organonitrate case above a CHO co-isolate reads the merged fragment as
  C5H9O⁺ in the one spectrum where the peak sat 0.3 ppm low. Weighted by the
  signal the fragment sits 0.05 ppm below C4H8N^N⁺ and 5.16 ppm above C5H9O⁺,
  beyond 5 ppm: the co-isolate is not a candidate (a candidate there took half
  of a fragment it cannot make where it sits).
- The mirror (tested, synthetic): the co-isolate makes C5H9O⁺, centroided
  1.2 ppm high at NCE 10–30, the target never loses signal, and at NCE 1 a stray
  centroid of 5 counts sits at the exact C4H8N^N⁺ m/z, which only the target can
  read — within 5 ppm of the others, so it merges with them. Asking for a
  reading in **every** spectrum let that centroid veto the co-isolate, which
  reads the fragment in the three spectra holding 99.7 % of it: the target got
  the fragment by composition, its F fell to 0.53 with its branching flagged at
  every energy above NCE 1, and the co-isolate read no E50 (16.67 is its own).
  Weighted, the fragment sits 1.2 ppm above C5H9O⁺ and 4.0 ppm below C4H8N^N⁺:
  both ions are candidates, and the intact test decides (below).
- When the jitter is where the signal is — the organonitrate's peak 0.3 ppm low
  at NCE 10, 20 and 30 — the fragment sits 4.92 ppm above C5H9O⁺, inside the
  tolerance of both formulas, and both ions are candidates (tested): the
  spectra cannot tell which composition it is.

An ion that reads it only as formulas beyond the tolerance of that m/z is left
out, and `test_note` says so (`not a candidate: C15H22O3⁺ reads it only as
C5H9O⁺ (+5.16 ppm) — beyond the reading tolerance of its signal-weighted m/z
85.06523`); when no formula read in it lies within the tolerance (the readings
disagree), every ion that reads it is a candidate, said. On our own files
every fragment is read as one formula by the same ions in every spectrum: the
rule changes none of their tables (tested).

**An organic reading outranks a carbon-poor polyoxy one.** In 15N data
C2F3 = H2^NO4 within 0.21 mDa, so CF3COO⁻ (C2F3O2⁻) reads as H2^NO6⁻ and
C3H2F3O2⁻ as CH4^NO6⁻ — sub-formulas of every [M+^NO3]⁻ cluster, and the
fragment gate (§3) admits them. When the formulas a fragment can be where it
sits include a carbon-bearing one (more than `POLYOXY_MAX_C` 1 C) and a
**carbon-poor polyoxy** one (`polyoxy_inorganic`: ≤ 1 C, ≥ `POLYOXY_MIN_O` 4 O,
nothing but C, H, N, ^N and O — a cluster of inorganic pieces on a nitrate or
carbonate core, never an organic fragment), an ion that can read it only as the
latter is not a candidate, said (`not a candidate: C6H12^NO9⁻ reads it only as
H2^NO6⁻ — a carbon-poor polyoxy composition, while C2F3O2⁻ (organic) is read
there too`). Without that, a fluorinated co-isolate's in-source CF3COO⁻ was
shared by survivors with the intact nitrate cluster, whose F fell below 1
(tested, synthetic). The rule needs the organic reading: with no fluorinated
ion named, nothing reads the peak as C2F3O2⁻ and the cluster takes it by
composition (§11.9). And it needs an organic reader the **sample holds**: one
in no MS2 spectrum of the window nor in the MS1 window (`absent_ions`,
below) cannot have made the peak, so the polyoxy readers stay candidates,
said (`C9H14F3O4⁻ reads it as C2F3O2⁻ (organic) but is absent from this
sample: C6H12^NO9⁻, which reads it only as carbon-poor polyoxy, stays a
candidate`). A shared ion list naming a fluorinated ion that falls inside
another window where it is absent once handed that ion the cluster's fragment
with weight 1 and moved the cluster's E50 from 16.67 to 21.8, while the window
note said the absent ion weighs 0 (tested, synthetic).

**A side lobe is not a fragment.** A peak list processed from Orbitrap profiles
can carry weak centroids a few line widths beside a much brighter line that
the profile does not hold (`chem.sidelobes`, the MS1 guard). Read against the
window ions such a centroid is a sub-formula like any other — of a large O/N-rich
ion almost anything is — and was given to that ion by composition: in a
synthetic organonitrate window the target, which never fragments, read a
"fragment" C7H14^NO3⁺ (loss C3H7NO2) from a 1 % centroid 2.2 mDa below its
co-isolate's C11H13O⁺ (2.3 line widths at a resolving power of 170 000, where
a negative side lobe sits); its F fell to 0.991 while it never lost a count,
and its branching was flagged at every energy (tested). A fragment **every**
peak of which, in every spectrum it is in, sits
within `SATELLITE_FWHM` (5) line widths of a peak of the same spectrum that is
at least `SATELLITE_RATIO` (50) times brighter is set aside
(`satellite_fragments`): its peaks stay in the TIC as unexplained, it gets no
row in `ms2_fragments.csv`, and the window says so (`side lobe: C7H14^NO3⁺: each
of its 3 peaks (3 spectra) sits within 5 line widths of a peak >= 50x brighter
…`).
The width is the brighter line's m/z over the resolution its centroid reports;
where a line reports none the peak is not tested (a run whose centroids carry
no resolution is read as before). A fragment one of whose peaks clears the band
or the ratio is an ion and is kept; the cost is a real fragment that is 50×
weaker than a line within 5 widths at every energy it is seen — set aside,
said. On our own three files no fragment is a side lobe (tested).

- **A named ion the sample lacks decides nothing.** A candidate whose precursor
  is in **no MS2 spectrum of the window** (at any energy — an ion located only
  at a later energy, the target too, is not absent and keeps its fragments) and
  whose MS1 share is a **measured** 0 (`absent_ions`) weighs 0 (below), is in
  no curve (not even F_lo of a fragment it shares), and is **set aside before
  the intact test and the guard**, said in `test_note` (`C12H19O⁺: in no MS2
  spectrum of the window nor in the MS1 window (absent from this sample):
  weight 0, set aside before the intact test`); the guard's bound counts a
  fragment it shares as the other ion's own. Kept in the test it was never
  excluded (absent at the lowest energy, it cannot be tested), so the test
  left two candidates where it should leave one and the guard, which reads a
  single call, was skipped: in the guard case below a third named ion this
  sample lacks let the 1 % co-isolate take 100 % of C9H11⁺ — its E50 5.0
  instead of 7.48, the main ion's F flat at 1 (tested, synthetic). Nor does
  the intact test wait for it to read the fragment: the onset is where every
  candidate the sample holds reads it. An absent CHO ion that reads an
  organonitrate's fragment only where it sits at C5H9O⁺ (the 0.44 mDa CO /
  N^N alias, at the tolerance edge), not at its C4H8N^N⁺ onset, once moved
  the onset one energy later as a 'stray': a decisive intact-test call became
  an ambiguous split with the co-isolate's branching flagged, or a survivor
  split became a call — every weight, F and E50 the same now with and without
  the name (tested, synthetic). A candidate
  whose MS1 share is **unknown** (no window table, or beyond the inventoried
  span) is not absent: it stays, untestable, and is split equally (below).
- **One candidate** — or one the sample holds, the others set aside — → that
  ion (`basis` `composition`), as if the absent ones were not named.
- **Only absent candidates** → **no ion** (`call` `''`, `basis` `absent only`,
  every weight 0, `weight_source` `none`, said in `test_note`): an ion the
  sample lacks decides nothing, so it takes nothing either. The fragment is in
  no curve (not in F_lo, which counts every composition candidate, either), so
  the absent ion shows no breakdown — F, F_lo and S NaN, E50 `no_data` —
  rather than an F of 0 read off a peak the sample's own ions cannot make; its
  peaks count as unexplained in the ion budget, as without the name.
- **Several** → the **intact test** on TIC-normalised precursor fractions: a
  candidate whose fraction at the fragment's onset is ≥ `INTACT_FRACTION` (98 %)
  of its fraction at the **lowest energy** — not at the step before the onset:
  an ion that fell before the fragment appeared may still be the one making it,
  and is not excluded for holding still since (tested: a shared fragment first
  seen at the third energy stays ambiguous) — has lost nothing while the
  fragment appeared, and is **excluded** — unless the fragment is in-source (an onset at
  the lowest energy says nothing), a candidate is absent at the lowest energy
  (it cannot be tested and stays), or every candidate would be excluded (none
  is). The **onset** the test reads is the first energy at which **every**
  candidate the sample holds reads the fragment (`shared_presence`; an ion
  set aside as absent is not waited for) — when what some of them
  read before it is a **stray**: at every earlier energy at most
  `ONSET_STRAY_SHARE` (5 %) of the fragment's TIC fraction at that first
  shared energy. A weak peak only some of them can read sits where the others
  cannot make it, so it is not yet the fragment they share. In the mirror case
  above the stray centroid (1.7 % of the fragment's fraction at NCE 10) made
  the fragment 'in-source' and no test ran (an even split, the target's F
  falling to 0.69, flagged); read from NCE 10, where both ions read it, the
  target is still 100 % of its NCE 1 fraction and is excluded, and the
  co-isolate is called — E50 16.67, the target's F 1, nothing flagged, the
  same E50s as without the stray centroid (tested). A **larger** presence is
  the fragment itself, placed by a sub-ppm jitter where only some candidates
  read it, and its own onset stands — at the lowest energy: in-source, no
  test. The reverse of the mirror (tested, synthetic): the target holds an
  in-source C4H8N^N⁺ of 300 counts at every energy and never fragments, read
  at NCE 1 by the target only, from NCE 10 on 0.5 ppm low, where the falling
  co-isolate reads it as C5H9O⁺ too. Read at NCE 10 the target was intact and
  excluded, and the co-isolate was called by `intact test` — with the 300
  counts only the target can have made at NCE 1 — its E50 11.67 and the
  target's F flat at 1, nothing flagged; the NCE 1 peak holds 1× the
  fragment's fraction at NCE 10, so it is in-source and split by the
  survivors (the co-isolate's E50 14.17; without the jitter the target's by
  composition, 16.67). `test_note` says which rule applied: the onset moved (a
  stray, with its share), the fragment's own onset kept (with its share), or
  no spectrum holds a reading by every candidate (no onset: no test). Like F,
  the intact test, the guard and the branching see only observed products:
  when a window ion loses signal to a product outside the observed m/z range
  (§11.3) the TIC falls, and a co-isolate that fragments can keep its TIC
  fraction — excluded as intact, the shared fragment called the other's, and
  nothing flagged (tested, synthetic: a 5 % loss of the co-isolate read 1.000 of
  its lowest-energy fraction while the target lost 6 % unobserved). S sees it
  (with a steady supply); a TIC that falls between energies says to look. One
  candidate left → called (`intact test`) — unless the **mass-balance
  guard** withdraws it: the fragment's TIC-fraction gain at some energy (every
  energy is checked, not only the last) exceeds everything the called ion held
  at the **lowest energy** (× `BRANCHING_FLAG`; not its largest share at any
  energy: as the TIC falls a minor ion's share can rise without it holding more,
  tested) — its precursor **plus the fragments only it can make** by
  composition (`held_at_lowest`). Why the
  guard: in the 2026-09-10 file's 179.107 window C9H11⁺ appears at NCE 5, where
  C11H15O2⁺ is still 98.7 % of its NCE 1 fraction and the 0.55 % co-isolate
  C11H16^NO⁺ lost 1132 counts against the fragment's 1118 — the onset is the
  co-isolate's; but by NCE 10 C9H11⁺ has grown by 0.0083 of the TIC, past the
  0.0055 that co-isolate held (it has no fragment of its own at NCE 1), and by
  NCE 45 it holds 39k counts, 17× everything that co-isolate held, and is
  C11H15O2⁺'s (−C2H4O2). One call per fragment cannot be both; the guard leaves
  it ambiguous. Why the ion's own products count: a fragile ammonium adduct can
  be mostly its in-source products at the lowest step (a ^N fragment only it
  can make), and what those products go on to make — a second-generation
  fragment another window ion could make by composition — is still carried by
  it. A bound of its precursor alone withdrew such a correct call and handed
  the fragment to an intact co-isolate, which then read a false E50 while the
  adduct's branching went negative, unflagged (tested, synthetic). The guard
  reads a call made at a moved onset as any other (tested: a co-isolate of
  1/40 the size, called at the shared onset, cannot carry the growth — withdrawn,
  both notes kept).
- Else **`ambiguous`**, with **share weights** over the remaining candidates,
  each candidate weighed by what is measured of **it** (`share_weights`) — one
  candidate's missing number never changes how the others split:
  - its lowest-energy MS2 survivor when it has one (> 0) — whatever the MS1
    window holds: an ion in the MS2 spectrum is never given 0 for want of an
    MS1 centroid;
  - in neither the lowest-energy MS2 spectrum nor the MS1 window (an MS1 share
    of 0, measured): weight **0**, and the others split as if it were not named
    (it never reaches the split: set aside above; and an equal split is over
    the candidates not absent, tested);
  - no survivor but an MS1 share: that share put on the survivors' scale
    (survivor per unit share of the candidates that have both), said in
    `test_note`;
  - no candidate with a survivor: the MS1 shares alone;
  - a candidate with no survivor whose MS1 share is not known (no window table
    for the parent, or beyond the span `ms2 extract` inventoried, §11.1), or no
    candidate with both to set the scale: an equal split over the candidates
    not absent from both, said.

  `weight_source` is `ms2 survivors`, `ms2 survivors + ms1 window`, `ms1 window`
  or `equal` (and `none` for a fragment only absent ions can make). Excluded candidates weigh 0. The MS1 shares are **one-to-one**:
  each centroid goes to the named ion nearest it (within max(`TARGET_MATCH_PPM`,
  `tol_mda`), its m/z read through the folder's axis offset, §11.1), never to two; and a window ion the MS1 window holds no centroid
  for has a **measured share of 0**, not a missing one — inside the inventoried
  span; beyond it, NaN (said in a note when its precursor is in no MS2 spectrum
  either). Two earlier rules failed an ion list shared across windows. An equal
  split whenever a survivor was missing took half of every fragment a named but
  absent ion could make from the target (a synthetic run's E50 16.7 read 21.8).
  Then MS1 shares for **every** candidate whenever one lacked a survivor: on
  standard A's adduct window, naming C11H23⁺ too (in no MS2 spectrum, 0.025 % of
  the MS1 window) switched all five shared fragments from the survivor split
  0.719 / 0.281 to 0.526 / 0.474 and moved the two E50s from 8.61 / 6.77 to
  10.81 / 3.38. Weighed per candidate, it takes its 0.025 % and the E50s move by
  < 0.005 NCE; without its MS1 centroid nothing moves at all (tested, with a
  synthetic phantom too).

**What the survivor split assumes.** Survivor weights are fixed: each
candidate takes a shared fragment in proportion to its intensity at the lowest
energy, at every energy — as if the window's ions fragmented alike. On the
2026-09-11 standard A's adduct window (155.1559: C10H20^N⁺, with C10H19O⁺
co-isolated 13 mDa below) every fragment is a sub-formula of both ions and
C10H17⁺ and C6H9⁺ are already there at NCE 1, so all five are `ambiguous`,
split 0.719 / 0.281 (tested):

- When every fragment of a window is shared by all its ions and split by
  survivors, every ion's F at the lowest energy is one number by construction:
  F_X = I_X / (I_X + (I_X / ΣI) · ΣI_f) = 1 / (1 + ΣI_f / ΣI) — 0.713 for both
  ions there. It says what the window holds, not how fragile each ion is.
- The split breaks the per-ion balance while the window's holds. Its products
  gain 0.996, 1.010 and 1.030 of what its two precursors lose at NCE 5, 15 and
  25 (TIC fractions); but the adduct's branching sums to 1.415 at NCE 5 and
  1.100 at NCE 15 (both flagged) and the co-isolate's to 0.566 and 0.835. The
  co-isolate is the more fragile ion — its TIC fraction falls by 28 % by NCE 5,
  the adduct's by 11 % — and the survivor split gives it less than it made. The
  flag here says the split is off, not that a fragment is mis-attributed.

The adduct's F E50 (8.61) carries that uncertainty, and F_lo..F does not bound
it on both sides: F_lo, every shared fragment to the adduct, reads 6.14; a
split by each ion's own TIC-fraction loss at each energy reads 10.11 for the
adduct and 3.96 for the co-isolate (the survivor split: 8.61 and 6.77). That
loss split is not used. It gives every ion the window's ratio of gain to loss,
so the per-ion check could no longer fail for a shared fragment whatever the
attribution; it is undefined at the lowest energy, where two of the five
fragments are already present and nothing has yet been lost; and like the
survivor split it divides every fragment alike. Nor is there a fit (below). So
the survivor split stays and its breach is flagged: read the E50 of a window
whose fragments are all shared together with its F_lo and its branching flags —
on standard A's adduct the splits considered here span NCE 6.1–10.1.

**One fragment, one call.** The merge cannot tell a peak that jitters across an
alias from two fragments of the two compositions — 0.44 mDa apart, each strong
at its own energies — and makes them one fragment with one call; its part
where only some candidates read it goes with that call. On our own files no
fragment reads as two compositions.

The `losses` of `ms2_fragments.csv` follow its `candidates`, in window order,
each candidate's loss taken from the lowest energy at which it reads the
fragment as one of the formulas it can be where it sits (else at which it
reads it at all).

**Scale-free.** Every quantity of the attribution — the intact test, the guard,
the branching — is a TIC fraction, and F and F_lo are ratios within one
spectrum: a spectrum scaled as a whole (a drift of the precursor supply, space
charge, an AGC misprediction) changes no call, weight, F, F_lo, E50 on them or
branching; only S and the raw intensities move (tested at three energies and
two factors; the intact test on raw counts made two calls ambiguous and moved
both E50s by 3 NCE).

No least-squares deconvolution: with two or three window ions and a handful of
energies a fit has as many unknowns as points, and would hand back a number
for a fragment the data cannot attribute.

**Branching** (`ms2_branching.csv`): of X into fragment f at energy E,
`w(X, f) · (φ_f(E) − φ_f(E0)) / (φ_X(E0) − φ_X(E))`, φ = intensity / TIC, E0 the
lowest energy; NaN where X lost nothing; a fragment that fragments on counts
negative; and at an energy where X is unobserved (no precursor, no fragment it
can make) the ratio, the gain and the loss are all NaN — X was not seen there,
it did not lose everything it held. Its sum per (X, E) is `branching_sum` in
`ms2_breakdown.csv`, and a sum above `BRANCHING_FLAG` (1.05) is **flagged**
(`branching_flag`, and a note) — never clipped: a precursor cannot give more
than it lost, among the products the spectra hold. The flag reads the
weighted gain against the loss (`branching_breach`), not the ratio: where X lost
nothing or **gained** (the ratio is NaN) any growth of its products breaks the
invariant worst of all, and is flagged too — the note gives the products' gain
and X's own change there. On the 2026-09-10 file: C11H13O⁺ at NCE 11 (its
products +0.0205 of the TIC while its own fraction rose by 0.0020) and C7H6NS⁺
at NCE 61 (+0.0123, +0.0067), each a larger excess than the flagged sum 1.059
of C7H6NS⁺ at NCE 81. Three causes, all
said in the note: a fragment mis-attributed or a share weight off (the
2026-09-10 file's 197.13 co-isolate C12H21O2⁺, 0.8 % of the window, gets 1.8 %
of the adduct's in-source fragments by survivor share and barely fragments
itself: sums 2–6); ions not named here losing signal; or products leaving the
observed m/z range (§11.3) — the last two take signal out of what is named, and
TIC normalisation turns that into a gain of every named fraction. The note
gives, for each step it names (each value labelled with its NCE, in the order
the steps are named), how far the unexplained share of the TIC moved from the
lowest energy and the TIC against the lowest energy's: unnamed ions move the
unexplained share; a TIC that falls while the unexplained share holds says
products left the observed range (or the precursor supply fell). On the
2026-09-11 standard A's
[M+H]⁺ window (one window ion, so mass balance is exact) the sum at NCE 5 is
1.173: an excess of 0.173 in branching units, which times the precursor's
TIC-fraction loss (0.0664) is 0.0115 of the TIC — exactly what the unexplained
share fell by. About half of that (0.0058) is the precursor's **own side
lobes**: unexplained centroids within 4.6 mDa of it, three of them within
1.6 mDa of one another, which no named ion can claim; the rest is unnamed
signal elsewhere in the window.

### 11.6 Constants

| constant | value | role |
| --- | --- | --- |
| `readings.LADDER_K` | 4 | the piece count the ladder gate admits |
| `analyze.TARGET_MATCH_PPM` | 10 ppm (floor `tol_mda`) | target = named ion nearest the located precursor (read through the folder's axis offset) |
| `analyze.ION_CHECK_PPM` / `_MDA` | 10 ppm / 5 mDa | the typed m/z of `--ion` vs the formula |
| `sources.WINDOW_PAD_DA` | 0.005 Da | window ions: ±(width/2 + pad) of the key |
| `attribute.INTACT_FRACTION` | 0.98 | the intact test (an ion that has not started to fragment wanders ±2 %: the 2026-09-10 file's C7H6NS⁺ reads 82.2 % of its spectrum at NCE 1 and 83.9 % at 11) |
| `attribute.BRANCHING_FLAG` | 1.05 | branching sum flag; the guard's margin |
| `attribute.POLYOXY_MAX_C` / `POLYOXY_MIN_O` | 1 C / 4 O | a carbon-poor polyoxy H/N/O composition, outranked by an organic reading of the same fragment by an ion the sample holds (§11.5) |
| `attribute.ONSET_STRAY_SHARE` | 0.05 | the intact test moves a fragment's onset to the first energy every candidate reads it only past a stray: ≤ this share of the fragment's TIC fraction there, at every earlier energy (§11.5) |
| `attribute.SATELLITE_RATIO` / `SATELLITE_FWHM` | 50× / 5 line widths | a fragment every peak of which sits this close to a peak this much brighter is a side lobe, set aside (the MS1 side-lobe guard's ratio and band, `chem.sidelobes`; §11.5) |
| `breakdown.E50_THRESHOLD` | 0.5 | E50 |
| `breakdown.MONOTONIC_TOL` | 0.02 | non-monotonic flag |
| `breakdown.N2_MASS` (`--gas-mass`) | 28.0061 Da | E_cm (an assumption) |
| `analyze.FIGURE_TOP_FRAGMENTS` | 6 | fragments drawn in panel B |
| `replicates.BOOTSTRAP_B` (`--bootstrap`) | 1000 | resamples of each E50's interval (0 = none; §11.11) |
| `replicates.BOOTSTRAP_MIN` | 100 | the fewest resamples a band is read off (fewer are refused: the percentiles of a few rest on one or two of them) |
| `replicates.BOOTSTRAP_SEED` | 20261018 | the resampling seed, with the parent key per window |
| `replicates.BOOTSTRAP_PERCENTILES` | 16 / 84 % | the interval: those percentiles of the resampled E50s (what the band holds: §11.11) |
| `replicates.MATCH_DA` | 1e-6 Da | a spectrum's peak and its per-scan rows are one peak |
| `replicates.SCALE_BAND` | 0.5–2 | the factor (averaged height / per-scan mean) a peak's per-scan readings may be scaled by; outside it the peak is held (§11.11) |
| `replicates.ZERO_WIDTH_REL` | 1e-9 | an E50 band whose bounds agree to this share of their size is of zero width |
| `readings.HARMONIC_ORDERS` | 2f, 3f | the harmonics read (at m/z / 4, m/z / 9; §11.12) |
| `readings.HARMONIC_PPM` | 40 ppm | how far a ghost may sit from m/z / n² |
| `readings.HARMONIC_CAP` | 2f 0.5 %, 3f 8 % | the most a ghost holds of its line in the same spectrum |
| `readings.HARMONIC_CLEAR_SN` | 10 | the S/N a reading above the cap needs to make its place an ion's at every energy (§11.12) |

### 11.7 Outputs

`peaky ms2 analyze --run-dir DIR --ion MZ=FORMULA [--ion ...] [--ions CSV]
[--k-max K | --no-ladder-k4] [--tol-ppm 5] [--tol-mda 0.3] [--gas-mass 28.0061]
[--iso-width W] [--mass-offset-ppm PPM] [--bootstrap N] [--keep-harmonics] [--no-figures]
[--out-dir DIR]` writes into `--out-dir` (default: the run folder itself):

| file | content |
| --- | --- |
| `ms2_readings.csv` | every reading of every spectrum (§11.2) |
| `ms2_breakdown.csv` | per (window, spectrum, window ion): `nce`, `hcd_v`, `role` (target / co-isolate), `ion_mz`, `intensity` (I_X), `n_precursor_peaks`, `frac` (I_X / TIC), `frag_weighted`, `frag_all`, `f`, `f_lo`, `s` (S: valid with a steady precursor supply), `branching_sum`, `branching_flag`, `tic`, `budget_explained`, `budget_unexplained`, `n_peaks`, `n_unexplained` (`tic` and `n_peaks` without the harmonic ghosts), `n_harmonic`, `harmonic_intensity` (the ghosts left out, §11.12); with the per-scan tables (§11.11; else 0 / NaN) `n_scans` (the spectrum's), `intensity_cv` (the precursor's CV over them), `f_sd`, `f_sem`, `f_lo_sd`, `f_lo_sem`, `s_sd`, `s_sem` (the scan-to-scan spread of F, F_lo, S: sd, ddof 1, and sd / √n) |
| `ms2_e50.csv` | per (window, ion, curve F / F_lo / S): `role`, `ion_mz`, `e50_nce`, `e50_v`, `e50_ecm`, `flag`, `monotonic`, `n_energies`, `nce_min`, `nce_max`, `value_at_min`, `value_at_max`, `note`; the bootstrap interval (§11.11; else NaN / 0 / '') `e50_nce_lo`, `e50_nce_hi`, `e50_v_lo`, `e50_v_hi`, `e50_ecm_lo`, `e50_ecm_hi` (16th / 84th percentiles of the resampled E50s; NaN where the percentile lies past the measured energies or across an unobserved one), `e50_lo_censor`, `e50_hi_censor` (where a NaN bound lies: `below` the lowest energy, `above` the highest, `gap` across an unobserved energy — between two observed ones, at an end of the energies that resamples do not observe, or about every energy where a resample observes none; '' for a measured bound), `boot_n` (resamples), `boot_cross` (the share that cross), `one_scan_bracket` (how many of the energies the band rests on add none to the curve — the two the averaged crossing lies between and, for S, the lowest energy every resample's S is read against, once where it is one of the two — fewer than two scans, or no spread of the curve there: 1 to 3, the band holds no scatter of theirs and the note says so; 0 where none is, with no crossing, or with no band); with the per-scan tables whatever `--bootstrap` (else 0) `n_one_scan` (the window's energies of fewer than two scans — one, or no per-scan row — or whose every precursor / fragment peak is held: they add no spread) |
| `ms2_fragments.csv` | per (window, fragment — one physical peak across the energies, named by every rank-1 formula read there): `mz_theo` (NaN for a name with ` \| `), `candidates`, `n_candidates` (how many: `candidates_across`, after the ions beyond the tolerance of where it sits and the polyoxy readings outranked are left out), `losses`, `call`, `basis`, `weights` (`ion=w;…`), `weight_source`, `excluded`, `test_note` (the candidates set aside as absent from the sample; the intact test's note — its onset, when every candidate reads the fragment only from a later energy, moved past a stray or kept — and the guard's, then `candidates_across`'s), `onset_nce`, `onset_v`, `in_source`, `max_frac`, `max_frac_nce`, `n_spectra`, `presence_at_max`, `cv_at_max` (its presence and CV over the scans of the spectrum of its largest TIC fraction, §11.11; NaN without the per-scan tables) |
| `ms2_replicates.csv` | only with the per-scan tables (§11.11): per (window, spectrum, item) — `item` `precursor` (a window ion's) or `fragment`, `name` (the ion, or the fragment as `ms2_fragments.csv` names it) — `nce`, `hcd_v`, `n_scans`, `n_present` (scans in which a peak of it is read; 0 where none is, or where it is not matched), `presence` (their share), `mean` (on the averaged spectrum's scale: the averaged intensity), `sd`, `cv` (absence = 0), `matched` (False: a peak of it is held — read in no scan, or on another scale than its spectrum's, `SCALE_BAND` — whether or not another peak of it is read: `n_present` 0; presence, sd and CV NaN) |
| `ms2_branching.csv` | per (window, ion, fragment, energy above the lowest): `nce`, `hcd_v`, `weight`, `frag_gain`, `ion_loss`, `branching` (all three NaN where the ion is unobserved) |
| `breakdown_<parent key>.png` | (A) F of every window ion vs NCE, the F_lo..F band, the target emphasised, E50 in the legend; (B) the target's top fragments' TIC fraction vs NCE with its own; one colour, one ion across both panels (the fragments take colours no window ion wears; past the palette, grey with distinct dashes); volts on a secondary top axis when known; legends below the panels |
| `ms2_analyze_manifest.json` | the sample (from the extract manifest), the parameters (gate mode and its note, `k_max_read`, tolerances, gas mass, the isolation width and its source, the named ions, and every constant a call turns on: `intact_fraction`, `branching_flag`, `onset_stray_share`, `satellite_ratio` / `satellite_fwhm`, `polyoxy_max_c` / `polyoxy_min_o`, `e50_threshold`, `monotonic_tol`, `target_match_ppm`, `window_pad_da`, the m/z axis offset `mass_offset_ppm` and its `mass_offset_source`), each window's status, `reason`, target, co-isolates, `left_out` (aliases), `activation_kind`, located precursor, offset, NCEs and notes (and `offset_hint_ppm`, the
`--mass-offset-ppm` a window skipped for want of a named target needs, when its
nearest named ion is within the axis reach), the counts (incl. `four_piece_readings`, `harmonic_peaks` and `harmonic_by_order`, `replicate_rows`, `one_scan_energies`, `e50_intervals`; per window `harmonic_peaks`, `scans_per_energy`, `one_scan_energies`; the parameters' `harmonics` (on, orders, ppm, caps, `clear_sn`) and `replicates` (`per_scan`, `bootstrap`, `bootstrap_min`, `seed`, `percentiles`, `match_da`, `scale_band`)), the inputs' (the per-scan tables among them) and outputs' sha1, the code identity. No credentials, no URL (the PNGs carry no `Software` key) |

It prints one line per window (status, target, co-isolates, offset, the target's
E50 and, with the per-scan tables, its 16–84 % band: a NaN bound by its side —
`below range`, `above range`, `in a gap` — and its caveats, how many of the two
energies about the crossing add no spread and a zero width) and the
notes — except, in a folder of many parents (an inclusion-list folder), the
windows skipped for want of a named target: beyond
`cli.MS2_ANALYZE_UNNAMED_SHOWN` (5) of them they are counted in one line, their
notes kept in `ms2_analyze_manifest.json`. Those whose nearest named ion misses
by an axis offset (within `sources.AXIS_SEARCH_PPM`: the window's
`offset_hint_ppm`) are counted apart, the line naming the median
`--mass-offset-ppm` they need; a run in which no window was analysed says to
pass it rather than to name ions already named. Each window has a **status**
(and, when it has no breakdown, a `reason`, both in the manifest, with the
counts `analysed`, `readings_only`, `skipped`):

| status | `reason` | what was written |
| --- | --- | --- |
| `analysed` | — | readings, breakdown, E50s, fragments, branching, figure |
| `readings only` | `no energy-resolved spectrum` | readings only: energy-averaged spectra (a server older than Mascope 1.8, or an averaged extract) have no energy axis |
| `skipped` | `no named target` | nothing: no named ion at the located precursor (the note names the nearest) |
| `skipped` | `no precursor located` / `no spectrum with peaks` | nothing |

**Exit code:** 0 when a window was analysed, or when none was but some were
read (`readings only`). When every window is readings-only that is all the
folder holds, and it says so without `--ion` advice; when some windows are
readings-only and others skipped, the summary names the readings-only windows
(it never claims the folder has no energy axis: a skipped window was not
looked at), gives the `--ion` hint for each window skipped with no named
target, and lists any other skip reason. 1 when nothing was read: no MS2
parent in the folder (an MS1-only sample), every window skipped for want of a
named target (with the `--ion` hint), or for another reason (the reasons
given).

### 11.8 Known answers (our own data)

- **2026-09-11 standard A, [M+H]⁺ C10H17⁺** (137.1321, alone in its window):
  F E50 NCE 13.89 (6.33 V, E_cm 1.073 eV on N2). The raw-file breakdown of that
  day (per-scan centroids; survival = precursor / (precursor + tracked ions))
  read 13.8 (6.32 V); F agrees with its survival within 0.0013 at every step,
  i.e. within 0.07 NCE at the crossing's slope.
- **Standard A, the adduct C10H20^N⁺** (155.1559, C10H19O⁺ co-isolated 13 mDa
  below): F E50 8.61 (5.59 V), F_lo 6.14. The raw-file value, 1.6 (4.19 V),
  is a different definition: it put the co-isolate's intact peak and every
  shared fragment against the adduct. Rebuilt from the API spectra that
  definition reads 1.92 (rebuilt raw ≤ F_lo ≤ F at every energy; the raw
  table itself sits 0.0027 above F_lo at NCE 25). The rebuilt and the raw
  table differ by 0.0041 at NCE 1 and 0.0065 at NCE 5 (averaged S/N ≥ 3
  spectra of one isolation centre, against per-scan centroids pooled over
  both centres); at the table's slope there, 0.0147 per NCE, that bounds the
  E50 difference by 0.0065 / 0.0147 = 0.44 NCE — it is 0.32 (1.92 against
  1.61).
- **Standard B, [M+H]⁺** (137.13, summary refused): F E50 15.82, NCE only (volts
  and E_cm NaN, said); the raw-file breakdown read 16.3 (6.76 V). **Not** a
  difference of definitions: the raw-file survival (precursor / (precursor +
  the tracked ions)) rebuilt on the API spectra agrees with F within 0.0011 at
  every step and exactly at the crossing's two steps (NCE 10 and 15, where
  every fragment read is a tracked one), E50 15.821. The difference is in the
  spectra: per precursor, the API spectra hold ×1.021 and ×1.037 the tracked
  fragments the raw-file table does at NCE 10 and 15 (×0.963 at NCE 1). A
  fragment / precursor ratio off by 1 + d moves F by about F (1 − F) d ≤ d / 4;
  with every ratio within 4 %, F lies within 0.01 of the table and, at the
  crossing's slope (0.0169 per NCE), the E50 within 0.59 NCE — it is 0.50.
- **Standard A, per scan** (its server spectra read in every scan of their
  groups, `ms2 extract --per-scan`; 5–9 scans per energy): C10H17⁺ F E50 13.89
  as above, its 16–84 % band **13.84–13.94 NCE** (6.319–6.336 V, E_cm
  1.0716–1.0745 eV), every one of 1000 resamples crossing; the precursor's CV
  per spectrum 1.3–4.3 %, F's scatter from scan to scan 0.002–0.004 (§11.11).
  Rebuilt as a tMS2 file (`--file`, §12.6): its own E50 13.82 with a band about
  it.
- **2026-09-10 exploratory file**, the seven parents with the co-isolates that
  day's script settled: E50 C7H6NS⁺ above range (F 0.73 at NCE 91; its
  co-isolate C4H10NS2⁺ 15.5, spent by NCE 41), C11H13O⁺ 35.3, C11H15O2⁺ 21.1,
  C8H28^NO4Si4⁺ 1.78; the ammonium adducts C11H18^NO2⁺ and C12H22^NO2⁺ below
  range (F 0.44 and 0.38 at NCE 1). 88.0215 C3H6NS⁺ is the co-isolated
  C4H10NS2⁺'s by composition; C9H11⁺ in the 179.107 window is ambiguous (the
  guard, §11.5). No reading needed a fourth piece.

### 11.9 Gotchas

- **E50 on NCE is not E50 on volts.** NCE 1 is a different voltage at every
  m/z (§10.5); compare windows on `e50_v` / `e50_ecm` when the volts are known.
- **F of a co-isolate depends on its weights.** A minor co-isolate that shares
  big in-source fragments with the target gets their survivor share, which can
  dwarf its own peak; the F_lo..F band and the branching flag say so — the
  flag also where the co-isolate's own fraction never moved and its branching
  ratio is NaN (§11.5). When every fragment is shared, the survivor split sets
  every ion's F at the lowest energy to one number, and the E50 can lie above
  F's as well as below it (standard A's adduct: 6.1–10.1 across the splits
  considered, §11.5).
- **TIC normalisation reads ions you did not name.** The branching sums and the
  intact test are on TIC fractions; a co-isolate not named still moves them.
  Name every co-isolate with a settled formula — but the precursor's own side
  lobes (unexplained centroids a few mDa either side of it) move them too, and
  no name removes those (standard A's [M+H]⁺: half the excess, §11.5). A side
  lobe that reads as a sub-formula of a window ion is set aside as one only
  where its line reports a resolution (§11.5).
- **Name one formula of an alias pair.** Two named ions within the reading
  tolerance are not both read (§11.1): the one left out is in `left_out`.
- **One energy scale.** CID and HCD spectra of one parent are not one curve:
  the kind with the most spectra is broken down, the rest only read (§11.3).
- **S needs a steady precursor supply; F and F_lo need the products in the
  spectrum.** Orbitrap intensities are per unit injection time, so S is not a
  question of the injection time but of the supply (source, transmission,
  isolation) across the energies. F and F_lo hold whatever the supply did, but
  see only observed products (§11.3).
- **Products outside the observed m/z range are invisible** to F, F_lo, every
  TIC fraction, the intact test, the guard and the branching: an adduct whose
  main channel releases an ion below the scan's low-mass limit reads a high F
  (its E50 above S's), inflates every other fraction as the TIC falls (branching
  sums above 1, flagged — the note gives the TIC ratio), and can keep a
  fragmenting co-isolate's TIC fraction steady enough to be excluded as intact
  (§11.5). Compare F's E50 with S's, and read the TIC across the energies.
- **CF3COO⁻ reads as a nitrate-cluster fragment.** In 15N data C2F3 = H2^NO4
  within 0.21 mDa: CF3COO⁻ is H2^NO6⁻ and C3H2F3O2⁻ is CH4^NO6⁻ to every
  [M+^NO3]⁻ window ion, and the fragment gate admits both. Name the fluorinated
  co-isolate: its organic reading then outranks the polyoxy one (§11.5) —
  where the sample holds it. Unnamed, or absent from the sample, the cluster
  takes the peak by composition.
- **The ladder is per window ion and per window.** A step read by another
  window ion, or in another window, is no support.
- **A fragment name with ` | `** is one peak read as several compositions:
  by two window ions, or at two energies (§11.3). Its readings are in
  `ms2_readings.csv` under each formula; `mz_theo` is NaN.
- **The E50 band is the scans' scatter, not the method's error.** It says how
  far the E50 moves with another draw of the same scans; it holds nothing of
  the roles and weights (decided once, on the averaged spectra), the gas mass,
  the volts calibration, or a co-isolate not named. An energy of fewer than two
  scans, or whose every precursor / fragment peak is held, adds nothing to it
  (counted: `n_one_scan`); one of the two the crossing lies between (or one
  where only held peaks enter the curve) leaves the band too narrow, two leave
  it of zero width (`one_scan_bracket`, the note). Read off a few scans per
  energy, the band holds the true E50 less often than a 68 % band would
  (§11.11).
- **A real ion at m/z / 4 or m/z / 9 of a brighter line, holding less than the
  cap of it at every energy, is taken for its ghost** (§11.12): `--keep-harmonics`
  reads every peak as before. One that goes above the cap at some energy (at
  an S/N of `HARMONIC_CLEAR_SN` or more), or that some energy holds where no
  line's band does (its line gone there, or a ppm of jitter out of the band),
  is read at all of them; so is a true ghost that lands within the reading
  tolerance of such a product (as without the rule). One above the cap only
  near the S/N floor is read only where it is above the cap.

### 11.10 Code map

| function | role |
| --- | --- |
| `readings.gate_mode` / `read_window` / `ladder_gate` / `rerank` | the gate mode and read-at k_max; one window's readings across energies; the ladder gate; the re-ranking after a drop |
| `attribute.peak_roles` / `merge_fragment_peaks` / `intensity_matrices` | per (spectrum, peak) role; one fragment across energies; the I_X / I_fragment matrices |
| `attribute.satellite_fragments` / `set_aside_fragments` | the fragments that are side lobes of a brighter line; their peaks made unexplained |
| `attribute.fragment_mz` / `fragment_compositions` / `candidates_across` / `polyoxy_inorganic` / `shared_presence` | where a fragment sits (signal-weighted m/z); the formulas it can be there; its candidates (an organic reading outranks a carbon-poor polyoxy one); where every candidate the sample holds reads it (the intact test's onset) |
| `attribute.absent_ions` / `live_candidates` / `intact_test` / `share_weights` / `held_at_lowest` / `attribute_fragments` | the named ions the sample lacks; each fragment's candidates the sample holds; the intact test; the weights, per candidate; the guard's bound (precursor + the ion's own fragments at the lowest energy); the call per fragment (absent ions set aside, the onset past a stray, the guard) |
| `attribute.branching` / `branching_breach` | branching per energy; the mass-balance flag (also where the ratio is NaN) |
| `breakdown.survival` / `e50` / `e_cm` / `onset` / `ion_budget` | F, F_lo, S; E50 and its flag; E_cm; onset; the per-spectrum budget |
| `analyze.parse_ion_spec` / `read_ions_csv` / `collect_ions` | the named ions |
| `analyze.load_run` / `isolation_width` / `inventory_half_width` / `window_ions` / `drop_aliases` | the extract folder; the width; how far the extract's MS1 inventory reaches; a window's target and co-isolates; the aliases left out |
| `analyze.gate_note` | the manifest's words for the gate mode |
| `analyze.analyze_window` / `analyze_run` | one window; every window of a folder |
| `analyze.render_window` / `write_outputs` / `manifest` | the figure; the files; `ms2_analyze_manifest.json` |
| `readings.harmonic_ions` / `harmonic_ghosts` | the places a window reads as ions (in no line's band at some energy, or above the cap there clear of the S/N floor); the harmonic ghosts of one spectrum (§11.12) |
| `breakdown.e50_many` / `observed_ends` | `e50` of many curves at once (the bootstrap's resamples), with the two energies each crossing lies between; the first and last energy each curve is observed at (a resample's holes at an end) |
| `replicates.spectrum_scans` / `window_scans` / `spectrum_curves` / `spread` | a spectrum's peaks in its scans, on its averaged scale (a factor outside `SCALE_BAND`: held); a window's ion and fragment matrices per scan; F, F_lo, S per scan; sd and sem |
| `replicates.bootstrap_e50` / `window_seed` / `interval` / `zero_width` | the resampled E50s (and where each crossing, or each hole at an end, lies); a window's seed; the 16–84 % band and on which side a censored bound lies; a band of zero width |
| `analyze._replicates` | one window's replicate statistics and intervals |

### 11.11 Replicates and the E50 interval (`replicates.py`)

**When.** The folder holds the per-scan tables (§12.5): `ms2_scan_peaks.parquet`
— every kept peak of every spectrum read in every scan of its group, 0 where
a scan has none — and `ms2_scans.csv`. `ms2 extract --file` writes them for a
raw file's MS2, `ms2 extract --per-scan` for a sample's server spectra. Without
them `ms2 analyze` is what it was: every table and value as before, the new
columns blank (0 / NaN), no `ms2_replicates.csv`, and the run's notes say so
(`analyze.NO_PER_SCAN_NOTE`; tested on every committed fixture). Per-scan
tables that hold no row (skipped at extract) are noted and read as none.

**Decided once, re-read per scan.** The breakdown is decided on the averaged
spectra, exactly as §11.3–§11.5 say: which peak is which window ion's precursor,
which fragment (merged across energies, side lobes and harmonic ghosts set
aside), every fragment's composition candidates and share weights. Then the same
roles and weights are read in every scan j of every spectrum
(`window_scans`, `spectrum_curves`):

- I_X,j — X's precursor peak(s) in scan j; a fragment's intensity likewise
  (its peaks summed, absence = 0);
- F_j = I_X,j / (I_X,j + Σ w·I_frag,j), F_lo_j with every composition
  candidate's fragment given whole, S_j = I_X,j / I_X(E_min) with I_X(E_min) the
  averaged lowest-energy precursor; 0/0 is NaN, as in `survival`.

`ms2_breakdown.csv` gets each spectrum's `n_scans`, the precursor's CV over
them (`intensity_cv`), and the scan-to-scan spread of the three curves (`f_sd`,
`f_sem`, `f_lo_sd`, `f_lo_sem`, `s_sd`, `s_sem`; sd with ddof 1 over the scans
where the curve is observed, sem = sd / √n; NaN with fewer than two).
`ms2_replicates.csv` holds every item of every spectrum — each window ion's
precursor, each fragment the averaged spectrum holds — with its scans, how
many hold it, its presence (their share), mean, sd and CV (absence = 0);
`ms2_fragments.csv` the presence and CV of each fragment at the spectrum of its
largest TIC fraction (`presence_at_max`, `cv_at_max`).

**A ratio of means, not a mean of ratios.** The averaged F is F of the per-scan
MEANS — the averaged spectrum — not the mean of the per-scan F (which differs at
the third digit where the scans scatter; tested). Every peak's per-scan readings
x are put on its averaged height's scale, x · I_avg / mean(x)
(`spectrum_scans`): on an `--file` folder that factor is 1 (each spectrum IS the
mean of its scans, tested to 1e-12); on a `--per-scan` folder it carries the
server's height, which runs about 2 % below the per-scan mean on the median peak
(the per-scan mean is `scans.SERVER_HEIGHT_RATIO`, 1.020, times it) and varies
by peak: the factor is 0.89–1.01 on standard A's S/N-10 spectra, median 0.979
(tested). In both, the per-scan means are the analysed spectrum, and F rebuilt
from them is the analysis's F (tested). A spectrum's peak and its per-scan rows
are one peak when their m/z agree within `MATCH_DA` (1e-6 Da: the per-scan table
carries the spectra table's m/z, the same float; two peaks of one spectrum are
never that close). A peak no scan reads (or that reads 0 in every scan) is held
at its averaged height in every scan — its spread is not in the statistics — and
counted (`matched` False, in no scan: `n_present` 0, presence and CV NaN; a
window note) — and so is an item read over two peaks (a split peak, summed) one
of which is held: unmatched as a whole, `n_present` 0 though its other peak is
read (tested). So is a peak whose factor lies outside `SCALE_BAND` (0.5–2, both
ends in; a window note names the factors): its rows are not that peak read scan
by scan — one stray centroid in one of n scans reads it at a factor of about n,
and so scaled it would fake a CV of √n; per-scan rows read off another m/z axis
than the spectra's read a few peaks partly, at factors of several — while on our
own three files' S/N-3 spectra, the server's peaks read scan by scan, the factor
is 0.85–1.14 (tested, synthetic: a stray centroid's factor of 3.16 is held). A
curve that only held peaks enter scatters by nothing anyone measured: S has no
spread (NaN) where its precursor is held, F and F_lo none where something
entering them is held and nothing entering them — the precursor, a weighted
fragment (for F_lo a composition candidate's) — is read; one held peak among
read ones leaves the spread of those. A spectrum with no per-scan row at all is
held at its averaged spectrum (a window note), and counts among the energies
that add no spread; so does a spectrum of two or more scans whose every
precursor / fragment peak is held (a window note; the bootstrap draws its
averaged spectrum, as it does one with no row: the same band, tested). A
parent whose energies recorded different scan ranges is cut to the range all of
them recorded at extract (§12.4): its per-scan rows are those of the cut
spectra, so no product is read at one energy and unrecorded at another here
either (checked on cut parents: every per-scan row is a peak of its cut
spectrum).

**The E50 interval** (`bootstrap_e50`, `interval`). `--bootstrap N` resamples
(default `BOOTSTRAP_B` = 1000; 0 = none; fewer than `BOOTSTRAP_MIN`, 100, are
refused — the 16th / 84th percentiles of a few resamples rest on one or two of
them: of one, the 'band' is that resample's E50, of zero width and not holding
the averaged one; on standard A the bounds scatter between seeds by 0.017 NCE
at 10 resamples, 0.006 at 100 and 0.002 at 1000): each resample draws every
energy's
scans with replacement, as many as it has — all of a spectrum's peaks from the
same draw, a scan's precursor and its fragments together — rebuilds the curves
from the resampled means (S against the resample's own lowest energy) and reads
the same E50 rule off each (`breakdown.e50_many`: `e50` column by column,
vectorised over the resamples; tested equal to `e50` on random curves with
holes, every flag met). `e50_nce_lo` / `_hi` are the 16th / 84th percentiles of
the resampled E50s (`BOOTSTRAP_PERCENTILES`), on volts likewise (`e50_v_lo` /
`_hi`, when every crossing's volts are known) and E_cm from those; `boot_cross`
is the share of resamples that cross. Every resample with an answer is ranked,
each as what it is known to be: an interpolated E50 itself; one already below
0.5 at the lowest energy an E50 below every energy, one never below 0.5 one
above every energy; one that crosses across an unobserved energy (`gap`) an E50
somewhere between the two energies about the hole. A resample observes only the
energies the averaged curve observes, and where every scan it draws at the
curve's first energy holds nothing of the ion (no precursor, nothing it makes)
it does not observe it there: below 0.5 at the first energy it does observe,
its E50 is known only to lie at or below that one — a hole at that end, `gap`
too, never an E50 below every energy; the mirror at the last energy (tested,
synthetic: a co-isolate read in one of the two scans at the first energy, and
one whose fragment alone is read in one of the two at the last — a quarter of
the resamples are such holes, and both bounds are NaN, `gap`, not a bound read
off the crossing resamples; the ends are the averaged curve's own, not the
axis's: a curve the averaged spectrum does not observe at the first energy is
no hole there). A resample its holes leave ONE observed energy (a curve the
averaged spectrum observes at two energies, a hole at one of them) still
reads which side of 0.5 it lies on there, and is ranked as that: below 0.5 at
the averaged curve's first energy, below every energy; below 0.5 at a later
one, a hole in [−∞, that energy]; at or above 0.5 at the last, above every
energy; at an earlier one, a hole in [that energy, +∞]. One left no observed
energy (S, read against a lowest energy whose draws hold no precursor, is
nowhere) is a hole about every energy, [−∞, +∞]. So no resample drops out of
the ranking for its holes (tested: the two-energy versions of the windows
above read NaN, `gap`, on both sides — not the crossings' band, nor one of zero
width; and resample by resample on synthetic draws); the share 'with one
energy observed or none' is that of a curve the averaged spectrum itself
observes at fewer than two energies. A percentile that falls among
the resamples beyond the energies or the holes is NaN, and `e50_lo_censor` / `e50_hi_censor` say on which side — `below`
the lowest energy, `above` the highest (a lower bound can lie there too, when
most resamples stay above 0.5), or `gap` — never a bound read off the other
resamples alone; one that falls on an interpolated E50 whose rank a few hole
resamples could move is the outer of the two readings (the band widened, never
narrowed). A note is written where the resamples disagree with the averaged
curve (an E50 some of them do not cross, or a curve without one that some of
them do): the shares — crossing, below, above, across a hole, and with one
energy observed or none — add up to 100 %, and each NaN bound is named by its
side. An energy of **fewer than two scans** (one, or no per-scan row), or
whose every precursor / fragment peak is held, has nothing to draw from: it adds
no spread (`n_one_scan`; a window note naming the NCE of each one-scan energy,
another those with no row, another those held); a window with no energy of two
scans read has no interval (said). Where it is one of the two energies the
averaged crossing lies between — or the curve has no spread there (only held
peaks enter it, named 'no spread') — the band holds none of its scatter — too
narrow; with both, every resample reads the averaged curve there and the band
has zero width, no measurement: `one_scan_bracket` counts them, the note names
them and flags a zero-width band, and `ms2 analyze` prints both with the band.
S is read against each resample's own lowest energy, so its band rests on that
energy too: where it adds no spread (fewer than two scans, its precursor held,
no per-scan row), every resample's S shares one reference and the S band holds
none of its scatter — counted in S's `one_scan_bracket` (once, where it is one
of the two about the crossing) and named in its note as the energy S is read
against (tested). Whether S has a spread at an energy is its precursor's
(`intensity_cv` finite), not the per-scan S's: a precursor read in one of its
scans alone, the others empty, gives one finite S per scan (`s_sd` NaN), yet a
resample's S is its mean precursor, the empty scans 0, and moves with the draws
— no note calls that band too narrow (tested, at the reference and at an energy
about the crossing; F there, a ratio within the one scan that holds the ion,
does not move, and its note says so). On synthetic windows (NCE 1 / 10 / 20, 10 % scan-to-scan
scatter, five scans at NCE 10 and 20, 2000 datasets, 1000 resamples) the S band
holds the true E50 in 21.6 % of them with one scan at NCE 1, 48.2 % with two and
59.0 % with five; F, a ratio within one spectrum, does not rest on it (59–62 %
in all three). With no band (`--bootstrap 0`, or no energy of two scans)
`one_scan_bracket` is 0 and no E50 note names those energies; `n_one_scan`, a
fact of the window's scans, is kept, and so is the window note naming the
energies without spread (with `--bootstrap 0` without its 'nothing to the E50
interval': there is none; tested). A
band is of zero width when its bounds agree to `ZERO_WIDTH_REL` (1e-9) of their
size (`zero_width`), not only when their floats are equal: bounds read off
resamples of held readings differ in their last bits (6.810663806263728 against
…729; a resample's counts / n do not add up to exactly 1). Each
window draws from its own stream (`window_seed`: `BOOTSTRAP_SEED` and the parent
key), so its interval is the same on every run, whatever other windows the
folder holds or their order, and under every string-hash seed (tested: the two
standard-A windows that sort after the first read the same bands without it;
one process per seed). On standard A 1000 resamples scatter the bounds by
0.002–0.003 NCE between seeds about a band 0.10 NCE wide; a window's resamples
cost a millisecond or two (the analysis of a folder of a hundred-odd windows
takes seconds, the bootstrap a fraction of one). A resample's means are its
scans' counts times the scans' intensities: its memory is resamples × (scans +
peaks), never an array of every scan of every resample (tested).

**What the band holds.** A percentile bootstrap of a few scans: drawing n of n
scans with replacement spreads an energy's mean by √((n − 1) / n) of its
standard error, and the scatter itself is read off those few scans. So at a few
scans per energy the band holds the true E50 less often than a 68 % band would:
on synthetic windows (3 energies, 10 % scan-to-scan scatter of the precursor and
its fragment, 4000 datasets each, 1000 resamples) it holds the true E50 in 51 %
of them at 2 scans per energy, 54 % at 3, 59 % at 5, 62 % at 7 and 66 % at 20.
Read it as the scans' scatter at that many scans, not as a confidence interval.

**Measured (our own data).** Standard A per scan (§11.8; 4–11 scans per
spectrum): two scans of a spectrum, averaged over its pairs, hold every fragment
of its S/N-3 spectrum on the median spectrum (95.5 % at the 10th percentile; the
worst pair 67.5 % there, 50 % in one spectrum); F from two scans is within
0.0023 of F from all of them on the median spectrum (over its pairs and window
ions; 0.0018 for the targets alone), and the worst pair moves a target's F by
0.009, a co-isolate's by up to 0.019 (a co-isolate's F of 0.71 in the 155.143
window at NCE 1); the precursor's CV per spectrum is 4.1 % on the median
spectrum, its top fragment's 3.4 %.

### 11.12 Harmonic ghosts (`readings.harmonic_ghosts`)

An Orbitrap reads m/z off an ion's axial frequency f (m/z ∝ 1/f², to the
calibration's higher terms), and the transient of a strong line carries weak
**harmonics** at 2f and 3f: peaks at about m/z / 4 and m/z / 9 that no ion
made. Read against the window ions they are sub-formulas like any other peak.
Our own 2026-09-10 exploratory file shows one: the 179.107 window's precursor
C11H15O2⁺ (179.1068) carries its 2f ghost at m/z 44.7766 in the server's S/N-3
spectra at NCE 1, 5, 10 and 15 — −1.9 to −2.7 ppm of m/z / 4, 0.098 / 0.098 /
0.094 / 0.081 % of the line while the line falls from 3.21e5 to 2.49e5 (the
committed S/N-10 spectra do not hold it; fixture
`ms2_nh4_exploratory_179_sn3.json`). A 3f ghost holds a few per cent of its line
and shows wherever m/z / 9 lies above the scan's first mass; 4f (m/z / 16) was
not seen. How far a ghost sits off m/z / n² depends on the calibration and on
the m/z (tens of ppm on some files, of either sign).

**The rule.** In each spectrum, for every line L (not only the precursor: a
co-isolate's and a strong fragment's ghosts are as real) and each order n of
`HARMONIC_ORDERS` (2f, 3f), the peaks within `HARMONIC_PPM` (40 ppm) of m/z(L) /
n² that hold at most `HARMONIC_CAP` of L's intensity in the same spectrum (2f:
0.5 %, five times our own ghost's; 3f: 8 %, a few per cent with room for a ghost
read near the S/N floor, where only the noise's upswings are seen) — of several,
the brightest (one line, one ghost per order; a second, weaker peak in the band
is something else). Above the cap a peak is an ion, whatever its place (real
ions at m/z / 4 of a line hold a few % to tens of % of it) — and at every energy
of the window. A ghost exists only beside its line and holds a constant share of
it, so a place some spectrum of the window reads as an ion (`harmonic_ions`) is
an ion's in all of them, and a peak within the reading tolerance of it is no
candidate in any spectrum (the tolerance as `subformula` reads it, max(5 ppm,
0.3 mDa): below m/z 60 — every 2f place of a line under m/z 240 — the 0.3 mDa
floor; tested, synthetic: C2H5O⁺, 24 ppm below m/z / 4 of C11H18NO⁺, read 6 ppm
high at 0.3 % of the line at one energy and above the cap at the others, is one
place, read at every energy). A spectrum reads a place as an ion where no line's
band holds it there (its line gone at that energy — a fragile precursor whose
products are the spectrum at the higher energies —, a product that a ppm of
jitter puts just out of the band, a spectrum of that peak alone), or where it
holds more than the cap of every line whose band does, read at an S/N of
`HARMONIC_CLEAR_SN` (10) or more (or none known). Near the S/N floor a peak is
seen only where the noise lifts it, read up to about two noise sd high — its
share of its line up to s / (s − 2) of the true one at S/N s: twice it at 4, 1.25
times at 10 — and that is where a true ghost reads above the cap (at the highest
energies, its line fallen); such a reading clears nothing, else one noisy energy
would switch the rule off for the ghost at every energy (tested, synthetic:
C15H26NO4⁺'s 2f ghost at 0.3 % of its line at three energies and 0.6 % at S/N
3.4 at the fourth stays a ghost at the three; at S/N 24 it is read at all four;
and the threshold itself: read at S/N 7.1 it stays a ghost, at S/N 12 it is
read at all four).
A product that grows from under the cap at its first energies to above it later,
or whose line is gone at the later ones, is so read at all of them — never half
erased, its onset, in-source flag and the intact test those of the reading
without the rule — and of the band's other peaks the brightest is the ghost, not
the product where it outshines the line's true ghost (tested, synthetic: a urea
adduct [M + urea + H]⁺ at 549.365 whose protonated urea CH5N2O⁺ sits 14 ppm
below m/z / 9 and holds 1 % and 4 % of it at the two lowest energies — with the
adduct beside it at every energy, and gone from the third on; a product 39.6 ppm
off m/z / 4 at one energy and 40.4 / 40.6 ppm at the others; its m/z moving by
a ppm between energies; a product just above the cap at one energy only). The
m/z are a ratio
within one spectrum: the axis offset cancels. A ghost is **never read**: its
rows in `ms2_readings.csv` become one unexplained-like row whose `harmonic`
names the line, order, offset and share, before the ladder gate (no ladder step
either); its role is `harmonic` — in no fragment, no attribution, no survival
sum, and **out of the TIC** (the line seen again, not an ion: counting it counts
the line twice), counted in `n_harmonic` / `harmonic_intensity`, a window note
and the manifest (`harmonic_peaks`, `harmonic_by_order`). `--keep-harmonics`
reads every peak as before (debugging). The energies only ever clear a place,
never confirm a ghost (a ghost's share of its line holds while the line falls,
0.098 → 0.081 % above, but a confirmation across energies would miss the ghost
of an energy that alone shows it): the band and the cap flag nothing off a
file's ghost band on any file checked, and no ghost flagged there sits where
another energy holds a peak above the cap. What the clearing costs: a true ghost
within the reading tolerance of a product that another energy reads as an ion is
read as that product at every energy, as it was without the rule; and a product
in a line's band that is above the cap only near the S/N floor is read only
where it is above the cap (a true ghost and it read alike there).

**What it changes, and what it costs.** A ghost's mass defect is its line's
over n², and rarely any formula's: no ghost on any file the rule was checked
on was read as a formula before, and no reading was flagged. So F, F_lo and
every E50 stay as they were there; what moves are the TIC fractions — the
intact test's, the branching sums', `frac` — by the ghost's share (in the
179.107 window C11H15O2⁺'s branching sum at NCE 15 drops from 1.0506 to
1.0488, under the flag). Where a ghost does land on a sub-formula it would have
been a fragment — tested, synthetic: C15H26NO4⁺'s 2f ghost 34 ppm above m/z / 4
reads C4H7O⁺ (−4.5 ppm), an "in-source fragment" of 0.3 % that set F 0.997
where nothing fragments; read as a ghost, F is 1. Of 1152 even-electron ions
C6–21 H O0–7 N0–1 (a synthetic search), one has a sub-formula read at exactly
m/z / 4 or m/z / 9, and 90 one somewhere inside the ±40 ppm band. The band is
also what lets a real weak ion be taken for a ghost: one at m/z / 4 or m/z / 9
of a line it holds less than the cap of at every energy of the window. On our
own three files the rule flags the four ghosts above and nothing else.

---

## 12. Per scan — `peaky ms2 extract --per-scan | --file`, `peaky ms2 files`

**Code:** `peaky/ms2/scans.py` — PURE: the scan table, the method layout, the
precursor mapping, the headerless-scan inference, the averaging; and in
`peaky/ms2/sources.py` the orchestrators `extract_sample(per_scan=True)` and
`extract_file`, the polarity rule (`resolve_polarity`) and the file listing
(`needs_check`, `check_file`), on three new fetchers in `peaky/io/io_mascope.py`
(`fetch_file_metadata`, `fetch_sample_files`, `fetch_ms2_timeseries`). Read-only.

### 12.1 What it is for

The MS2 endpoints (§10) serve averaged spectra only, and only for the scans whose
scan filter the server's raw reader can parse. A raw file's `/metadata` holds
more: every scan's centroids and, for every scan the server reads as MS order
`Ms`, the instrument's **trailer**. Two uses:

1. **Targeted-MS2 files the server cannot read.** On an Orbitrap Exploris file
   recorded with an inclusion-list (tMS2) method, the server's reader garbles
   the scan filters: random analyzers, ionizations and polarities (`ITMS - c EI`,
   `TQMS - c EI`, `SQMS + c FABI`, `Sector - c`), every MS2 scan read as
   `Full ms [40.0000-<HighMass>]` with no `ms2 <m/z>@hcd<E>`. The listing reads
   polarity `+-` and processing status `needs_chemistry`: no sample item, no MS2
   endpoint, nothing §10 can read. The trailer is right on every row, so
   `ms2 extract --file` rebuilds the MS2 from it.
2. **The scans behind an ordinary file's spectra.** `ms2 extract --per-scan`
   reads which scans each server spectrum averages and every server peak in each
   of them: the basis of `ms2 analyze`'s replicate statistics and E50 intervals
   (§11.11). The spectra stay the server's, so every known answer of §11 is
   unchanged.

### 12.2 The payload

`GET sample/files/{sample_file_id}/metadata` (the sample record's
`sample_file_id`; 6.7 / 8.6 / 19.8 MB for our own files of 140 / 233 / 434
scans, so a 300-s read timeout, `METADATA_TIMEOUT`):

| key | what it holds |
| --- | --- |
| `num_of_scans` | N |
| `stats_per_scan` | `{"<scan, 1-based>": row}` — a row ONLY for a scan the server reads as MS order `Ms`: on an ordinary MS2 file its MS1 scans (our 2026-09-11 monoterpene standard A: 61 rows of 140 scans; its 78 MS2 scans and the first scan have none); on a garbled file the MS2 scans too (their garbled order reads `Ms`), a fraction of the scans without |
| row keys read | `ScanType` (the filter: never trusted), `LowMass`, `HighMass`, `TIC`, `Scan Event:`, `Scan Node:` (`Full Scan` / `tMS2` / `''`), `HCD Energy:` (the NCE; `''` on MS1), `HCD Energy V:`, `MS2 Isolation Width:`, `Ion Injection Time (ms):`, `Max. Ion Time (ms):`, `Micro Scan Count:`, `AGC Fill:`, `FT Resolution:`. `Monoisotopic M/Z:` is 0 on every row of these files: no precursor source |
| `centroids_meta` | `time` (N starts, s) and `data` (N × `{mzs, intensities, noises, resolutions}`) |

`GET sample/files?instrument=&datetime_min=&datetime_max=` lists an instrument's
raw files: id, file name, `length`, `polarity`, `processing_status` (`done`, e.g.
`needs_chemistry`, or none: our own server lists no status) and its detail. The
times are UTC (compared with the files' `datetime_utc`; `--since` / `--until` with
an offset are converted). The server filters by instrument only when one is
named, so an empty `--instrument` is refused. It is asked page by page
(`io_mascope.FILE_LIST_PAGE`, 500 files, sorted by the unique file id so no page
skips or repeats one, `FILE_LIST_PAUSE_S` 1 s apart) until a page comes back
short: asked whole, a busy instrument's day was one 3.5 MB answer. At
`FILE_LIST_MAX_PAGES` (200) full pages the listing is an error
(`FileListTruncated`: narrow the range) — the files beyond are anywhere in the
range, the ids being random — and when the server counts more files in the range
(its answer's `results`) than its pages held (a file recorded while they were
read), `ms2 files` says so. The time series of a
spectrum (`get_timeseries(parent key, activation=...)`) names its scans by their
`time` (ISO strings from the epoch).

### 12.3 The rules

**The trailer, never the filter.** Each scan's MS order is its trailer's: MS2 when
`HCD Energy:` is a number, MS1 when it is `''` on a `Full Scan` or `''` node. A
row with no HCD energy on another node (e.g. a `tSIM` experiment) keeps its order
unread: it is no Full Scan (not in the MS1 average) and no MS2 scan (in no
spectrum), said in the mapping's notes, and the cycle walk never crosses it
(below); its filter is checked as an MS1 one (`SIM ms`). A
filter is used only where it agrees with its trailer (`filter_ok`: an Orbitrap
analyzer, one polarity, the trailer's order and NCE, its scan range); the file's
verdict is `yes` / `no` / `partly` (`filter_verdict`). Polarity is read off the
filters only when every one agrees, with one sign; a garbled file's signs are
random and its listing reads `+-`, so `--polarity` is required there
(`resolve_polarity`: the explicit sign, checked against a one-sign record and
readable filters; else the file record's; else the filters'; else refused).
`ms2 extract --file` asks the server for the file's `/metadata` only, never its
listing record, so there the record is not consulted: the polarity is
`--polarity`'s or the readable filters'. `ms2 files --check` writes the
listing's sign into the command it suggests when the listing has one sign; a
garbled file's listing reads `+-` (and a file named with `--check` outside the
listed range has no listing row), so there the command carries `--polarity <+|->`
and the note asks for the method's sign. A file whose filters are all readable and
carry both signs, with MS2 scans read by trailer, is a dual-polarity method: its
spectra would average every MS2 scan and its window every MS1 scan whatever their
sign, so it is refused whatever `--polarity` says (`DualPolarityError`; the check
gives no command). An ordinary dual-polarity file (no MS2 trailer row) reads `MS2
without trailer`.

**The method layout.** On an Exploris inclusion-list method the MS2 experiments'
events run from 1 in experiment order, one per list row: one shared list of n rows
over K experiments (one NCE each) gives events 1..n, n+1..2n, …; the Full Scan
is the last event, K·n + 1. Every event's trailer NCE is then its
experiment's. Several lists = one per
experiment, in event order. A list that does not fit is refused through the Full
Scan event (it must be K·n + 1, or the lists' summed rows + 1: the refusal says how
many MS2 events the method has and, from its NCE runs, the list that would fit),
then through one NCE per experiment. A shared list whose row count only divides
the method's (3 rows for a 6-row method passes both checks) splits each of the
method's experiments in two: two of its experiments then carry one NCE and their
scans alternate in time (their known scans overlap in time), and that is refused
too — unless the scan range follows the precursor in both and the list fits the
scan-range rule on every event of theirs seen: the rule has then checked the rows
(a divisor list misses it by the spacing of its rows), and the two are two
experiments of one NCE run in parallel (each cycle through one, then the other;
noted). Where the range follows the precursor in both and the list misses the rule,
the refusal names the events the rule puts off the list (its rows are wrong: two
parallel experiments run with it would have fitted). On a fixed range nothing
tells the two apart: refused, the message saying that a parallel method takes the
list once per experiment (`--inclusion-list L.csv L.csv`, which maps it). Two experiments of one NCE that run one after the other,
in either order, are valid (time order is free: the Full Scan, the last event,
runs first). A list maps an inclusion-list method's numbering (the MS2 events
from 1, the Full Scan last): a Full Scan numbered before the MS2 events is refused
saying so, and a file with no MS2 trailer row at all (an ordinary MS2 file) has no
event to fit a list to — the list is not used (noted) and the file reads `MS2
without trailer`. The layout is **closed** only when a list and the Full Scan event pin every
experiment's range. Without a list it never is: the experiments are then the NCE
runs of the events seen (sized by the Full Scan event when it splits them evenly),
and two adjacent experiments at one NCE read as one run, an experiment never seen
as none — a 'one 8-row experiment' that a walk would wrap through where the method
ran two 4-row ones. On an open layout the cycle walk is restricted (below).

**Precursors**, per event, from: a readable filter (a scan's own readable
filter wins: a data-dependent event names a new precursor each cycle — mapped scan
by scan on a fixed range; where its scan range follows each precursor the event
reads several ranges and is refused, one event being one scan range) > the list
(event → row) > the scan-range rule (no list). The **scan-range rule** of a
"define first mass" method: HighMass = 1.02 × (precursor + isolation width / 2) +
20 Da, to a few hundredths of a mDa (the width term is the method's documented
one, not a fit). The scan range is a setting of each
experiment, so the rule is judged **per experiment** (`rule_verdicts`): it
follows the precursor where two or more events were seen and their HighMass
values differ, it is a fixed range where every event reads one HighMass, and an
experiment with one event seen counts as following only when the file's HighMass
values differ and no experiment reads a fixed range. Nothing is derived in a
fixed-range experiment (its precursors would all coincide), nor for an event
whose HighMass another event of its experiment shares: without a list that is a
refusal naming the events ("pass the method's inclusion list"). A list is ALWAYS
checked against the rule where it holds: a residual beyond 0.5 mDa
(`RULE_TOL_MDA`, 20 × the largest measured) is an error naming the events — a list
in the wrong order, the wrong list, a row missing. Where it does not hold, a list
(or readable filters) is required, and is checked against the Full Scan event
(when the method has one), the experiments' NCEs and any readable filters, not
against the scan range; where it cannot be judged (one event seen), a misfit is
noted, not refused. `--no-rule-check` skips the check for a method that does not
set its range by the precursor. Precursors within 1 mDa are
one parent, keyed by their median rounded to 4 decimals (the server's own
grouping); the spectrum key is the server's, `<parent>@hcd<NCE:.2f>`.

**Scans with no trailer row** are placed by walking the scan cycle
(`infer_headerless`): inside an experiment's time block the instrument walks its
events in order, cyclically; at a block switch the next block starts at its first
event. The model is first checked on every pair of consecutive known scans (one
pair it does not allow, or a block the known scans return to, and nothing is
inferred). A run between two known scans takes the ONE walk the cycle allows;
none or several (e.g. a block of one event at a switch) leaves it unassigned. A
run across a block switch, and a leading or trailing run, are walked only on a
closed layout whose every block has a known scan and with no trailer row of a scan
of no block (below): a block of the list's experiments cannot hide in the run, but
a trailer row with no MS order or no event shows an experiment outside the layout
(e.g. a `tSIM` one) running in the file, and a block of it whose scans lost their
rows could — such runs stay unassigned (`runs_beside_outside_scans`, said), runs
inside one block are still placed. An experiment outside the layout none of whose
scans keeps a row is invisible (§12.7);
on a layout that is not closed only a run inside one block, straight from one
known scan to the next (no wrap, exactly as many scans as events between them), is
placed. A gap holding a scan of no block (a trailer row with no MS order or no
event, e.g. a `tSIM` scan) is not walked: its headerless scans stay unassigned
(`runs_blocked_by_header_rows`), and only scans with no trailer row are ever
written. Every walked scan is checked against its event's LowMass / HighMass — a
Full Scan's against the Full Scans' (a centroid outside it leaves the run
unassigned); an event never seen with a
trailer takes its experiment's range where that range is fixed (and keeps it),
or — where the scan-range rule holds and is checked — the HighMass the rule
predicts from its list row (a check only: the scan keeps no range it never
recorded); else the run is unverifiable. A seeded fuzz of random methods (1–3
experiments, shared or own lists, a Full Scan block or none, rows lost at random,
events never seen, adjacent experiments at one NCE, an event skipped once, a short
block hidden inside another — but not one whose length completes the cycle it
interrupts, which nothing in a file can tell apart, §12.7) places no scan on
another event or spectrum. Every
headerless scan is counted, placed or not, and why. Nothing is inferred on an ordinary file (no MS2 event has a trailer row:
a walk over its MS1 rows alone would call every MS2 scan a Full Scan).

**The server's groups, scan by scan** (`--per-scan`). Each time of a spectrum's
time series names one scan: on our own files the time and the scan's
`centroids_meta.time` agree to < 1 ms on every MS2 scan (`TIME_MATCH_S`), none on
an MS1 row, none claimed twice. Scans with no trailer row that no group holds are
counted and noted — the first scan (dropped by the server's bad-first-scan rule)
and, after the record's `t1`, a hidden tail's MS2 scans (§10.5), which have
neither a trailer nor a group to name their precursor or energy. The server
multiplies the m/z it serves by the file's calibration factor (the sample
record's `mz_calibration.par.calibration_factor`), while `/metadata`'s centroids
are the raw axis: each server peak is read in the scans at its m/z / factor and
keeps the server's m/z (`sources.calibration_factor`; unmatched, a 3 ppm factor
left 45 of standard A's 230 peaks with no centroid in any scan, 5 ppm 163). A
server peak that still reads 0 in every scan of its spectrum is counted
(`per_scan.peaks_all_zero`) and noted.

### 12.4 Averaging (`--file`)

A group's scans are pooled and clustered by m/z in two steps: a seed bin opens at
every gap > 1 ppm (`SEED_PPM`), then adjacent seed bins merge where their gap is
< 0.5 × the narrower FWHM (`MERGE_FWHM`; FWHM = m/z / the bin's intensity-weighted
resolution). Per cluster: m/z and resolution intensity-weighted over the
centroids; intensity the mean over the group's N scans, a scan without the peak
counting 0 (two centroids of one scan in one cluster are one reading of that
scan); S/N = the intensity-weighted mean per-scan S/N × n / √N (n the scans
holding the peak); kept at S/N ≥ `--noise-threshold` (3). This reproduces the
server's S/N-3 spectra of our own monoterpene standards A and B and the
2026-09-10 exploratory file (47 spectra, 893 peaks): 892 matched, 2 extra (both at
S/N 3.00–3.01), m/z identical (891 within 0.01 ppm), resolution identical, S/N
1.005 × on the median (5–95 %: 1.001–1.010). A flat gap does worse (3 ppm joins
side lobes to their line, 1 ppm splits jittered peaks). The intensity runs
1.020 × the server's on the median peak (5–95 %: 1.000–1.07) — the server re-reads
heights off an averaged profile the payload lacks; reported
(`SERVER_HEIGHT_RATIO`), never corrected. Every scan's centroids carry a
resolution of their own (measured, not the instrument's setting); the nominal
R200 · √(200 / m/z) from `FT Resolution:` is used only where a payload has none
(the centroids' own resolution reads 1.033 × the nominal on strong peaks and
0.73 × it on S/N 3–10 ones) — the R200 of the scans being averaged (a group's
trailer rows, else the MS2 scans'; the MS1 scans' for the MS1 average), never one
for the whole file: a Full Scan and its tMS2 scans are often set apart (e.g.
120000 / 60000). A payload where only some scans carry resolutions takes the
nominal value for the others (noted): a seed bin with no resolution has no width
and is joined to no neighbour. It is counted per average — the scans of the MS2
spectra, and the MS1 scans apart — and a nominal value is recorded only where a
scan of that average lacked resolutions: the manifest's
`averaging.resolution_source` (with `scans_without_resolutions` and
`nominal_r200`) says it for the MS2 spectra (`per-scan centroids`, `per-scan
centroids; nominal (FT Resolution …) where a scan carries none`, `nominal: FT
Resolution …`, or `none`, then noted: no peak is merged by width and the
side-lobe rules are off), `averaging.ms1_resolution_source` (with
`ms1_scans_without_resolutions`) for the MS1 average, each noted as its own MS
order's (MS1 scans with none and no single FT Resolution on their rows: the
window's peaks of theirs read NaN, said). The window inventory (§10.4) averages the file's MS1
scans the same way (every cluster kept, no S/N floor, as the server's MS1
centroids) at each parent's trailer width (`--iso-width` replaces it). The MS1
scans are averaged only where they are read — the window, and the axis offset
when the MS2 precursors read too few (below) — and never for a file with no MS2
parent; every sum runs over the (cluster, scan) pairs that hold a centroid, so
memory grows with the centroids, not clusters × scans (an MS1 average keeps every
noise cluster, ~55 new ones per scan, and as a dense matrix it grew ~N^1.7 with the
scans, over a GB for a few hundred MS1 scans).

**One range per parent.** The scan range is per scan, and an experiment's first
mass can differ from the others' (an automatic first mass in one block, the
method's in the others): a product between the two reads as absent at that energy
only, and its F jumps there. When a parent's spectra differ in the range their
scans recorded (by more than `RANGE_CHECK_DA`), every spectrum of that parent is
cut to the range all of them recorded — said on its rows ("scan range differs
between this parent's energies (first mass 40.00 at NCE 5/30, 51.30 at NCE 15):
every spectrum of it is cut to m/z 51.30–…"), the peaks removed counted in the
manifest (`averaging.range_cuts`, every spectrum's range in
`averaging.scan_ranges`).

**The m/z axis.** A raw file's centroids are the instrument's own axis, which the
server calibrates for the spectra it serves but nothing calibrates here: it can
sit tens of ppm off the method's precursors, enough for the heavier parents'
precursors to fall outside the search window of §10.3 and a side lobe to be taken
for them. `sources.axis_offset` reads the offset off the file, against
the method's TYPED precursors (the keys): per parent, at its lowest energy, the
strongest centroid of its isolation window is its precursor when it lies within
`AXIS_SEARCH_PPM` (50 ppm) of the key; the offset is the median over the parents
(≥ `AXIS_MIN_PARENTS` 3; else the MS1 window's parents; else the few there are,
said) — the median out-votes a parent whose window a co-isolate dominates. With
two such few, nothing out-votes it (their median is their mean): a reading is kept
only where every other reading parent's window bears it out — a precursor
candidate (≥ 10 % of its window's strongest) within the track tolerance of its key
× (1 + that reading); the others are dropped (`unconfirmed`, said), and with none
borne out no offset is read (a synthetic two-parent file 12 ppm off, one window's
co-isolate 4 mDa below its target outshining it: the mean of its −41 and −12 ppm
readings, −26.4, left `ms2 analyze` matching neither window; the rule reads
−11.6). A
reading is **contested**, and dropped (`contested_keys`; the reading kept in
`contested`), when its strongest centroid lies off the key while another precursor
candidate (≥ 10 % of it) sits ON the key: a peak on the key is evidence against
an axis shift. A one-parent file whose co-isolate, 29 ppm below the key, outshone
the target at the lowest NCE read −29 ppm off it, its search was centred on the
co-isolate, and `ms2 analyze` took the co-isolate for its target; contested, no
offset is read (said), the target is located on its key. One parent cannot tell
that from a file whose axis does sit off and whose key holds an unrelated ion:
there the ion on the key is located — so the contested reading's ppm is quoted in
the extraction's note and on that parent's rows ("if the m/z axis does sit … off,
the ion on the key is a co-isolate"). **Only a key typed finely enough reads one** (`AXIS_KEY_PPM`, 2 ppm: half a
unit of the key's last decimal over its m/z, `sources.key_rounding_ppm`). A key
carries the precision it was typed with (a list row, or the precursor the rule
derives from a HighMass that row set), and one typed with two decimals sits up to
5 mDa from its ion by that alone: an on-axis synthetic file whose list read
117.09 / 155.14 / 187.13 / 199.17 / 217.14 gave +8.5 / +19.2 / +15.3 / −3.1 /
+15.6 ppm — a "median offset" of +15.3 ppm that was the typing, and `ms2 analyze`
then skipped every window (our own standard B's key 137.13 reads +16.6 ppm the
same way). Such keys are counted (`coarse_keys`), never read: 4 decimals qualify
from m/z 25, 3 from m/z 250 (a 4-decimal key ending in 0, 117.0910 → 117.091,
reads as 3); with every key left out no offset is recorded, and the notes say so.
On our own standard A (keys 137.1321, 155.1429, 155.1559) the offset reads
+0.5 ppm. The precursor search and the MS1 window's `is_parent` are centred on each
key × (1 + offset); an offset of `AXIS_NOTE_PPM` (3 ppm) or more is noted on every
parents row; the manifest records the offset (`parameters.ms2_axis_offset`), and
`ms2 analyze` reads the folder through it (§11.1). On such a file the precursors
and the E50s are then those of the same file with every m/z divided by
(1 + offset). With the offset measured, a key typed finely enough looks within
the track tolerance of its key × (1 + offset) first: its own ion there is the
precursor, however much brighter a neighbouring list row's ion a few mDa away in
the same window is (the wide search's 10 % share rule would drop the weak one and
give both parents the bright ion); a key typed coarsely keeps the wide search.
The key-offset note (`KEY_OFFSET_NOTE_MDA`) is read against the key × (1 +
offset), and two parents whose located precursors coincide (their windows
overlap, one ion read for both) are said on both rows, on any folder. The MS1
window's `is_parent` is the MS1 centroid at the precursor the MS2 spectra located
(within the track tolerance), none when the MS1 scans hold nothing there — said
on that parent's rows (the MS1 scans did not see it: its MS1 share there is
unseen, not measured absent), as is a window past the MS1 scans' recorded m/z
range (not measured). The manifest's `per_scan` records that range
(`ms1_scan_range`) and the MS1 and MS2 scans' time spans (`ms1_time_s`,
`ms2_time_s`), and MS1 scans that all precede the MS2 scans are noted (the window
inventory is the composition before the MS2 ran).
Where no offset is measured, the side-lobe guard (§10.3) is on: a precursor whose
only candidate is a lobe is not located.

### 12.5 Commands and outputs

```bash
# the scans behind a sample's server spectra (+ one /metadata, + one time series per spectrum, 1 s apart)
peaky ms2 extract --sample-id <ID> --per-scan [--save-metadata]
# a raw file's MS2 rebuilt per scan (a garbled tMS2 file needs --polarity; its list checks the mapping)
peaky ms2 extract --file <FILE_ID> --inclusion-list <list.csv> [<list2.csv> ...] --polarity + \
    [--save-metadata] [--noise-threshold 3] [--iso-width W] [--no-window] [--no-rule-check]
peaky ms2 extract --file <FILE_ID> --metadata-json <ms2_file_metadata.json> ...   # offline, no credentials
# an instrument's raw files (times in UTC); --check reads the /metadata of the ones that need it,
# the ones the server built no sample from first, then the longest (at most --max-check, 5)
peaky ms2 files --instrument <NAME> --since 2026-09-11T10:00 [--until ...] [--check [FILE_ID ...]] [--out files.csv]
# a --file folder whose m/z axis sits off: analyze reads the offset its manifest records, or
peaky ms2 analyze --run-dir <folder> --ion MZ=FORMULA ... [--mass-offset-ppm PPM]
```

The folder (`<out-dir>/ms2/<sample or file id>/`) holds §10.4's four tables and
manifest plus:

| file | content |
| --- | --- |
| `ms2_scans.csv` | one row per scan (`scans.SCAN_COLUMNS`): `scan`, `t_s`, `has_header`, `order`, `event`, `node`, `nce`, `hcd_v`, `iso_width`, `low_mass`, `high_mass`, `inj_time_ms`, `max_inj_time_ms`, `microscans`, `agc_fill`, `tic`, `ft_resolution`, `filter` (as written), `filter_ok`, `filter_parent_mz`, `n_centroids`, `min_centroid_mz`, `max_centroid_mz`, `inferred`, `parent_mz`, `precursor_source` (`filter` / `inclusion list` / `scan-range rule` / `server`), `experiment`, `slot` (the list row, 1-based), `spectrum`, `note` |
| `ms2_scan_peaks.parquet` | every kept peak in every scan of its spectrum (`scans.SCAN_PEAK_COLUMNS`): `spectrum`, `peak_mz`, `scan`, `t_s`, `intensity` (0 where the scan has none: the mean over a peak's rows is its spectrum's intensity with `--file`), `n_centroids`; `ms2 analyze` reads it (§11.11) |
| `ms2_file_metadata.json` | `--save-metadata`: the payload as the server answered it (kept also when the scans cannot be mapped: re-run with `--metadata-json`) |

The manifest gains `source` (`server MS2 endpoints` / `… + the file's per-scan
metadata` / `file metadata, per scan`) and `per_scan`: with `--per-scan` the
matched and asked times per spectrum, the scans with no trailer row in no group
and those of a hidden tail, the filter verdict, the calibration factor the
server's peaks were matched through (`mz_calibration_factor`) and the peaks
that read 0 in every scan (`peaks_all_zero`); with `--file` the layout, the
experiments, the filter verdict, the rule fit (`slope`, `offset_da`,
`max_residual_mda`, `events_off`), the inference counts, the precursor sources,
the averaging rule and every tolerance (`scans.tolerances`). With `--file` its
`server` block is null (nothing is the server's), `parameters` carries the
polarity and where it came from, the lists (base name, rows, and the file's sha1
and bytes: the list decides the mapping, so the run names it by its content),
`rule_check`,
the m/z axis offset (`ms2_axis_offset`: ppm, source, parents, spread, per parent,
the keys typed too coarsely to read it, the readings contested by a candidate on
their key (`contested`: key → ppm) and those no other parent's window bore out
(`unconfirmed`)) and whether the side-lobe guard ran
(`precursor_lobe.guard_on`: only where no offset was measured),
the payload read with `--metadata-json` (`metadata_json`: its base name, sha1,
bytes, `offline`), and the isolation width: the trailer's when it is one, else
`null` with `isolation_widths` per parent — and `ms2 analyze` then reads each
window at its own parent's width (`analyze.PER_PARENT_WIDTH`). `ms2 analyze`
reads a `--file` folder as it is. Every file of the folder is in its manifest: a
re-run removes the per-scan tables it does not write and a kept
`ms2_file_metadata.json` it neither keeps (`--save-metadata`) nor reads
(`--metadata-json` on that file, then listed among the files). The code identity
names `scans_version` beside `sources_version` and `io_mascope_version`.

`ms2 files` prints the listing (every file up to 40; beyond, the ones that need a
check) and `--out` writes every file (`sources.FILES_COLUMNS`). A file needs a
check when its polarity is not one sign or its processing status is set and not
`done` (a file with no status is not flagged: our own server lists none).
Without ids `--check` never fetches a routine file; a file it names is read
whatever its listing says (`--max-check` caps both). Its own picks are, first, the
flagged files the server built no sample from (a status other than `done`: a garbled tMS2
file reads `needs_chemistry`), then the others (a dual-polarity file it processed
reads `+-` and `done`) — within each the longest first (an MS2 run holds minutes
of scans, a short composite or calibration file seconds), the newest among equals
— at most `--max-check`, and says so where the cap leaves some unchecked: each
check reads the whole raw file's `/metadata`, about 0.1 MB per MS1 scan and 5 kB per
MS2 scan (our own files: 6.7–19.8 MB for 140–434 scans), so an hour of MS1 scans is
hundreds of MB. A file named with
`--check` that the listed range does not hold is checked all the same, its row
marked `listed` False. A checked file gets its scans, trailer rows, MS2
scans by trailer, headerless scans, filter verdict, events, NCEs, isolation
widths, the scan-range rule's verdict (per experiment where they differ), what a
mapping without a list gives, and the `ms2 extract --file` command line (with
`--polarity` when its filters name none — the listing's sign when it has one, else
`<+|->` and a note — and `--inclusion-list` when it is needed: required when a
fixed range leaves events without a precursor, or for a file with headerless
scans, whose layout only a list pins; advised for rule-derived precursors). A
file refused for a reason no list lifts (a stepped energy, two MS1 events) gets no
command, its reason in the note; an unreadable payload fails that file's check
alone. An ordinary MS2 file reads `MS2 without trailer`: extract its sample with
`--per-scan`. A raw file's failure (`--file`, `--check`) points to `ms2 files` for
its id, never to the sample listing.

### 12.6 Known answers (our own data)

- **Standard A** (2026-09-11; fixture `ms2_monoterpene_a_metadata.json`): the
  time series name 78 of its 78 MS2 scans, each group's count the server's; the
  one scan with no trailer row in no group is scan 1. Our averages of those
  groups reproduce all 230 peaks of the server's S/N-3 spectra (nothing extra),
  m/z within 0.005 ppm, resolution identical, S/N 1.0015–1.011 ×, intensity 1.020 ×
  on the median peak (0.93–1.12). The server's peaks read in every scan: 1517
  rows, none all zero — and the same 1517 readings, bit for bit, when the
  record carries a calibration factor (1 + 5, −13.5 or +28.6 ppm) and the served
  m/z are scaled by it.
- **Standard A as a tMS2 file** (semi-synthetic: its real centroids under tMS2
  trailers, garbled MS2 filters, 22 trailer rows dropped inside their blocks, one
  list per experiment): every scan placed (23 headerless: 22 by the cycle, the
  first walked back into the Full Scan block), the list fits the rule to 0 mDa,
  and the 12 spectra equal the time-series mapping's exactly. `ms2 analyze`:
  C10H17⁺ F E50 NCE 13.82 against 13.88 from the server's S/N-3 spectra (13.89
  from its S/N-10 ones, §11.8): F within 0.0016 at every NCE (at most, at NCE
  20), which at the crossing's slope (0.018 per NCE) bounds the E50s' difference
  by 0.09 NCE; it is 0.06. A list in the wrong order is refused by
  the rule; an extra list by the Full Scan event. Without a list the scan range
  names every precursor (its 56 MS2 scans with a row); its 23 headerless scans
  stay unassigned (no list pins the layout).
- **Standard B** (summary refused): 67 of 67 MS2 scans named; 81 scans with no
  trailer row in no group — scan 1 and 154–233, the latter after `t1`: the hidden
  tail (§10.5). **The 2026-09-10 exploratory file:** 213 of 213; 37 in no group,
  36 of them the tail after scan 398.

### 12.7 Limits and gotchas

- **Polarity**: a garbled file's filters and listing name none; pass the
  method's sign. Nothing splits a file by polarity: one whose readable filters
  carry both signs and whose MS2 scans have trailer rows is refused.
- **Inference is conservative.** A run the cycle allows several ways (a block of
  one event at a block switch), a run no walk of its length fits (an event
  skipped once), blocks that interleave in time, an experiment never seen with a
  trailer (an acquisition stopped early), and — on a layout no list pins (no
  list, or no Full Scan event) — any run that wraps, crosses a switch, leads or
  trails: those scans stay unassigned, counted and noted — never guessed. The
  method's list pins the layout. One case nothing in a file can tell: a short
  block of another experiment, every scan of it without a trailer row, inside a
  block's time, whose length completes that block's cycle (the block resuming at
  its first row) — its scans are exactly what the block would have run, and are
  placed as its own unless their centroids contradict those events' scan
  ranges.
- **An acquisition stopped inside an experiment** leaves that experiment's last
  rows never scanned: the mapping report counts each experiment's
  `rows_scanned` and lists `events_never_scanned`, a note names the experiment,
  its rows never scanned and the file's last MS2 scan, and the parents with no
  spectrum at its NCE (`missing_energies`) say so on their rows ("no spectrum at
  NCE …: its event was never scanned"). Their breakdown in `ms2 analyze` spans
  the energies they have (its `nce_max`, `n_energies`): an E50 "above range"
  there is above that parent's highest NCE, not the method's. Other events never
  scanned are noted too.
- **An event seen only through scans the cycle placed** takes its experiment's
  isolation width (`events_width_from_experiment`); noted only where the
  trailers read more than one width.
- **A data-dependent event** (readable filters naming a new precursor per cycle)
  maps scan by scan on a fixed range; where its scan range follows each
  precursor, its scans read several ranges and the file is refused.
- **The isolation width is the trailer's**, not the list's `Q1 Resolution`
  column (a row's own width is not always applied).
- **A stepped collision energy** has no single NCE: refused.
- **Heights**: per-scan averages run ~2 % above the server's (§12.4); a ratio of
  peaks of one spectrum moves much less.
- **An off-axis file** (§12.4, "The m/z axis"): the offset is measured from the
  file's own precursors against keys typed finely enough, and the precursor
  search, the MS1 window and `ms2 analyze` are centred on it. It assumes one
  offset for the file (MS1 and MS2 alike) and that most parents' lowest-energy
  windows are dominated by their precursor; with one or two parents nothing
  out-votes a co-isolate, so a reading contradicted by a candidate on its key is
  dropped, and of two readings only the one the other parent's window bears out
  is kept. A one-parent file whose axis really sits off AND that holds an
  unrelated ion (≥ 10 % of the target) on its key then reads no offset, and its
  precursor search, left on the key, locates that ion: `ms2 analyze` reads it as
  the target. Nothing in one parent tells the two apart; the dropped reading's
  ppm is on the parent's rows and in the notes — check the window. Keys typed with two decimals measure
  nothing: no offset is recorded, the search stays on the keys with its side-lobe
  guard on — a parent whose only candidate is a side lobe is reported ambiguous,
  never read as the precursor. A folder `ms2 analyze` cannot match ("no named ion
  matches the precursor … pass --mass-offset-ppm X") takes the offset by hand.
- **A calibrated sample's per-scan readings** (`--per-scan`) assume the record's
  calibration factor is the one the server applied to the spectra; an axis it
  does not explain leaves peaks reading 0 in every scan, counted and noted.
- **A list whose rows divide the method's** is refused only when the experiments
  it implies carry one NCE in pairs and alternate in time, and the scan-range rule
  does not check their rows (where it does, they are two parallel experiments of
  one NCE); on a fixed range such a parallel method takes the list once per
  experiment. A list of the right length with the wrong rows on a fixed-range
  method passes every check (nothing in the file names its precursors).
- **A scan of no block** (a trailer row with no MS order or no event, e.g. a
  `tSIM` experiment's): kept as read, in no spectrum and not in the MS1 average;
  headerless scans in a gap beside it stay unassigned, and — since it shows an
  experiment outside the layout — so do runs across a block switch and leading or
  trailing runs anywhere in the file. An experiment outside the layout none of
  whose scans keeps a trailer row is invisible: its row-less scans at a block
  switch read as the next events of the block before them, unless their
  centroids contradict those events' scan ranges.
- **`/metadata` is large** — the whole raw file's, about 0.1 MB per MS1 scan and 5 kB
  per MS2 scan: 7–20 MB for our own files of a few hundred scans, hundreds of MB for
  an hour of MS1 scans. `--check` reads it only for the files that need it or that
  it names, at most `--max-check`; `--per-scan` asks it once per sample, whatever
  part of the file the sample holds.

### 12.8 Constants and code map

| constant (`scans.py`) | value | role |
| --- | --- | --- |
| `SEED_PPM` / `MERGE_FWHM` | 1 ppm / 0.5 | the two clustering steps |
| `SERVER_HEIGHT_RATIO` | 1.020 | our intensity over the server's on the median peak (reported) |
| `TIME_MATCH_S` | 1 ms | a time-series time and its scan |
| `RANGE_CHECK_DA` | 0.01 Da | a walked scan's centroids inside its event's range; and a parent whose spectra's recorded ranges differ by more is cut to the range they share (§12.4) |
| `RULE_SLOPE` / `RULE_OFFSET_DA` / `RULE_TOL_MDA` | 1.02 / 20 Da / 0.5 mDa | the scan-range rule and its check |
| `RULE_DISTINCT_DA` | 1 mDa | HighMass values compared at this, per experiment: ≥ 2 seen events reading one value = a fixed range, values that differ = following the precursor, one seen event = unknown; an event whose value another of its experiment shares derives nothing; file-wide, < 2 distinct values derive nothing anywhere |
| `FILTER_RANGE_TOL_DA` / `FILTER_NCE_TOL` | 1 mDa / 0.005 | a filter agrees with its trailer |
| `SOURCE_MATCH_MDA` | 1 mDa | a list row and its event's readable filters must agree; an event's readable filters name one precursor (a scan of it with a garbled filter takes their median), else one per scan (such a scan has none). The rule is checked with `RULE_TOL_MDA` |
| `PARENT_TOL_DA` | 1 mDa | precursors chained into one parent |
| `RULE_DECIMALS` | 4 | a precursor derived from the rule is rounded to the decimals a method types |
| `sources.TIMESERIES_PAUSE_S` | 1 s | between `--per-scan`'s time-series requests |
| `sources.AXIS_SEARCH_PPM` / `AXIS_KEY_PPM` / `AXIS_MIN_PARENTS` / `AXIS_NOTE_PPM` | 50 ppm / 2 ppm / 3 / 3 ppm | the m/z axis offset: where a parent's precursor is read, how finely its key must be typed to read it, how many name it, when it is noted |
| `sources.PRECURSOR_LOBE_RATIO` / `_FWHM` | 50× / 5 line widths | the precursor's side-lobe guard, where the axis offset is unknown (§10.3) |
| `io_mascope.METADATA_TIMEOUT` | (15, 300) s | the `/metadata` request |
| `io_mascope.FILE_LIST_PAGE` / `FILE_LIST_PAUSE_S` / `FILE_LIST_MAX_PAGES` | 500 / 1 s / 200 | the file listing, page by page; at the cap, `FileListTruncated` |

| function | role |
| --- | --- |
| `scans.scan_table` / `filter_verdict` / `filter_polarity` / `parse_filter` | one row per scan off the trailer; the filters' verdict and polarity |
| `scans.read_inclusion_list` / `method_layout` / `map_precursors` / `infer_headerless` | the lists; events → experiments (`Layout.closed`); precursors; the cycle walk |
| `scans.rule_verdicts` / `rule_holds` / `rule_fit` / `rule_text` | the scan-range rule per experiment; its fit; in words |
| `scans.assign_from_times` / `scan_peak_intensities` | the server's groups scan by scan; its peaks in every scan |
| `scans.average_scans` / `build_spectra` / `ms1_average` / `window_inventory` / `extract_payload` | the averaging; the §10 tables; a payload end to end |
| `sources.extract_sample(per_scan=True)` / `extract_file` / `resolve_polarity` / `dual_polarity` / `calibration_factor` | the orchestrators; the polarity rule, and the dual-polarity refusal; the record's m/z calibration factor (`--per-scan`) |
| `sources.axis_offset` / `key_rounding_ppm` / `locate_precursor` | a raw file's m/z axis offset; how far a key's typing alone puts it; the precursor and its side-lobe guard |
| `sources.needs_check` / `file_row` / `check_file` / `files_frame` | `ms2 files` |
| `io_mascope.fetch_file_metadata` / `fetch_sample_files` / `fetch_ms2_timeseries` | the payload, the listing, a spectrum's time series |
| `analyze.parent_widths` / `folder_axis_offset` / `offset_hint` | each window at its own parent's width; the folder's m/z axis offset (a `--file` folder); the `--mass-offset-ppm` a window skipped for want of a named target needs, within the axis reach |
