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
tables by `peaky/ms2/sources.py` and `peaky ms2 list | extract` — is §10.

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
  losses, and so more wrong formulas, at every peak.
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

`get_timeseries` is not read yet (breakdown curves are a later stage). Parents are
always the **server's keys** (`parent_mz`), never re-derived.

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
MS1 window.

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
survival, E50 or other breakdown numbers — that is the next stage.

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
