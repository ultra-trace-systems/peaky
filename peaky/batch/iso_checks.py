"""The isotope checks of a batch (C11+a, C11+b; docs/EVIDENCE_LEVELS.md §3
`iso_veto`, `lead_lift`, §4.2).

A committed formula makes a claim about its isotope lines: how many carbons the
13C line counts, which heavy lines MUST be there, and how tall the M+2 region
may be. Three checks read those claims off the batch's stamped time series and
refute the formula where the series says otherwise -- a hard input (5b), and the
refuted pair leaves its neutral's chan2 / branch pools in both directions (rule
K's `alien` mechanism). A fourth, rule H, writes a POSITIVE fact instead: the
ion's exact halogen line, which lifts a tentative lead the line answers
(evidence._measure). All four are batch facts: they need the stamped time
series, never run per file, are never an axis, never in `cross`.

The instrument class is batch-generic: Orbitrap-class when the batch's peak-width
model resolves >= ORBITRAP_R200 at m/z 200 (`batch_summary.resolution.r_at_200`),
TOF-class otherwise; a batch with no width model is TOF-class (conservative: no
rule C).

  rule C  (the 13C carbon count; Orbitrap-class only) -- per (neutral, adduct)
        pair the batch stamped, not ion-only: in each spectrum the brightest M0
        stamp and the nearest peak to it + 1.0033548 within C_TOL_PPM; a spectrum
        is used when h0 x C x R13C >= C_KDL x its noise edge (1st-percentile
        height); >= C_NMIN used spectra. The pooled area ratio, corrected for the
        width ((m+1)/m)^1.5 and 17O (nO x R17O), and the pooled height ratio
        (17O only) read a carbon count each (/ R13C), with a delta-method se;
        both are divided by (1 + bias), the LEVEL-FREE median relative misread
        over every testable heteroatom-free pair. The formula is contradicted
        when area AND height both miss the ion's carbons by more than
        max(1.5, 0.25 C, 3 se); a 'too many' reading whose M+1 slot is another
        named pair's line in > C_OCC_MAX of the used spectra is untestable.
        Exempt: ions below the scan start + 1 Da (the batch's lowest m/z), and a
        14N nitrate cluster on a profile that also clusters on the 15N-labelled
        nitrate (its M+1 sits 6.32 mDa above the taller 15N sibling's line).
  REQ     (a required heavy line is absent) -- Orbitrap-class: per pooled pair
        whose ION carries Br / Cl / S / Si, the ion's count-aware fine structure, components closer than one FWHM of the width
        model merged into observable lines, placed relative to the STAMPED line
        (a Br2 ion is committed on its 79Br81Br line); required: every halogen
        group >= REQ_FRAC of the stamped line (81Br per Br, 37Cl per Cl, the
        Br2 1:2:1 and Cl2 patterns), and, on an Orbitrap-class batch without Br
        or Cl, 34S, 29Si and 30Si when the component is >= REQ_SHARE of its
        line. A line is detectable in a spectrum when its expected height >=
        REQ_DET_X x the spectrum's floor (the engine's height gate: noise edge x
        the batch's height_cutoff_x_edge); present when a peak sits within
        max(1 ppm, 4 sigma) of the blend centroid or the pure component.
        Refuted: >= REQ_NMIN detectable spectra and the line present in <=
        REQ_ABSENT_FRAC of them. TOF-class (`_req_tof`): per pooled pair whose
        ION carries Br / Cl (the reagent's included), its own M+2 line in every
        spectrum showing the pair, judged by the tier pass's primitive
        (assignment/satellites.heavy_line_verdict: the whole M+2 cluster
        predicted, testable at k_detect x the batch's detection edge, seen
        within max(15 ppm, 3 sigma) at >= 0.6x, untestable where another line
        holds the position or a split line could carry it); refuted over >=
        REQ_TOF_NMIN testable spectra, at least as many as the blended ones,
        where it is seen in < REQ_TOF_SEEN_MAX of them. Both classes: a pair
        with no stamp takes the tallest peak within the stamp window of its
        pooled m/z. On a TOF-class batch a refuted merged winner is Candidate
        (`tof_m2_gates`, after the stamp).
  HIGH    (a heavy line too high for the formula; variant V4) -- per pooled
        pair: at any heavy offset (81Br, 37Cl, 34S, 30Si, 18O, 13C2) the nearest
        peak within HIGH_TOL_PPM (Orbitrap 1, TOF 10) co-varies with the M0
        (r(log area) >= HIGH_RMIN over >= HIGH_NMIN spectra, present in >=
        HIGH_FRAC of them) at a pooled area AND height ratio >= HIGH_XEXP x the
        ion's count-aware expected M+2 and >= HIGH_FLOOR -- unless the line is
        another committed M0 or a plain '13C' child in > HIGH_SLOT_MAX of its
        spectra (slot guards), sits nearer a non-isotopic +2 alias than the heavy
        spacing (Orbitrap-class), or runs above expected + HIGH_CAP (no isotope
        envelope is that tall).
  rule H  (the halogen lock, C11+b; both classes) -- per pooled pair whose ION
        carries Cl or Br: in each spectrum the brightest M0 stamp and the
        nearest peak to it + the exact 37Cl - 35Cl / 81Br - 79Br spacing
        (LOCK_D) within LOCK_TOL_PPM (1 ppm, of the M0's m/z). A `lock` when the
        line is present in >= LOCK_FRAC of >= LOCK_NMIN M0 spectra, r(log area)
        >= LOCK_RMIN, the pooled area ratio in [LOCK_LO, LOCK_HI] x n x
        LOCK_PER_ATOM for the ion's count n (`count_window`), and the M0 not
        itself a heavy line (`heavy`: a line one Cl or Br spacing below it,
        co-varying, at a count window of either element). The silicon test
        (`si_rich`; the 2026-09-28 decision, "the 29Si line decides", its
        blended reading revised 2026-09-29): from LOCK_SI_MZ (~206.3, where
        the 30Si spacing, 0.206 mDa below 37Cl, enters the 1 ppm window) a Cl
        lock is refused when the ion's M+1 region shows the 29Si line a
        Si-rich ion making the partner from 30Si must carry -- n = ratio /
        0.0335 silicons, 29Si n x 0.0508 at +0.99957 Da, 3.79 mDa below 13C; a
        chlorinated ion carries none. Resolved (the width model's FWHM at the
        +1 m/z < 3.79 mDa): the line itself within 1 ppm, present and
        co-varying like the partner, at >= half its expected area -- where no
        such line is present at all the picker may not have parted it from
        13C, and the +1 region decides as where they blend (`unparted`).
        Blended: the +1 region (every peak from 1 ppm below 29Si to 1 ppm above
        13C), present and co-varying with the M0 like the partner (r(log
        region area, log M0 area) >= LOCK_RMIN), is refused on its POSITION:
        its area-weighted position, the median over the spectra, >= half-way
        from the reading's own +1 position (`m1_line`: 13C, 2H, 15N, 17O, 33S
        and the reading's own 29Si) toward the blend a Si reading implies. A
        reading that itself carries 1-2 Si is also refused on its excess (the
        Si-reading guard): the region reading >= half the implied 29Si area
        above that reading's own +1 line -- its own 29Si pulls the half-way
        mark toward 29Si, so its position can fall short (`_silicon`). Why the
        position and not the height (the excess): whatever the peak picker
        reports for a blend -- a full-area centroid, or one peak at its apex
        carrying part of its area, as the regression batches' picker does on
        their siloxanes -- a Si-rich ion's +1 peak sits toward 29Si while a
        chlorinated ion's sits at its own position; the area does not survive
        that way (an apex carries 72-84 % of the blend, and a carbon-rich Cl
        reading's own 13C line absorbs the rest), and rule C, which the excess
        test leant on, reads no ion too dim for its 13C line. The excess test
        as first built (excess AND position) therefore let dim siloxane
        misreads lock with nothing refuting them, at any m/z from 206.3 (refute
        round 2). 81Br needs no test (30Si sits 1.11 mDa below it). What the
        test cannot refuse: a misread whose +1 region is absent from more than
        40 % of its spectra or does not co-vary with its M0 (the gates that
        keep a real ion's neighbours from refusing its lock); and it refuses a
        real Cl lock where another ion's line near the 29Si position merges into
        a dim ion's +1 region while the ion's own 13C line supplies the
        co-variation (~1.0-1.4 per 1000 real Cl ions placed on real +1 regions,
        against 0.5-0.9 for the excess test). The Si-reading guard can refuse a
        real Si1-2 Cl reading on any excess in its +1 region, a neighbour's line
        at the 13C position included -- it asks no position (measured 0 / 0 / 1
        of the 30 / 12 / 37 real Si1-2 ions of the regression batches from m/z
        206.3, forced blended with a one-Cl partner; the one a TOF C5H7NO2Si
        [M+Br]-: its region reads 4.1x the excess the guard needs while it sits
        2.22 mDa above 29Si, short of its mark at 1.17 -- a Br adduct, whose Cl
        reading would carry both halogens and which rule H therefore never
        tests; on the readings it would test, 0 / 0 / 0). On a TOF the position
        is no silicon signature: its peak list reads an ion's +1 region ~1.3 mDa
        below the ion's own +1 position (the regression TOF's median over its 69
        present regions; ~1.4 below the 13C spacing), past the half-way mark on
        some real ions -- harmless while no TOF Cl lock forms above m/z 206.3
        (the batch's one TOF lock, C2H3ClO2 [M+NO3]- at m/z 156, sits below the
        silicon test's threshold). Never locked
        (`untestable`): an ion carrying Si >= 3, both Cl and Br (a blended M+2),
        a pair with no M0 stamp, one stamped on a heavy isotopologue, one
        stamped in < LOCK_NMIN spectra.
        The halogen must be the sample's (`reagent`, 2026-09-28 decision): the
        batch's reagent supplies up to `reagent_supply` atoms of it per ion (the
        most any adduct of the profile adds), and a line counts only where the
        ion carries more than that -- whatever the adduct label says (the ion
        BrHNO3- is HNO3 [M+Br]- or HBr [M+NO3]-). Each row also carries
        `budget_ok`: the neutral passes the batch context's element budget with
        the locked halogen's cap lifted (the CF2 exemption's construction), the
        gate an element-budget lead is lifted by. At 1 ppm the check is
        Orbitrap-only in practice: a TOF's line scatter is several ppm.

Measured on the three regression batches through this engine against the trunk
with rule K (C18 + K, C19(c)); identified / ion claim signal in % of the batch
series (the runs equal the replay row for row):
  rule C  labelled-nitrate Orbitrap: 19 pairs refuted, 11 merged rows move with
          the pool exclusion (4 of them acid-branch siblings 3b -> 4b),
          identified 45.148 -> 44.713, ion 33.224 -> 32.711; uronium Orbitrap: 2
          refuted, 1 move (5a -> 5b); the TOF: TOF-class, none. 0 of 62 roster /
          well-known species refuted; three Si10-Si12 readings whose +1 line is
          mostly 29Si are not read.
  REQ     labelled nitrate: 152 pooled / 104 merged refuted, 71 merged moves,
          identified -0.774, ion -0.843 (Br / Cl efficiency 1); uronium: 3 / 2
          (Si efficiency 0.58: D7.urea and D5 [M+H]+ untestable, not refuted),
          identified -0.257; the TOF: 94 / 27 at both windows (Br 0.96, Cl 0.70),
          identified 0.000, ion -0.126. The TOF branch re-read as the ion's own
          M+2 line: 993 of 3273 pooled pairs refuted on the bromide /
          nitrate TOF batch (the former branch: 117); with the per-file test
          and the doublet its merged Assigned rows go 355 -> 287 (none of its 34
          true readings lost); a nitrate-only low-resolution TOF moves 0.
  HIGH    labelled nitrate: 11 refuted, 5 merged moves (chloride adducts read as
          aromatic [M+^NO3]-), ion -0.202; uronium: none (no Br fits m/z 131.08);
          the TOF: 64 / 23, 6 merged moves; none identified.
  all three: identified / ion 45.148 / 33.224 -> 44.374 / 32.045 (labelled
          nitrate, 76 merged moves), 73.822 / 12.552 -> 73.565 / 12.549 (uronium,
          3), 11.207 / 4.054 -> 11.207 / 3.907 (the TOF, 25). First measured before
          rule K (C17 + U: 48.266 / 30.879 -> 47.319 / 28.967 on the labelled
          nitrate); the difference is rows rule K already refutes.
  rule H  (against the C11+a trunk, the same batches; the C / REQ / HIGH rows
          recomputed byte-identical) labelled nitrate: 282 tested, 16 locked
          (173 no lock, 92 untestable, 1 heavy), the 16 an independent line
          scan locks; 4 leads lifted (C6H9ClO3, C7H11ClO5, C6H10Cl2O4 [M-H]-,
          HBr [M+^NO3]-: 5b -> 4b), ion 32.045 -> 32.158; uronium: 8 / 1, the
          C7H11ClO2 urea cluster lifted (Cl its only budget violation), ion
          12.549 -> 12.576; the TOF: 2620 / 1, 659 `reagent` (ions the
          bromide reagent could have put the Br into), no lift. Identified 0 on
          all three; a lock at +-4 / 8 / 12 / 16 mDa around either spacing moves
          nothing. The silicon test reads the four Cl locks above m/z 206.3
          (C7H11ClO5, C6H10Cl2O4, C10H19ClO3 [M-H]-; the urea cluster): no 29Si
          line at its own position on any (`unparted`); their +1 region sits
          at the ions' own 13C position, 1.5-1.8 mDa short of the half-way
          mark, and reads -0.04 to +0.03x above their own +1 line -- the locks
          and the lifts stand (the tables are identical under the excess test
          and the position test). On the uronium batch's own siloxanes no Cl
          lock can form (none passes the one-Cl gates at 1 ppm over its
          spectra). As a control of the test alone, their M+2 handed over as
          the partner (within 1 ppm of the 37Cl or the 30Si spacing where
          present in >= 60 % of the spectra, else taken within 3 ppm of the
          30Si spacing where 1 ppm finds it in < 60 % -- Si10 24 %, Si11 0 %
          at the 30Si spacing, 46 % / 1 % at the 37Cl one) and read as each
          CHNOS(+Si <= 2)+Cl formula within 1 ppm, over all, the bright and
          the dim half of the spectra: of 224 such cases the position test
          refuses 115, rule C refutes 48 more, 61 pass -- all 61 with a +1
          region the gates cannot read (the D7 urea cluster's in 26 % of its
          spectra, the D5 urea one's at r 0.59); the excess test refused 2,
          left rule C 150 and let 72 pass. Where the real lock gates pass (the
          Si12 line's bright half, 11 readings) rule C refutes every one.
          Over the full series (and its bright half) rule C refutes every one
          of the 37 Cl1 / Cl2 readings (23 one-Cl) of the D5, Si10 and Si11
          urea clusters within 1 ppm, and Si12's 19 (11); in the dim half it
          is untestable on the D5 urea cluster's 3, 2 of Si10's and all 19 of
          Si12's. The D7 urea cluster's readings are rule-C-untestable, and
          only the lock's 1 ppm gate stops it (its M+2 line is in 0 % of its
          spectra at the 37Cl spacing). On the real
          Si-free ions of the two Orbitrap batches, forced blended with a
          one-Cl partner at the bottom of the window, the position test fires
          on 0 / 1, as the excess test did: the co-variation gate's work (the
          excess test without it fired on 23 / 14 from m/z 206.3, 37 -> 1 --
          the 24th labelled-nitrate fire of the earlier 38 sat at m/z 206.04,
          below the threshold once the AME2020 30Si spacing moved it -- and the
          position test without it on 43 of the 277 with a +1 region); on the
          TOF, blended at every mass, 24 (the excess test 17): the seven more
          are Si-free ions whose +1 region sits 1.4-2.6 mDa below the ion's own
          +1 position (the area-weighted region, the median over the spectra;
          1.7-2.5 mDa below the 13C spacing, 1.2-2.1 mDa above 29Si).
"""
from __future__ import annotations

import math
import re
import warnings
from itertools import product

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

CHECKS = ("C", "REQ", "HIGH", "H")
CHECK_NAME = {"C": "rule C", "REQ": "REQ", "HIGH": "HIGH", "H": "rule H"}
#: the checks that refute (a veto); rule H writes a positive fact instead
VETO_CHECKS = ("C", "REQ", "HIGH")
#: Orbitrap-class: the batch's width model resolves at least this at m/z 200
ORBITRAP_R200 = 50_000.0
D13C = 1.0033548378
# --- rule C
R13C = 0.010816
R17O = 0.000381
C_TOL_PPM = 3.0
C_KDL = 5.0
C_NMIN = 8
C_TOL_ABS, C_TOL_REL, C_TOL_SE = 1.5, 0.25, 3.0
C_OCC_MAX = 0.2
C_HETERO = ("F", "S", "Si", "Cl", "P", "I", "Br")
#: rule C reads the +1 line as 13C only where 13C makes at least this share of it
#: (a Si-rich ion's +1 line is mostly 29Si, 3.8 mDa lower: 2026-09-27 decision)
C_MIN_13C_SHARE = 0.5
#: the other light-element contributions to an ion's +1 line, per atom relative to its all-light line
_M1_OTHER = {"H": 0.000115 / 0.999885, "N": 0.00364 / 0.99636, "O": 0.00038 / 0.99757,
             "S": 0.0075 / 0.9499, "Si": 0.04685 / 0.92223}
# --- REQ
REQ_FRAC = 0.25          # a halogen group is required when >= 0.25x the stamped line
REQ_SHARE = 0.5          # an S / Si component must be >= 50 % of its observable line
REQ_DET_X = 3.0          # detectable: expected height >= 3x the spectrum's floor
REQ_NMIN = 3
REQ_ABSENT_FRAC = 0.2
REQ_ORBI_MIN_PPM = 1.0
REQ_ORBI_SIGMA_K = 4.0
REQ_MERGE_FWHM = 1.0
# --- REQ on a TOF-class batch (`_req_tof`): the ION's own M+2 line in every
# spectrum showing the pair, judged by satellites.heavy_line_verdict (the tier
# pass's per-file test); refuted over >= REQ_TOF_NMIN testable spectra, at least
# as many as the blended ones, where it is seen in < REQ_TOF_SEEN_MAX of them
REQ_TOF_NMIN = 10
REQ_TOF_SEEN_MAX = 0.3
# --- the TOF merged-row gates (`tof_m2_gates`, after the stamp)
#: the doublet: a merged row's line at TOF_DBL_LO..TOF_DBL_HI x the line one
#: 79Br -> 81Br spacing below it, in >= TOF_DBL_SHARE of the spectra showing it,
#: is that line's 81Br partner, not an M0 -- unless its own reading carries Br /
#: Cl whose M+2 the batch sees (REQ's seen share >= TOF_DBL_OWN_SEEN)
TOF_DBL_LO, TOF_DBL_HI = 0.58, 1.56
TOF_DBL_SHARE = 0.5
TOF_DBL_OWN_SEEN = 0.5
#: an element's heavy lines, as tall as this batch shows them: the median seen/theory
#: height over the pairs whose line is present in >= REQ_EFF_SEEN of >= REQ_NMIN
#: detectable spectra, from >= REQ_EFF_PAIRS pairs, read within [REQ_EFF_FLOOR, 1]
#: (never above theory; 2026-09-27 decision: the uronium batch's Si lines run
#: 0.4-0.5x theory on real siloxanes, their peaks 1.3-1.5x wider)
REQ_EFF_SEEN = 0.5
REQ_EFF_PAIRS = 3
REQ_EFF_FLOOR = 0.25
_MIN_REL = 1e-5
_ISO = {
    "C": [(0.0, 0.98930), (1.0033548378, 0.01070, "13C")],
    "H": [(0.0, 0.999885), (1.0062767, 0.000115, "2H")],
    "N": [(0.0, 0.996360), (0.9970349, 0.003640, "15N")],
    "O": [(0.0, 0.997570), (1.0042169, 0.000380, "17O"), (2.0042464, 0.002050, "18O")],
    "S": [(0.0, 0.949900), (0.9993878, 0.007500, "33S"), (1.9957959, 0.042500, "34S")],
    "Cl": [(0.0, 0.757600), (1.9970499, 0.242400, "37Cl")],
    "Br": [(0.0, 0.506900), (1.9979521, 0.493100, "81Br")],
    "Si": [(0.0, 0.922230), (0.9995683, 0.046850, "29Si"), (1.9968442, 0.030920, "30Si")],
}
_MAXK = {"Br": 6, "Cl": 6, "C": 3, "Si": 3, "S": 2, "O": 2, "H": 1, "N": 1}
# --- HIGH
HIGH_TOL_PPM = {"orbitrap": 1.0, "tof": 10.0}
HIGH_PARENT_PPM = {"orbitrap": 2.0, "tof": 12.0}
HIGH_NMIN = 8
HIGH_FRAC, HIGH_RMIN, HIGH_XEXP, HIGH_FLOOR = 0.6, 0.8, 3.0, 0.2
HIGH_SLOT_MAX = 0.2
HIGH_CAP = 3.0
_A13, _A15, _A18 = 0.0107 / 0.9893, 0.00364 / 0.99636, 0.00205 / 0.99757
_A34, _A37, _A81, _A30 = 0.0425 / 0.9499, 0.2424 / 0.7576, 0.4931 / 0.5069, 0.03092 / 0.92223
HIGH_OFFSETS = {"81Br": 1.9979535, "37Cl": 1.9970499, "34S": 1.9957959, "30Si": 1.9968442, "18O": 2.0042463,
                "13C2": 2.0067097}
#: the element each heavy offset needs in the ion (18O / 13C2: any formula makes them)
_OFFSET_ELEMENT = {"81Br": "Br", "37Cl": "Cl", "34S": "S", "30Si": "Si"}
#: the non-isotopic +2 aliases an Orbitrap can tell from a heavy spacing
HIGH_ALIASES = {"F<->OH": 1.995660, "C3<->F2": 1.996810, "N2<->CH2O": 2.004410}
#: per-atom height of each element's heavy line (for the atoms a line implies)
_OFFSET_PER_ATOM = {"81Br": _A81, "37Cl": _A37, "34S": _A34, "30Si": _A30}
#: the element-fit search: an ion carrying the implied atoms must fit its M0 mass
#: with a CHNOS rest (H <= 2C + N + 4) within this window (2026-09-27 decision)
HIGH_FIT_PPM = {"orbitrap": 5.0, "tof": 20.0}
# --- rule H (C11+b): the exact-spacing halogen lock -- a POSITIVE fact, never a veto
LOCK_D = {"Cl": 1.9970499, "Br": 1.9979521}       # 37Cl - 35Cl, 81Br - 79Br (the _ISO spacings)
LOCK_PER_ATOM = {"Cl": 0.3198, "Br": 0.9728}      # the heavy line per atom of the light one
LOCK_OFFSET = {"Cl": "37Cl", "Br": "81Br"}
LOCK_TOL_PPM = 1.0                                # both classes: on a TOF it seldom finds a line
LOCK_NMIN, LOCK_FRAC, LOCK_RMIN = 20, 0.6, 0.8
LOCK_LO, LOCK_HI = 0.65, 1.45                     # the count window: [LO, HI] x n x per atom
#: the heavy check: the counts a lighter line's window is tried at, per element
LOCK_HEAVY_N = {"Cl": (1, 2, 3, 4), "Br": (1, 2)}
LOCK_SI_MAX = 2                                   # an ion with Si >= 3 is never locked
#: a line stamped this far above the ion's all-light m/z is a heavy isotopologue
LOCK_STAMP_MAX_DA = 0.5
# the silicon test (2026-09-28 decision: "the 29Si line decides"; its blended
# reading the +1 peak's position since 2026-09-29). A 30Si line sits 0.206 mDa
# below the 37Cl one, inside the 1 ppm window from LOCK_SI_MZ on; there a Cl lock
# is refused where the M+1 region carries the 29Si line a Si-rich ion making the
# partner's area from 30Si must carry. 81Br needs none: 30Si sits 1.11 mDa below
# it, outside 1 ppm below m/z ~1100. D29SI is the isotope table's 29Si - 28Si
# spacing, 0.9995683 (0.2 uDa above AME2020's 0.9995681, 28.976494665 -
# 27.976926535: 0.0006 ppm at m/z 350). D30SI is the AME2020 30Si - 28Si spacing
# (29.973770136 - 27.976926535); the _ISO / HIGH_OFFSETS entry (1.9968442) is 0.6
# uDa high and stays: re-rounding it there would move HIGH's offset_mda on 45 / 11
# / 73 rows of the three regression batches, two offset labels -- the labelled
# nitrate's C5H8Cl2O2 [M-H]- 30Si -> 37Cl (a rule H lock whose HIGH row names its
# Cl2 M+2 a 30Si line seen in 8 % of its spectra: a `consistent` row is labelled
# by its largest ratio over the expected line, not by `_label`'s nearest spacing)
# and the TOF's C11H10N2O14 [M+NO3]- 37Cl -> 30Si --, n_used / ratio / r /
# presence on 10 rows, other_m0 on 3 TOF HIGH rows and the note on 9 rows; no
# verdict. Left for its own card (2026-09-29).
D29SI, D30SI = _ISO["Si"][1][0], 1.9968436
LOCK_SI_PER_ATOM = {"29Si": _ISO["Si"][1][1] / _ISO["Si"][0][1], "30Si": _ISO["Si"][2][1] / _ISO["Si"][0][1]}
LOCK_SI_MZ = (LOCK_D["Cl"] - D30SI) / (LOCK_TOL_PPM * 1e-6)
#: the share of what the Si reading implies that refuses a Cl lock: the 29Si line's area (resolved); where
#: 29Si and 13C blend, the +1 region's shift toward the Si blend -- or, on a reading carrying 1-2 Si, its excess
LOCK_SI_FRAC = 0.5

TABLE_COLUMNS = (
    "neutral_formula", "adduct", "check", "instrument", "ion", "mz", "stamped", "n_spectra", "n_used",
    "verdict", "veto",
    # rule C
    "n_carbon", "c_area", "c_height", "se_area", "se_height", "bias_area", "bias_height", "occupied",
    # REQ
    "line", "expected", "line_eff", "n_present", "det_frac", "window_ppm", "n_present_wide", "det_frac_wide",
    # HIGH (and rule H: offset .. other_m0)
    "offset", "ratio_area", "ratio_height", "r", "presence", "offset_mda", "other_m0", "other_13c",
    # rule H
    "lock", "element", "n_halogen", "ratio_lo", "ratio_hi", "heavy_cl", "heavy_br", "budget_ok", "budget_why",
    "si_n", "si29_expected", "si29_seen", "si29_mode",
    "note",
)
VERDICTS = {
    "C": ("agree", "ambiguous", "contradict", "untestable", "scan_edge", "exempt_14N"),
    "REQ": ("present", "absent", "untestable"),
    "HIGH": ("consistent", "guarded", "too_high"),
    "H": ("lock", "no_lock", "si_rich", "heavy", "reagent", "untestable"),
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TABLE_COLUMNS))


# --------------------------------------------------------------------------- the batch
def _resolution(resolution):
    from peaky.chem.resolution import Resolution
    if resolution is None:
        return None
    if isinstance(resolution, Resolution):
        return resolution
    if isinstance(resolution, dict):
        return Resolution.from_dict(resolution) if resolution.get("coef") is not None else None
    return Resolution.coerce(resolution)


def instrument_class(resolution) -> str:
    """'orbitrap' when the batch's width model resolves >= ORBITRAP_R200 at m/z
    200, else 'tof' -- and 'tof' when there is no width model at all."""
    rp = _resolution(resolution)
    if rp is None:
        return "tof"
    r = rp.r_at(200.0)
    return "orbitrap" if np.isfinite(r) and r >= ORBITRAP_R200 else "tof"


def _scale(mass_scale) -> tuple[float, float]:
    """(sigma_ppm, stamp_ppm) of the batch's traces.MassScale (or its as_dict)."""
    if mass_scale is None:
        return float("nan"), 6.0
    get = mass_scale.get if isinstance(mass_scale, dict) else (lambda k, d=None: getattr(mass_scale, k, d))
    sigma = get("sigma_ppm")
    stamp = get("stamp_ppm")
    sigma = float(sigma) if sigma is not None and np.isfinite(float(sigma)) else float("nan")
    stamp = float(stamp) if stamp is not None and np.isfinite(float(stamp)) else float(get("tol_ppm", 6.0) or 6.0)
    return sigma, stamp


class _Series:
    """The stamped batch time series, one array per column, sorted by (spectrum,
    m/z), with each spectrum's noise edge (its 1st-percentile height)."""

    def __init__(self, ts: pd.DataFrame):
        t = ts.copy()
        t["mz"] = pd.to_numeric(t["mz"], errors="coerce")
        t["height"] = pd.to_numeric(t["height"], errors="coerce")
        t["area"] = pd.to_numeric(t["area"], errors="coerce") if "area" in t.columns else t["height"]
        t = t[np.isfinite(t["mz"]) & (t["height"] > 0) & (t["area"] > 0)]
        for c in ("role", "neutral_formula", "adduct", "iso_label", "ion_formula"):
            t[c] = t[c].astype(object).where(t[c].notna(), "").astype(str) if c in t.columns else ""
        self.sids = np.array(sorted(t["sample_item_id"].astype(str).unique()))
        code = {s: i for i, s in enumerate(self.sids)}
        t["code"] = t["sample_item_id"].astype(str).map(code).astype(int)
        t = t.sort_values(["code", "mz"], kind="mergesort").reset_index(drop=True)
        self.t = t
        self.code = t["code"].to_numpy()
        self.mz = t["mz"].to_numpy(float)
        self.h = t["height"].to_numpy(float)
        self.a = t["area"].to_numpy(float)
        self.role = t["role"].to_numpy()
        self.nf = t["neutral_formula"].to_numpy()
        self.ad = t["adduct"].to_numpy()
        self.label = t["iso_label"].to_numpy()
        self.key = self.code * 1e4 + self.mz
        self.n = len(self.sids)
        edge = t.groupby("code")["height"].apply(lambda h: float(np.percentile(h.to_numpy(float), 1.0)))
        self.edge = edge.reindex(range(self.n)).to_numpy(float)
        self.scan_start = float(self.mz.min()) if len(self.mz) else float("nan")
        # the brightest M0 stamp of each (neutral, adduct) per spectrum
        m0 = t[(t["role"] == "M0") & (t["neutral_formula"] != "") & (t["adduct"] != "")]
        m0 = m0.sort_values("height", ascending=False, kind="mergesort").drop_duplicates(
            ["neutral_formula", "adduct", "code"])
        m0 = m0.sort_values(["neutral_formula", "adduct", "code"], kind="mergesort")
        self.m0 = {k: g for k, g in m0.groupby(["neutral_formula", "adduct"], sort=False)}
        # pair ids of every stamped row (the slot guards)
        pk = pd.Series(list(zip(self.nf, self.ad)))
        self.pair_code, self.pair_index = pd.factorize(pk)

    def window(self, codes, targets, ppm):
        """(lo, hi) index ranges of the peaks within +-ppm of each (spectrum, target)."""
        codes = np.asarray(codes)
        targets = np.asarray(targets, float)
        tol = targets * ppm * 1e-6
        lo = np.searchsorted(self.key, codes * 1e4 + targets - tol, "left")
        hi = np.searchsorted(self.key, codes * 1e4 + targets + tol, "right")
        return lo, hi

    def tallest(self, codes, targets, ppm):
        """Index of the tallest peak within +-ppm of each (spectrum, target), -1 where none."""
        lo, hi = self.window(codes, targets, ppm)
        out = np.full(len(lo), -1, dtype=np.int64)
        for i in np.nonzero(hi > lo)[0]:
            out[i] = lo[i] + int(np.argmax(self.h[lo[i]:hi[i]]))
        return out

    def nearest(self, codes, targets, ppm_of=None, ppm=None, reach=(-1, 0)):
        """Index of the peak nearest each (spectrum, target) in the same spectrum
        (-1 where the spectrum is empty or, with `ppm`, nothing sits within ppm x
        `ppm_of` / 1e6 of the target)."""
        codes = np.asarray(codes)
        targets = np.asarray(targets, float)
        idx = np.searchsorted(self.key, codes * 1e4 + targets)
        best = np.full(len(targets), -1, dtype=np.int64)
        bestd = np.full(len(targets), np.inf)
        for d in reach:
            k = np.clip(idx + d, 0, max(len(self.key) - 1, 0))
            if not len(self.key):
                break
            ok = self.code[k] == codes
            dist = np.abs(self.mz[k] - targets)
            upd = ok & (dist < bestd)
            best[upd], bestd[upd] = k[upd], dist[upd]
        if ppm is not None:
            ref = targets if ppm_of is None else np.asarray(ppm_of, float)
            best[~(bestd / ref * 1e6 <= ppm)] = -1
        return best


def _pooled(frames: dict) -> pd.DataFrame:
    """One row per (neutral, adduct) the pooled per-file ledgers commit: the ion
    formula of its first M0 row and the median m/z (as evidence._measure pools
    them), and whether it is an ion-only pair."""
    from peaky.assignment import evidence as EV
    parts = [f for f in (frames or {}).values() if f is not None and len(f) and "role" in f.columns]
    cols = ["neutral_formula", "adduct", "ion", "mz", "ion_only"]
    if not parts:
        return pd.DataFrame(columns=cols)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    m0 = frame[frame["role"].astype(str) == "M0"].copy()
    if m0.empty:
        return pd.DataFrame(columns=cols)
    m0["__n"] = m0["neutral_formula"].fillna("").astype(str) if "neutral_formula" in m0.columns else ""
    m0["__a"] = m0["adduct"].fillna("").astype(str) if "adduct" in m0.columns else ""
    m0["__io"] = EV.is_ion_only(m0).to_numpy()
    m0["__mz"] = pd.to_numeric(m0["mz"], errors="coerce") if "mz" in m0.columns else np.nan
    m0["__ion"] = m0["ion_formula"].astype(str) if "ion_formula" in m0.columns else ""
    g = m0.groupby(["__n", "__a"], sort=True)
    out = pd.DataFrame({"ion": g["__ion"].first(), "mz": g["__mz"].median(), "ion_only": g["__io"].any()})
    out = out.reset_index().rename(columns={"__n": "neutral_formula", "__a": "adduct"})
    return out[(out["neutral_formula"] != "") & (out["adduct"] != "")].reset_index(drop=True)[cols]


def ion_counts(neutral: str, adduct: str, ion) -> dict:
    """The ion's composition: its ion formula when it carries a charge sign, else
    neutral + adduct (a ledger can hold the NEUTRAL in `ion_formula`) --
    evidence.ion_composition."""
    from peaky.assignment import evidence as EV
    return EV.ion_composition(neutral, adduct, ion)


# --------------------------------------------------------------------------- rule C
def _labelled_sibling(adduct: str, prof) -> bool:
    """A 14N nitrate cluster on a profile that also clusters on the 15N-labelled
    nitrate: its M+1 slot sits 6.32 mDa above the taller 15N sibling's line."""
    from peaky.assignment import evidence as EV
    lab = EV.LABEL_FOLD.get(adduct)
    return bool(lab and lab != adduct and lab in set(getattr(prof, "adducts", None) or ()))


def _rule_c(S: _Series, pooled: pd.DataFrame, prof) -> pd.DataFrame:
    scope = set(zip(pooled.loc[~pooled["ion_only"].astype(bool), "neutral_formula"],
                    pooled.loc[~pooled["ion_only"].astype(bool), "adduct"]))
    keys = [k for k in S.m0 if k in scope]
    if not keys:
        return _empty()
    rows = []
    for k in keys:
        g = S.m0[k]
        codes = g["code"].to_numpy()
        mz0 = g["mz"].to_numpy(float)
        j = S.nearest(codes, mz0 + D13C, ppm=C_TOL_PPM)
        hit = j >= 0
        jj = np.where(hit, j, 0)
        h1 = np.where(hit, S.h[jj], 0.0)
        a1 = np.where(hit, S.a[jj], 0.0)
        same = (S.nf[jj] == k[0]) & (S.ad[jj] == k[1])
        occ = hit & (((S.role[jj] == "M0") & ~same) | ((S.role[jj] == "iso_child") & ~same & (S.nf[jj] != "")))
        ion = g["ion_formula"].mode()
        ion = str(ion.iloc[0]) if len(ion) else ""
        ic = _parse_ion(ion)
        nC = int(ic.get("C", 0))
        if nC < 1:
            continue
        mzm = float(np.median(mz0))
        wf = ((mzm + 1.0) / mzm) ** 1.5
        oth = ic.get("O", 0) * R17O
        h0, a0 = g["height"].to_numpy(float), g["area"].to_numpy(float)
        e = S.edge[codes]
        u = h0 * nC * R13C >= C_KDL * e
        n = int(u.sum())
        share, other = c13_share(ic)
        rec = dict(neutral_formula=k[0], adduct=k[1], ion=ion, mz=mzm, n_spectra=int(len(g)), n_used=n,
                   n_carbon=nC, het=sum(C.parse_formula(k[0]).get(x, 0) for x in C_HETERO),
                   c13_share=share, m1_other=other)
        if n > 0:
            for nm, x0, x1, w in (("area", a0[u], a1[u], wf), ("height", h0[u], h1[u], 1.0)):
                s0, s1 = x0.sum(), x1.sum()
                rr = s1 / s0
                res = x1 - rr * x0
                se = np.sqrt(n / max(n - 1, 1) * (res ** 2).sum()) / s0 if n > 1 else np.nan
                rec[f"c_{nm}_raw"] = (rr / w - oth) / R13C
                rec[f"se_{nm}"] = se / w / R13C
            rec["occupied"] = float(occ[u].mean())
        rows.append(rec)
    if not rows:
        # every stamped pair is carbon-free (HBr, HNO3): no 13C line to read
        return _empty()
    d = pd.DataFrame(rows)
    for c in ("c_area_raw", "c_height_raw", "se_area", "se_height", "occupied"):
        if c not in d.columns:
            d[c] = np.nan
    edge = d["mz"] < S.scan_start + 1.0
    # the level-free bias: the median relative misread over every testable
    # heteroatom-free pair (the same veto set as a bias fitted on levelled acids)
    pop = d[(d["het"] == 0) & (d["n_used"] >= C_NMIN) & ~edge]
    bias = {u: (float(((pop[f"c_{u}_raw"] - pop["n_carbon"]) / pop["n_carbon"]).median()) if len(pop) else 0.0)
            for u in ("area", "height")}
    d["bias_area"], d["bias_height"] = bias["area"], bias["height"]
    d["c_area"] = d["c_area_raw"] / (1.0 + bias["area"])
    d["c_height"] = d["c_height_raw"] / (1.0 + bias["height"])
    nc = d["n_carbon"].astype(float)
    tola = np.maximum.reduce([np.full(len(d), C_TOL_ABS), C_TOL_REL * nc, C_TOL_SE * d["se_area"].fillna(0)])
    tolh = np.maximum.reduce([np.full(len(d), C_TOL_ABS), C_TOL_REL * nc, C_TOL_SE * d["se_height"].fillna(0)])
    oa = (d["c_area"] - nc).abs() > tola
    oh = (d["c_height"] - nc).abs() > tolh
    v = np.where(d["n_used"] < C_NMIN, "untestable", np.where(~oa, "agree", np.where(oh, "contradict", "ambiguous")))
    v = pd.Series(v, index=d.index, dtype=object)
    v[(v == "contradict") & (d["c_area"] > nc) & (d["occupied"] > C_OCC_MAX)] = "untestable"
    # a +1 line 13C does not dominate is no carbon count (a Si-rich ion's is mostly 29Si)
    v[d["c13_share"] < C_MIN_13C_SHARE] = "untestable"
    ex14 = d["adduct"].map(lambda a: _labelled_sibling(a, prof))
    d["verdict"] = np.where(edge, "scan_edge", np.where(ex14, "exempt_14N", v))
    d["veto"] = d["verdict"].eq("contradict")
    d["note"] = [_c_note(r) for r in d.itertuples(index=False)]
    d["check"] = "C"
    d["stamped"] = True
    return d


def _parse_ion(ion: str) -> dict:
    return C.parse_formula(str(ion).rstrip("+-."))


def c13_share(ion: dict) -> tuple[float, str]:
    """(13C's share of the ion's +1 line, the largest other contributor) -- the
    +1 line also holds 29Si, 33S, 15N, 2H and 17O, unresolved from 13C where the
    peak is wider than their spacing; rule C reads a carbon count only where
    13C makes at least C_MIN_13C_SHARE of it."""
    c13 = ion.get("C", 0) * R13C
    other = {el: ion.get(el, 0) * a for el, a in _M1_OTHER.items() if ion.get(el, 0) > 0}
    tot = c13 + sum(other.values())
    if tot <= 0:
        return 0.0, ""
    top = max(other, key=other.get) if other else ""
    return float(c13 / tot), top


def _c_note(r) -> str:
    finite = isinstance(r.c_area, float) and np.isfinite(r.c_area)
    txt = (f"13C reads {r.c_area:.1f} C for {r.n_carbon} (height {r.c_height:.1f}, se {r.se_area:.1f}) "
           f"over {r.n_used} spectra") if finite else ""
    why = {"scan_edge": "below the scan start + 1 Da, not tested",
           "exempt_14N": "its M+1 slot is the 15N sibling's line, not tested"}.get(r.verdict)
    if why:
        return f"{txt}; {why}" if txt else why
    share = getattr(r, "c13_share", 1.0)
    if isinstance(share, float) and np.isfinite(share) and share < C_MIN_13C_SHARE:
        iso = {"Si": "29Si", "S": "33S", "N": "15N", "H": "2H", "O": "17O"}.get(getattr(r, "m1_other", ""), "other lines")
        why = f"the +1 line is mostly {iso} (13C {share:.0%} of it): no carbon count"
        return f"{txt}; {why}" if txt else why
    if not finite or r.n_used < C_NMIN:
        return f"{r.n_used} of {r.n_spectra} spectra bright enough for the 13C line (needs {C_NMIN})"
    if r.verdict == "untestable":
        return txt + f"; the M+1 slot is another pair's line in {r.occupied:.0%} of them"
    return txt


# --------------------------------------------------------------------------- REQ
def _element_dist(el: str, n: int):
    lev = _ISO[el]
    a0 = lev[0][1]
    heavy = lev[1:]
    kmax = min(n, _MAXK.get(el, 2))
    out = []
    for ks in product(*[range(kmax + 1)] * len(heavy)):
        t = sum(ks)
        if t > kmax:
            continue
        coef = math.comb(n, t) * math.factorial(t) / math.prod(math.factorial(k) for k in ks)
        rel = coef * float(np.prod([(h[1] / a0) ** k for h, k in zip(heavy, ks)]))
        if rel < _MIN_REL:
            continue
        out.append((sum(h[0] * k for h, k in zip(heavy, ks)), rel, {h[2]: k for h, k in zip(heavy, ks) if k}))
    return out


def fine_structure(counts: dict) -> pd.DataFrame:
    """The ion's isotopologues relative to its all-light line (count-aware,
    multinomial per element; components < 1e-5 dropped): shift, rel, tags."""
    comps = [(0.0, 1.0, {})]
    for el, n in counts.items():
        if el not in _ISO or n <= 0:
            continue
        new = []
        for s0, r0, t0 in comps:
            for s1, r1, t1 in _element_dist(el, int(n)):
                r = r0 * r1
                if r < _MIN_REL or s0 + s1 > 8.5:
                    continue
                t = dict(t0)
                t.update(t1)
                new.append((s0 + s1, r, t))
        comps = new
    df = pd.DataFrame({"shift": [c[0] for c in comps], "rel": [c[1] for c in comps], "tags": [c[2] for c in comps]})
    return df.sort_values("shift", kind="mergesort").reset_index(drop=True)


def _lines(fs: pd.DataFrame, fwhm: float):
    """Observable lines: consecutive components closer than one FWHM merge."""
    g = np.concatenate([[0], np.cumsum(np.diff(fs["shift"].to_numpy()) >= REQ_MERGE_FWHM * fwhm)]).astype(int)
    fs = fs.assign(line=g)
    w = fs["rel"].to_numpy()
    rel = np.bincount(g, weights=w)
    cen = np.bincount(g, weights=w * fs["shift"].to_numpy()) / rel
    return fs, cen, rel


def required_lines(counts: dict, fwhm: float, stamped_shift: float, tof: bool):
    """(stamped centroid, [dict(element, label, centroid, pure, ratio, share)]) --
    the heavy lines the ion's formula requires, relative to the stamped line."""
    fs, cen, rel = _lines(fine_structure(counts), fwhm)
    sl = int(np.argmin(np.abs(cen - stamped_shift)))
    sc, sr = float(cen[sl]), float(rel[sl])
    req = []
    nbr, ncl = counts.get("Br", 0), counts.get("Cl", 0)
    if nbr + ncl > 0:
        halo = fs[[all(k in ("81Br", "37Cl") for k in t) for t in fs["tags"]]]
        nominal = [int(t.get("81Br", 0) + t.get("37Cl", 0)) for t in halo["tags"]]
        for k, g in halo.groupby(nominal):
            main = g.loc[g["rel"].idxmax()]
            ln = int(main["line"])
            if ln == sl:
                continue
            ratio = float(rel[ln]) / sr
            if ratio < REQ_FRAC:
                continue
            el = "BrCl" if (nbr and ncl) else ("Br" if nbr else "Cl")
            lab = (f"M+{2 * k}" if k else "M0") + (f" ({_tag_text(main['tags'])})" if main["tags"] else " (all-light)")
            req.append(dict(element=el, label=lab, centroid=float(cen[ln]), pure=float(main["shift"]), ratio=ratio,
                            share=float(g["rel"].sum() / rel[ln])))
        return sc, req
    if tof:
        return sc, req        # 34S / 29Si / 30Si are unresolved inside a TOF's M+1 / M+2 clusters
    for el, lab in (("S", "34S"), ("Si", "29Si"), ("Si", "30Si")):
        if counts.get(el, 0) <= 0:
            continue
        m = fs[[t == {lab: 1} for t in fs["tags"]]]
        if m.empty:
            continue
        main = m.iloc[0]
        ln = int(main["line"])
        if ln == sl:
            continue
        share = float(main["rel"] / rel[ln])
        if share >= REQ_SHARE:
            req.append(dict(element=el, label=lab, centroid=float(cen[ln]), pure=float(main["shift"]),
                            ratio=float(rel[ln]) / sr, share=share))
    return sc, req


def _tag_text(tags: dict) -> str:
    return " ".join(f"{k}" if v == 1 else f"{v}x{k}" for k, v in tags.items())


def _req(S: _Series, pooled: pd.DataFrame, klass: str, rp, sigma: float, stamp: float,
         x_edge: float, log=None) -> pd.DataFrame:
    """REQ on an Orbitrap-class batch (a TOF-class batch runs `_req_tof`)."""
    heavy = ("Br", "Cl", "S", "Si")
    win = max(REQ_ORBI_MIN_PPM, REQ_ORBI_SIGMA_K * sigma if np.isfinite(sigma) else 0.0)
    floor = S.edge * x_edge
    # pass 1: every pair's required lines, where they would sit and what is there
    cases = []
    for r in pooled.itertuples(index=False):
        counts = ion_counts(r.neutral_formula, r.adduct, r.ion)
        if not any(counts.get(el, 0) > 0 for el in heavy):
            continue
        try:
            mz0 = C.ion_mz(r.neutral_formula, r.adduct)
        except Exception:
            continue
        g = S.m0.get((r.neutral_formula, r.adduct))
        stamped = g is not None and len(g) > 0
        if stamped:
            codes, pm, ph = g["code"].to_numpy(), g["mz"].to_numpy(float), g["height"].to_numpy(float)
        else:
            if not np.isfinite(r.mz):
                continue
            codes = np.arange(S.n)
            j = S.tallest(codes, np.full(S.n, float(r.mz)), stamp)
            ok = j >= 0
            codes, pm, ph = codes[ok], S.mz[j[ok]], S.h[j[ok]]
            if not len(codes):
                continue
        sc, req = required_lines(counts, rp.fwhm(mz0), float(np.median(pm)) - mz0, False)
        if not req:
            continue
        fl = floor[codes]
        lines = []
        for q in req:
            t1 = pm + (q["centroid"] - sc)
            t2 = pm + (q["pure"] - sc)
            hl = _line_height(S, codes, t1, t2, win)
            lines.append(dict(q, t1=t1, t2=t2, pres=hl > 0, seen=hl / np.maximum(ph * q["ratio"], 1e-12)))
        cases.append((r, stamped, codes, pm, ph, fl, lines))
    # the batch's own line efficiency per element (how tall the lines it sees come out)
    eff = line_efficiency(cases)
    if log is not None and eff:
        log("[iso_checks] REQ line height vs theory: "
            + ", ".join(f"{el} {v:.2f} ({n} pairs)" for el, (v, n) in sorted(eff.items())))
    # pass 2: the verdicts, with each line's detectability at the height this batch shows
    rows = []
    for r, stamped, codes, pm, ph, fl, lines in cases:
        out = []
        for q in lines:
            e = _eff_of(eff, q["element"])
            det = ph * q["ratio"] * e >= REQ_DET_X * fl
            nd = int(det.sum())
            pres = q["pres"]
            npd = int((pres & det).sum())
            frac = npd / nd if nd else np.nan
            testable = nd >= REQ_NMIN
            absent = testable and frac <= REQ_ABSENT_FRAC
            npw = frw = np.nan
            x = {k: v for k, v in q.items() if k not in ("t1", "t2", "pres", "seen")}
            out.append(dict(x, eff=e, n_det=nd, n_present=npd, det_frac=frac, n_present_wide=npw,
                            det_frac_wide=frw, testable=testable, absent=absent))
        ab = [x for x in out if x["absent"]]
        tst = [x for x in out if x["testable"]]
        pick = (min(ab, key=lambda x: (x["det_frac"], -x["n_det"])) if ab else
                min(tst, key=lambda x: (x["det_frac"], -x["n_det"])) if tst else
                max(out, key=lambda x: x["n_det"]))
        verdict = "absent" if ab else ("present" if tst else "untestable")
        note = "; ".join(_req_note(x, win) for x in ab) if ab else _req_note(pick, win)
        rows.append(dict(neutral_formula=r.neutral_formula, adduct=r.adduct, ion=str(r.ion), mz=float(np.median(pm)),
                         stamped=stamped, n_spectra=int(len(codes)),
                         n_used=pick["n_det"], line=pick["label"], expected=pick["ratio"], line_eff=pick["eff"],
                         n_present=pick["n_present"], det_frac=pick["det_frac"], window_ppm=win,
                         n_present_wide=pick["n_present_wide"], det_frac_wide=pick["det_frac_wide"],
                         verdict=verdict, veto=bool(ab), note=note))
    if not rows:
        return _empty()
    d = pd.DataFrame(rows)
    d["check"] = "REQ"
    return d


def _batch_edge(S: _Series, edge_cps) -> float:
    """The batch's typical detection edge: `edge_cps` when it is a positive
    number, else the median of the spectra's own (1st-percentile) edges."""
    try:
        e = float(edge_cps)
    except (TypeError, ValueError):
        e = float("nan")
    if np.isfinite(e) and e > 0:
        return e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(S.edge)) if len(S.edge) else float("nan")


def _m0_reads(S: _Series, r, stamp: float):
    """(stamped, codes, m/z, height) of a pooled pair's M0 line per spectrum: its
    brightest M0 stamp, else the tallest peak within the stamp window of its
    pooled m/z (None when it is nowhere)."""
    g = S.m0.get((r.neutral_formula, r.adduct))
    if g is not None and len(g) > 0:
        return True, g["code"].to_numpy(), g["mz"].to_numpy(float), g["height"].to_numpy(float)
    if not np.isfinite(r.mz):
        return None
    codes = np.arange(S.n)
    j = S.tallest(codes, np.full(S.n, float(r.mz)), stamp)
    ok = j >= 0
    if not ok.any():
        return None
    return False, codes[ok], S.mz[j[ok]], S.h[j[ok]]


def _req_tof(S: _Series, pooled: pd.DataFrame, rp, sigma: float, stamp: float, edge_cps=None,
             log=None) -> pd.DataFrame:
    """REQ on a TOF-class batch: the ION's own M+2 line (Br / Cl, the reagent's
    included) in every spectrum showing the pair, by the tier pass's primitive
    (satellites.heavy_line_verdict) -- the floor k_detect x the batch's
    detection edge (tiers.TOF_ASSIGN_FLOOR_X_EDGE x `edge_cps`), the window
    max(the scorer's TOF match window, 3 x the batch's per-ion scatter `sigma`),
    a sighting at >= 0.6x the prediction, and both blend guards. A pair is
    refuted (absent, a veto) over >= REQ_TOF_NMIN testable spectra, at least as
    many as the blended ones, where the line is seen in < REQ_TOF_SEEN_MAX of
    them; present where it is seen in more; untestable otherwise. A pair
    stamped on a heavy isotopologue (its line > 0.5 Da off the ion's all-light
    m/z) is not tested. `occupied` is the share of its spectra whose M+2
    position another line holds (blended)."""
    from peaky.assignment import satellites as SAT
    from peaky.assignment import tiers as T
    win = SAT.heavy_line_window_ppm(sigma, None)
    edge = _batch_edge(S, edge_cps)
    floor = T.TOF_ASSIGN_FLOOR_X_EDGE * edge if np.isfinite(edge) else float("nan")
    cases, flat = [], []
    for r in pooled.itertuples(index=False):
        counts = ion_counts(r.neutral_formula, r.adduct, r.ion)
        ratio, shift, label = SAT.heavy_line_prediction(counts)
        if ratio <= 0:
            continue
        try:
            mz0 = C.ion_mz(r.neutral_formula, r.adduct)
        except Exception:
            continue
        got = _m0_reads(S, r, stamp)
        if got is None:
            continue
        stamped, codes, pm, ph = got
        heavy = abs(float(np.median(pm)) - mz0) > T.TOF_M2_MONO_DA
        k = len(cases)
        cases.append((r, stamped, codes, pm, ratio, label, heavy))
        if not heavy:
            flat.append(pd.DataFrame({"case": k, "code": codes, "pm": pm, "ph": ph, "ratio": ratio,
                                      "shift": shift}))
    if not cases:
        return _empty()
    status = {}
    if flat:
        F = pd.concat(flat, ignore_index=True).sort_values(["code", "case"], kind="mergesort")
        st = np.empty(len(F), dtype=object)
        fcode = F["code"].to_numpy()
        pm_, ph_ = F["pm"].to_numpy(float), F["ph"].to_numpy(float)
        ra_, sh_ = F["ratio"].to_numpy(float), F["shift"].to_numpy(float)
        fw = float(rp.coef) * np.power(pm_ + sh_, float(rp.exponent)) + float(rp.offset)
        # the series is sorted by (spectrum, m/z): one spectrum's lines are one slice
        bounds = np.searchsorted(S.code, np.arange(S.n + 1), "left")
        cut = np.flatnonzero(np.diff(fcode)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(F)]):
            c = int(fcode[a])
            lo, hi = bounds[c], bounds[c + 1]
            v = SAT.heavy_line_verdict(S.mz[lo:hi], S.h[lo:hi], pm_[a:b], ph_[a:b], ra_[a:b], sh_[a:b], floor,
                                       win_ppm=win, fwhm=fw[a:b])
            st[a:b] = v["status"]
        F["status"] = st
        status = {k: g["status"].value_counts().to_dict() for k, g in F.groupby("case", sort=False)}
    rows = []
    for k, (r, stamped, codes, pm, ratio, label, heavy) in enumerate(cases):
        n = status.get(k, {})
        n_seen, n_abs = int(n.get(SAT.HL_SEEN, 0)), int(n.get(SAT.HL_ABSENT, 0))
        n_blend, n_dim = int(n.get(SAT.HL_BLENDED, 0)), int(n.get(SAT.HL_DIM, 0))
        n_test = n_seen + n_abs
        share = n_seen / n_test if n_test else np.nan
        testable = (not heavy) and n_test >= REQ_TOF_NMIN and n_test >= n_blend
        absent = bool(testable and share < REQ_TOF_SEEN_MAX)
        verdict = "absent" if absent else ("present" if testable else "untestable")
        line = f"M+2 ({label})"
        what = f"the ion's own {line} line ({ratio:.2f}x the stamped line)"
        if heavy:
            note = f"{what} not tested: the pair is stamped on a heavy isotopologue"
        else:
            where = (f"within {win:.3g} ppm at >= {SAT.HEAVY_LINE_FRAC:g}x; {n_blend} blended, "
                     f"{n_dim} under the {floor:.3g}-cps floor")
            if testable:
                note = f"{what} seen in {n_seen} of {n_test} testable spectra ({where})"
            else:
                note = (f"{what} testable in {n_test} spectra (needs {REQ_TOF_NMIN} and at least as many as "
                        f"the blended; {where})")
        rows.append(dict(neutral_formula=r.neutral_formula, adduct=r.adduct, ion=str(r.ion), mz=float(np.median(pm)),
                         stamped=stamped, n_spectra=int(len(codes)), n_used=n_test, line=line, expected=float(ratio),
                         line_eff=np.nan, n_present=n_seen, det_frac=share, window_ppm=win,
                         occupied=(n_blend / len(codes)) if len(codes) else np.nan,
                         verdict=verdict, veto=absent, note=note))
    d = pd.DataFrame(rows)
    d["check"] = "REQ"
    if log is not None:
        log(f"[iso_checks] REQ (TOF, the ion's own M+2 line): window {win:.3g} ppm, floor {floor:.3g} cps "
            f"({T.TOF_ASSIGN_FLOOR_X_EDGE:g}x the batch edge {edge:.3g}); {int((d['verdict'] == 'absent').sum())} "
            f"refuted, {int((d['verdict'] == 'present').sum())} present, "
            f"{int((d['verdict'] == 'untestable').sum())} untestable of {len(d)}")
    return d


def line_efficiency(cases) -> dict:
    """{element: (efficiency, n pairs)}: per element, the median over the pairs
    whose line is present in >= REQ_EFF_SEEN of its >= REQ_NMIN theory-detectable
    spectra of that pair's median seen/theory height, read within
    [REQ_EFF_FLOOR, 1]; an element with fewer than REQ_EFF_PAIRS such pairs is
    absent from the dict (read as 1: the theory height)."""
    per: dict = {}
    for _r, _st, _codes, _pm, ph, fl, lines in cases:
        for q in lines:
            det = ph * q["ratio"] >= REQ_DET_X * fl
            nd = int(det.sum())
            if nd < REQ_NMIN:
                continue
            seen = q["pres"] & det
            if seen.sum() < REQ_EFF_SEEN * nd:
                continue
            per.setdefault(q["element"], []).append(float(np.median(q["seen"][seen])))
    return {el: (float(np.clip(np.median(v), REQ_EFF_FLOOR, 1.0)), len(v))
            for el, v in per.items() if len(v) >= REQ_EFF_PAIRS}


def _eff_of(eff: dict, element: str) -> float:
    """The efficiency REQ reads for a required line's element (a Br + Cl line: the
    smaller of the two where it has none of its own); 1 without a measurement."""
    if element in eff:
        return eff[element][0]
    if element == "BrCl":
        return min(eff.get("Br", (1.0, 0))[0], eff.get("Cl", (1.0, 0))[0])
    return 1.0


def _line_height(S: _Series, codes, t1, t2, ppm) -> np.ndarray:
    """The tallest peak within +-ppm of either position of each (spectrum, line), 0 where none."""
    out = np.zeros(len(codes))
    for t in (t1, t2):
        j = S.tallest(codes, t, ppm)
        ok = j >= 0
        out[ok] = np.maximum(out[ok], S.h[j[ok]])
    return out


def _req_note(x: dict, win: float) -> str:
    e = x.get("eff", 1.0)
    shown = "" if not (isinstance(e, float) and e < 0.995) else f"; this batch shows the element's lines at {e:.2f}x"
    what = f"the {x['label']} line ({x['ratio']:.2f}x the stamped line{shown})"
    if not x["testable"]:
        return f"{what} detectable in {x['n_det']} spectra (needs {REQ_NMIN})"
    where = f"within {win:.3g} ppm"
    seen = x["n_present"]
    if x["absent"]:
        return f"{what} absent in {x['n_det'] - int(seen)} of {x['n_det']} detectable spectra ({where})"
    return f"{what} present in {int(seen)} of {x['n_det']} detectable spectra ({where})"


# --------------------------------------------------------------------------- HIGH
def expected_m2(ion: dict) -> float:
    """The ion's count-aware expected M+2 / M0 (first order per element, plus
    the 13C2 and 13C15N pairs)."""
    nc, nn = ion.get("C", 0), ion.get("N", 0)
    return (ion.get("Br", 0) * _A81 + ion.get("Cl", 0) * _A37 + ion.get("S", 0) * _A34
            + ion.get("Si", 0) * _A30 + ion.get("O", 0) * _A18 + nc * (nc - 1) / 2 * _A13 ** 2
            + nc * nn * _A13 * _A15)


def _high(S: _Series, pooled: pd.DataFrame, klass: str) -> pd.DataFrame:
    tol, par = HIGH_TOL_PPM[klass], HIGH_PARENT_PPM[klass]
    keys = list(zip(pooled["neutral_formula"], pooled["adduct"]))
    P = len(keys)
    if not P or not S.n:
        return _empty()
    PM = np.full((S.n, P), np.nan)
    PA, PH = PM.copy(), PM.copy()
    stamped = np.zeros(P, bool)
    for i, k in enumerate(keys):
        g = S.m0.get(k)
        if g is not None and len(g):
            c = g["code"].to_numpy()
            PM[c, i], PA[c, i], PH[c, i] = g["mz"].to_numpy(float), g["area"].to_numpy(float), g["height"].to_numpy(float)
            stamped[i] = True
    for i in np.flatnonzero(~stamped):
        mz = float(pooled["mz"].iat[i])
        if not np.isfinite(mz):
            continue
        codes = np.arange(S.n)
        j = S.tallest(codes, np.full(S.n, mz), par)
        ok = j >= 0
        PM[codes[ok], i], PA[codes[ok], i], PH[codes[ok], i] = S.mz[j[ok]], S.a[j[ok]], S.h[j[ok]]
    npar = np.isfinite(PM).sum(0)
    ions = [ion_counts(n, a, x) for (n, a), x in zip(keys, pooled["ion"])]
    exp = np.array([expected_m2(x) for x in ions])
    sign = np.array([-1.0 if str(a).rstrip().endswith("-") else 1.0 for _n, a in keys])
    # the pair id of each column in the series' factorized (neutral, adduct) codes (-2: never stamped)
    pid = {k: i for i, k in enumerate(S.pair_index)}
    pcol = np.array([pid.get(k, -2) for k in keys])
    is_m0 = S.role == "M0"
    is_13c = (S.role == "iso_child") & (S.label == "13C")
    have = np.isfinite(PM)
    cc, pp = np.nonzero(have)
    per = {}
    for lab, D in HIGH_OFFSETS.items():
        ci = np.full(PM.shape, -1, dtype=np.int64)
        if len(cc):
            pm = PM[cc, pp]
            ci[cc, pp] = S.nearest(cc, pm + D, ppm_of=pm, ppm=tol, reach=(-2, -1, 0, 1))
        hasc = ci >= 0
        cj = np.where(hasc, ci, 0)
        CM = np.where(hasc, S.mz[cj], np.nan)
        CA = np.where(hasc, S.a[cj], np.nan)
        CH = np.where(hasc, S.h[cj], np.nan)
        both = hasc & have
        n = both.sum(0)
        frac = np.where(npar > 0, n / np.maximum(npar, 1), 0.0)
        ra = np.where(n > 0, np.where(both, CA, 0).sum(0) / np.maximum(np.where(both, PA, 0).sum(0), 1e-12), np.nan)
        rh = np.where(n > 0, np.where(both, CH, 0).sum(0) / np.maximum(np.where(both, PH, 0).sum(0), 1e-12), np.nan)
        X = np.where(both, np.log(np.where(both, PA, 1.0)), np.nan)
        Y = np.where(both, np.log(np.where(both, CA, 1.0)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # a column with no spectrum holding both
            xm, ym = np.nanmean(X, 0), np.nanmean(Y, 0)
            cov = np.nansum((X - xm) * (Y - ym), 0)
            vx, vy = np.nansum((X - xm) ** 2, 0), np.nansum((Y - ym) ** 2, 0)
            r = np.where((n >= HIGH_NMIN) & (vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)
            off = np.nanmedian(np.where(both, (CM - PM - D) * 1e3, np.nan), 0)
        nh = hasc.sum(0)
        oth = np.where(nh > 0, (hasc & is_m0[cj] & (S.pair_code[cj] != pcol[None, :])).sum(0) / np.maximum(nh, 1), 0.0)
        o13 = np.where(nh > 0, (hasc & is_13c[cj]).sum(0) / np.maximum(nh, 1), 0.0)
        v0 = ((npar >= HIGH_NMIN) & (frac >= HIGH_FRAC) & (r >= HIGH_RMIN) & (ra >= HIGH_FLOOR) & (rh >= HIGH_FLOOR)
              & (ra >= HIGH_XEXP * exp) & (rh >= HIGH_XEXP * exp))
        if klass == "orbitrap":
            pos = D + off / 1e3
            da = np.min([np.abs(pos - a) for a in HIGH_ALIASES.values()], axis=0)
            alias = np.where(np.isnan(pos), False, da < np.abs(off) / 1e3)
        else:
            alias = np.zeros(P, bool)
        cap = (ra <= exp + HIGH_CAP) & (rh <= exp + HIGH_CAP)
        # an element the formula lacks must fit the ion's mass in the number the line implies
        fit = np.ones(P, bool)
        el = _OFFSET_ELEMENT.get(lab)
        if el:
            for i in np.flatnonzero(v0):
                if ions[i].get(el, 0):
                    continue
                k = max(1, int(round(min(ra[i], rh[i]) / _OFFSET_PER_ATOM[lab])))
                fit[i] = element_fits(float(np.nanmedian(PM[:, i])), el, k, HIGH_FIT_PPM[klass], sign[i])
        v4 = v0 & (oth <= HIGH_SLOT_MAX) & ~alias & cap & (o13 <= HIGH_SLOT_MAX) & fit
        per[lab] = dict(n=n, frac=frac, ra=ra, rh=rh, r=r, off=off, oth=oth, o13=o13, alias=alias, cap=cap,
                        fit=fit, v0=v0, v4=v4)
    rows = []
    for i in np.flatnonzero(npar >= HIGH_NMIN):
        hit = [lab for lab in HIGH_OFFSETS if per[lab]["v4"][i]]
        cand = [lab for lab in HIGH_OFFSETS if per[lab]["v0"][i]]
        if hit:
            lab, verdict = _label(hit, per, i, tol * 1e-6 * float(np.nanmedian(PM[:, i]))), "too_high"
        elif cand:
            lab, verdict = _label(cand, per, i, tol * 1e-6 * float(np.nanmedian(PM[:, i]))), "guarded"
        else:
            fin = [x for x in HIGH_OFFSETS if np.isfinite(per[x]["ra"][i])]
            lab = max(fin, key=lambda x: per[x]["ra"][i] / max(exp[i], 1e-12)) if fin else "81Br"
            verdict = "consistent"
        st = per[lab]
        rows.append(dict(neutral_formula=keys[i][0], adduct=keys[i][1], ion=str(pooled["ion"].iat[i]),
                         mz=float(np.nanmedian(PM[:, i])), stamped=bool(stamped[i]), n_spectra=int(npar[i]),
                         n_used=int(st["n"][i]), expected=float(exp[i]), offset=lab, ratio_area=float(st["ra"][i]),
                         ratio_height=float(st["rh"][i]), r=float(st["r"][i]), presence=float(st["frac"][i]),
                         offset_mda=float(st["off"][i]), other_m0=float(st["oth"][i]), other_13c=float(st["o13"][i]),
                         verdict=verdict, veto=verdict == "too_high",
                         note=_high_note(st, i, lab, exp[i], int(npar[i]), ions[i], verdict)))
    if not rows:
        return _empty()
    d = pd.DataFrame(rows)
    d["check"] = "HIGH"
    return d


def _label(labs: list, per: dict, i: int, tol_da: float) -> str:
    """The tallest line among `labs`, named by the heavy spacing nearest it: every
    offset near +2 Da can reach the same peak (within the matching window
    `tol_da` of each other), and the spacing it sits nearest names it (the larger
    ratio alone named a 37Cl line 30Si)."""
    top = max(labs, key=lambda x: per[x]["ra"][i])
    pos = {x: HIGH_OFFSETS[x] + per[x]["off"][i] / 1e3 for x in labs}
    same = [x for x in labs if x == top or abs(pos[x] - pos[top]) <= tol_da]
    return min(same, key=lambda x: abs(per[x]["off"][i]))


def _high_note(st: dict, i: int, lab: str, exp: float, npar: int, ion: dict, verdict: str) -> str:
    ra = st["ra"][i]
    if not np.isfinite(ra):
        return f"no line at any heavy offset in {npar} spectra"
    txt = (f"a {ra:.2f}x line at the {lab} offset ({ra / max(exp, 1e-12):.0f}x the formula's M+2 {exp:.3f}), "
           f"r {st['r'][i]:.2f}, in {st['frac'][i]:.0%} of {npar} spectra")
    if verdict == "too_high":
        el = _OFFSET_ELEMENT.get(lab)
        if el and not ion.get(el, 0):
            return txt + f": the ion carries {el} the formula lacks"
        return txt + ": more than the formula can make"
    if verdict == "guarded":
        why = []
        if st["oth"][i] > HIGH_SLOT_MAX:
            why.append(f"another committed M0 in {st['oth'][i]:.0%} of its spectra")
        if st["o13"][i] > HIGH_SLOT_MAX:
            why.append(f"a plain 13C line in {st['o13'][i]:.0%} of its spectra")
        if st["alias"][i]:
            why.append("nearer a non-isotopic +2 alias than the heavy spacing")
        if not st["cap"][i]:
            why.append(f"above expected + {HIGH_CAP:g}: no isotope envelope")
        if not st["fit"][i]:
            why.append(f"no ion carrying the {_OFFSET_ELEMENT.get(lab, '')} this line needs fits its mass")
        return txt + "; not an isotope line of this ion (" + "; ".join(why) + ")"
    return txt


_FIT_MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956, "S": 31.97207100,
             "Br": 78.9183371, "Cl": 34.96885268, "Si": 27.9769265325}


def element_fits(mz: float, element: str, k: int, ppm: float, sign: float) -> bool:
    """Whether a singly charged ion at `mz` can carry `k` atoms of `element`: some
    CcHhNnOoSs rest (n <= 4, s <= 3, 0 <= h <= 2c + n + 4) makes up the rest of
    its mass within `ppm`. A line at a heavy spacing the ion cannot carry is no
    isotope line of it (an O2 <-> H2S analogue sits 0.06 mDa from 81Br)."""
    if not np.isfinite(mz) or mz <= 0:
        return True
    rest = mz + sign * C.M_E - k * _FIT_MASS[element]
    tol = mz * ppm * 1e-6
    if rest < -tol:
        return False
    if abs(rest) <= tol:
        return True
    m = _FIT_MASS
    c = np.arange(0, int(rest // m["C"]) + 1)
    n = np.arange(0, 5)
    o = np.arange(0, int(rest // m["O"]) + 1)
    sx = np.arange(0, 4)
    cc, nn, oo, ss = np.meshgrid(c, n, o, sx, indexing="ij")
    heavy = cc * m["C"] + nn * m["N"] + oo * m["O"] + ss * m["S"]
    h = np.round((rest - heavy) / m["H"])
    ok = (h >= 0) & (h <= 2 * cc + nn + 4) & (np.abs(heavy + h * m["H"] - rest) <= tol)
    return bool(ok.any())


# --------------------------------------------------------------------------- rule H
def reagent_supply(adducts, element: str) -> int:
    """How many atoms of `element` one reagent ion puts into an ion: the most
    any of the batch's adducts adds (a bromide batch's [M+HBr+Br]- adds 2);
    0 when the batch has no such reagent."""
    from peaky.assignment.tiers import _ion_counts
    return max([0] + [int((_ion_counts("C", a) or {}).get(element, 0)) for a in adducts or ()])


def budget_verdict(neutral: str, element: str, context) -> tuple[bool, str]:
    """(budget_ok, why): the neutral passes the context's element budget with
    `element`'s cap lifted -- the locked halogen is its only violation, if it has
    one (the CF2 exemption's construction, plausibility.demote_off_budget) --
    and the budget's own first violation ("" when none). No context: (True, "")
    -- the budget demote never ran."""
    import dataclasses
    from peaky.chem import contexts as X
    try:
        prof = X.get_context(context) if context else None
    except ValueError:
        prof = None
    if prof is None:
        return True, ""
    ok = bool(X.element_budget(neutral, dataclasses.replace(prof, **{f"max_{element}": 10 ** 6}))[0])
    return ok, str(X.element_budget(neutral, prof)[1] or "")


def count_window(element: str, n: int) -> tuple[float, float]:
    """The pooled area ratio a line at the element's spacing must read for an
    ion carrying `n` atoms of it: [LOCK_LO, LOCK_HI] x n x LOCK_PER_ATOM."""
    return LOCK_LO * n * LOCK_PER_ATOM[element], LOCK_HI * n * LOCK_PER_ATOM[element]


def lock_gates(presence: float, r: float, ratio: float, lo: float, hi: float) -> bool:
    """The line co-varies with the M0 and reads the ion's halogen count: present
    in >= LOCK_FRAC of the M0's spectra, r(log area) >= LOCK_RMIN, the pooled
    area ratio in [lo, hi] (NaN fails every gate)."""
    return bool(presence >= LOCK_FRAC and r >= LOCK_RMIN and lo <= ratio <= hi)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.var(x) <= 0 or np.var(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _partner(S: _Series, codes, pm, pa, ph, shift: float, pcol: int) -> dict:
    """The line at `shift` from each stamp (nearest within LOCK_TOL_PPM of the
    M0's m/z): its presence over the stamps, the pooled area and height ratio
    line / M0, r(log area), the median offset from `shift` (mDa) and how often
    the line is another committed pair's M0."""
    j = S.nearest(codes, pm + shift, ppm_of=pm, ppm=LOCK_TOL_PPM, reach=(-2, -1, 0, 1))
    hit = j >= 0
    n = int(hit.sum())
    out = dict(n=n, frac=n / max(len(pm), 1), ra=np.nan, rh=np.nan, r=np.nan, off=np.nan, oth=0.0)
    if n:
        jj = j[hit]
        la, lh = S.a[jj], S.h[jj]
        out.update(ra=float(la.sum() / pa[hit].sum()), rh=float(lh.sum() / ph[hit].sum()),
                   r=_pearson(np.log(pa[hit]), np.log(la)),
                   off=float(np.median(S.mz[jj] - pm[hit] - shift)) * 1e3,
                   oth=float(((S.role[jj] == "M0") & (S.pair_code[jj] != pcol)).mean()))
    return out


def _heavy(S: _Series, codes, pm, pa, ph, pcol: int) -> dict:
    """{element: (holds, lighter-line stats)}: the M0 is itself the heavy line of
    a lighter one -- a line one element spacing below it, present in >= LOCK_FRAC
    of the stamps, r >= LOCK_RMIN, the M0 / lighter area ratio in the count
    window of 1..LOCK_HEAVY_N of that element."""
    out = {}
    for el, D in LOCK_D.items():
        st = _partner(S, codes, pm, pa, ph, -D, pcol)
        ratio = 1.0 / st["ra"] if np.isfinite(st["ra"]) and st["ra"] > 0 else np.nan
        holds = any(lock_gates(st["frac"], st["r"], ratio, *count_window(el, k)) for k in LOCK_HEAVY_N[el])
        out[el] = (bool(holds), dict(st, ratio=ratio))
    return out


def silicon_window(el: str, mz: float) -> bool:
    """Whether a 30Si line can pass for the element's lock partner at an M0 of
    `mz`: only Cl's (30Si sits 0.206 mDa below 37Cl), and only from LOCK_SI_MZ on,
    where that is within LOCK_TOL_PPM of the M0's m/z."""
    return el == "Cl" and bool(mz >= LOCK_SI_MZ)


def m1_line(ion: dict) -> tuple[float, float]:
    """(height, position) of the ion's own +1 line: its +1 isotopes (13C, 2H,
    15N, 17O, 33S, 29Si) per atom relative to its all-light line, summed, and
    their height-weighted spacing above the M0 (D13C where it has none)."""
    parts = [(ion.get(el, 0) * p / lines[0][1], d) for el, lines in _ISO.items() for d, p, *_x in lines[1:]
             if d < 1.5 and ion.get(el, 0) > 0]
    h = sum(x for x, _d in parts)
    return float(h), (float(sum(x * d for x, d in parts) / h) if h > 0 else D13C)


def _silicon(S: _Series, codes, pm, pa, ph, ratio: float, ion: dict, rp, pcol: int) -> dict:
    """The silicon test of a Cl lock partner at `ratio` x the M0 (the 2026-09-28
    decision, "the 29Si line decides"; the blended reading revised 2026-09-29).
    Read as 30Si, the line makes n = ratio / 30Si-per-atom silicons, whose 29Si
    line is n x 29Si-per-atom at D29SI, 3.79 mDa below 13C: a chlorinated ion
    carries no such line, a Si-rich ion making the partner from 30Si must.

    Resolved (the width model's FWHM at the ion's +1 m/z < D13C - D29SI): the
    29Si line itself, looked for as the lock partner is -- within LOCK_TOL_PPM,
    present in >= LOCK_FRAC of the stamps, r(log area) >= LOCK_RMIN -- at >=
    LOCK_SI_FRAC of its expected area. Where no such line is present (in <
    LOCK_FRAC of the stamps) the peak picker may have reported the two lines as
    one after all (two Gaussians at a 13C / 29Si height ratio of ~0.4 part only
    from ~1.15 FWHM, refute B6), and the +1 region decides as where they blend
    (mode `unparted`).

    Blended (no width model: blended) and unparted: the +1 region (D29SI - 1
    ppm .. D13C + 1 ppm, every peak in it) must be present in >= LOCK_FRAC of
    the stamps and co-vary with the M0 (r(log region area, log M0 area) >=
    LOCK_RMIN over the spectra it is present in: a Si-rich ion's 29Si line is
    its own isotope line, as the partner is; an anti-correlated region is not),
    and then its POSITION decides: refused where the region's area-weighted
    position, the median over the spectra, sits >= LOCK_SI_FRAC of the way from
    the reading's own +1 position (`m1_line`) toward the blend its own +1 line
    and the implied 29Si line make. Position, not height: whatever the peak
    picker reports for a blend -- a full-area centroid, or one peak at its apex
    carrying part of its area -- a Si-rich ion's +1 peak sits toward 29Si, a
    chlorinated ion's at its own position; the region's area does not survive
    that way (an apex carries part of it, and a carbon-rich reading's own +1
    line absorbs the rest), so the excess test built first -- excess AND
    position -- locked dim misreads rule C cannot read (refute round 2). The
    exception (the Si-reading guard): a reading that itself carries 1-2 Si
    (Si >= 3 is never locked) has its own 29Si in its +1 line, which pulls the
    half-way mark toward 29Si; it is refused on the position OR the excess --
    the region reading >= LOCK_SI_FRAC of the implied 29Si area above the
    reading's own +1 line (pooled, sum over sum). An OR, not an AND (the
    decision's "a Si reading must ALSO show the excess", read as OR and
    confirmed 2026-09-29): as an AND the guard would ask a Si reading for more
    than a Si-free one and reopen the Si2 corner it closes -- 393 of 852
    synthetic blended scenario-2 / D7-urea cases (every one a Si1 / Si2
    reading) no longer refused, the dim scenario-2 Si1 and Si2 readings rule C
    cannot read (excess 0.87x / 0.72x of the need) and the Si2 corner (its
    position 0.06 mDa short of the mark) among them, and on the siloxane
    control the 73 Si readings the position refuses no longer refused by the
    silicon test (115 -> 42; rule C still refutes 64 of them, so 9 more
    misreads go unrefuted: 61 -> 70). The guard
    asks no position, so it can refuse a real Si1-2 Cl reading on any excess
    in its +1 region, a neighbour's line at the 13C position included. Where
    both criteria hold, the note names the position. Areas, not heights: the
    partner's ratio and the silicons it implies are area ratios, and a blend's
    height is not the sum of its lines' either.

    `holds`: the M+1 region shows the Si reading; `seen` is the 29Si area found
    (resolved) or the region's excess over the reading's own +1 line (blended /
    unparted); `pos` / `exc` which blended criterion is met; `si_ion` the
    reading's own Si count."""
    n_si = ratio / LOCK_SI_PER_ATOM["30Si"]
    e29 = n_si * LOCK_SI_PER_ATOM["29Si"]
    mz = float(np.median(pm))
    mode = "blended"
    if rp is not None and rp.fwhm(mz + D13C) < D13C - D29SI:
        st = _partner(S, codes, pm, pa, ph, D29SI, pcol)
        if st["frac"] >= LOCK_FRAC:
            holds = st["r"] >= LOCK_RMIN and st["ra"] >= LOCK_SI_FRAC * e29
            return dict(n=n_si, expected=e29, seen=st["ra"], mode="resolved", holds=bool(holds),
                        presence=st["frac"], r=st["r"])
        # no 29Si line present where the width model parts it from 13C: the picker may
        # not have parted them after all -- the +1 region decides, as where they blend
        mode = "unparted"
    m1, c1 = m1_line(ion)
    blend = (m1 * c1 + e29 * D29SI) / (m1 + e29)
    mid = pm + (D13C + D29SI) / 2
    lo, hi = S.window(codes, mid, ((D13C - D29SI) / 2 + pm * LOCK_TOL_PPM * 1e-6) / mid * 1e6)
    hit = hi > lo
    area, pos = np.zeros(len(pm)), np.full(len(pm), np.nan)
    for i in np.nonzero(hit)[0]:
        a = S.a[lo[i]:hi[i]]
        area[i] = a.sum()
        pos[i] = float((a * (S.mz[lo[i]:hi[i]] - pm[i])).sum() / area[i])
    excess = float(area[hit].sum() / pa[hit].sum()) - m1 if hit.any() else np.nan
    at = float(np.median(pos[hit])) if hit.any() else np.nan
    r = _pearson(np.log(pa[hit]), np.log(area[hit]))
    at_max = c1 - LOCK_SI_FRAC * (c1 - blend)
    si_ion = int(ion.get("Si", 0))
    pos, exc = bool(at <= at_max), bool(excess >= LOCK_SI_FRAC * e29)
    holds = hit.mean() >= LOCK_FRAC and r >= LOCK_RMIN and (pos or (si_ion > 0 and exc))
    return dict(n=n_si, expected=e29, seen=excess, mode=mode, holds=bool(holds), presence=float(hit.mean()),
                r=r, at=at, at_max=at_max, pos=pos, exc=exc, si_ion=si_ion)


def _lock(S: _Series, pooled: pd.DataFrame, klass: str, prof=None, context=None, rp=None) -> pd.DataFrame:
    adducts = (getattr(prof, "adducts", None) if prof is not None else None) or sorted(set(pooled["adduct"]))
    supply = {el: reagent_supply(adducts, el) for el in LOCK_D}
    pid = {k: i for i, k in enumerate(S.pair_index)}
    rows = []
    for r in pooled.itertuples(index=False):
        ion = ion_counts(r.neutral_formula, r.adduct, r.ion)
        els = [el for el in LOCK_D if ion.get(el, 0) > 0]
        if not els:
            continue
        el = els[0]
        n_x = int(ion.get(el, 0))
        exp = n_x * LOCK_PER_ATOM[el]
        lo, hi = count_window(el, n_x)
        g = S.m0.get((r.neutral_formula, r.adduct))
        stamped = g is not None and len(g) > 0
        npar = int(len(g)) if stamped else 0
        st, heavy, si = dict(n=0, frac=0.0, ra=np.nan, rh=np.nan, r=np.nan, off=np.nan, oth=0.0), {}, {}
        mz = float(r.mz) if np.isfinite(r.mz) else np.nan
        ok = False
        if stamped:
            codes, pm = g["code"].to_numpy(), g["mz"].to_numpy(float)
            pa, ph = g["area"].to_numpy(float), g["height"].to_numpy(float)
            mz = float(np.median(pm))
            pcol = pid.get((r.neutral_formula, r.adduct), -2)
            st = _partner(S, codes, pm, pa, ph, LOCK_D[el], pcol)
            heavy = _heavy(S, codes, pm, pa, ph, pcol)
            ok = lock_gates(st["frac"], st["r"], st["ra"], lo, hi)
            if ok and silicon_window(el, mz):
                si = _silicon(S, codes, pm, pa, ph, st["ra"], ion, rp, pcol)
        try:
            shift = mz - C.ion_mz(r.neutral_formula, r.adduct)
        except Exception:
            shift = np.nan
        hv = [e for e, (h, _x) in heavy.items() if h]
        why = ""
        if ion.get("Si", 0) > LOCK_SI_MAX:
            verdict, why = "untestable", f"the ion carries Si{ion['Si']} (rule H never locks Si >= {LOCK_SI_MAX + 1})"
        elif len(els) > 1:
            verdict, why = "untestable", "the ion carries both Cl and Br (its M+2 is a 37Cl / 81Br blend)"
        elif not stamped:
            verdict, why = "untestable", "no M0 stamp in the batch series"
        elif not np.isfinite(shift) or abs(shift) > LOCK_STAMP_MAX_DA:
            verdict, why = "untestable", (f"stamped {shift:+.3f} Da from the all-light ion: a heavy isotopologue"
                                          if np.isfinite(shift) else "no ion m/z for this reading")
        elif npar < LOCK_NMIN:
            verdict, why = "untestable", f"stamped in {npar} spectra (needs {LOCK_NMIN})"
        elif supply[el] and n_x <= supply[el]:
            verdict, why = "reagent", (f"the batch's reagent puts up to {supply[el]} {el} in an ion and this one "
                                       f"carries {n_x}: the line may be the reagent's")
        elif not ok:
            verdict = "no_lock"
        elif si.get("holds"):
            verdict, why = "si_rich", _si_why(si)
        elif hv:
            verdict, why = "heavy", "the M0 is itself a heavy line: " + ", ".join(
                f"{heavy[e][1]['ratio']:.2f}x the line one {LOCK_OFFSET[e]} spacing below it" for e in hv)
        else:
            verdict = "lock"
        budget_ok, budget_why = budget_verdict(r.neutral_formula, el, context)
        rows.append(dict(
            neutral_formula=r.neutral_formula, adduct=r.adduct, ion=str(r.ion), mz=mz, stamped=stamped,
            n_spectra=npar, n_used=int(st["n"]), verdict=verdict, veto=False, lock=verdict == "lock",
            offset=LOCK_OFFSET[el], ratio_area=st["ra"], ratio_height=st["rh"], r=st["r"], presence=st["frac"],
            offset_mda=st["off"], other_m0=st["oth"], expected=exp, element=el, n_halogen=n_x,
            ratio_lo=lo, ratio_hi=hi, heavy_cl=bool(heavy.get("Cl", (False,))[0]),
            heavy_br=bool(heavy.get("Br", (False,))[0]), budget_ok=budget_ok, budget_why=budget_why,
            si_n=si.get("n", np.nan), si29_expected=si.get("expected", np.nan), si29_seen=si.get("seen", np.nan),
            si29_mode=si.get("mode", ""), note=_lock_note(st, el, n_x, lo, hi, npar, mz, verdict, why, si)))
    if not rows:
        return _empty()
    d = pd.DataFrame(rows)
    d["check"] = "H"
    d["n_halogen"] = d["n_halogen"].astype("Int64")
    return d


def _si_text(si: dict) -> str:
    seen = f"{si['seen']:.2f}x" if np.isfinite(si["seen"]) else "none"
    return f"Si{si['n']:.1f} reading of the line ({seen} of {si['expected']:.2f}x, {si['mode']})"


def _si_why(si: dict) -> str:
    """A si_rich row's reason: the 29Si line found (resolved), or what the +1
    region shows (blended / unparted): its position (and the half-way mark) --
    also where a Si reading's excess holds too -- or else a Si reading's
    excess."""
    why = f"the M+1 region carries the 29Si line of a {_si_text(si)}"
    if si["mode"] == "resolved":
        return why
    if si["pos"]:
        at, at_max = (round((x - D29SI) * 1e3, 2) + 0.0 for x in (si["at"], si["at_max"]))    # no "-0.00"
        return why + (f": its +1 peak sits {at:.2f} mDa above 29Si, at least half-way toward that reading's blend "
                      f"(<= {at_max:.2f})")
    return why + (f": the reading carries Si{si['si_ion']} and its +1 peak reads at least half that 29Si above the "
                  f"reading's own +1 line")


def _lock_note(st: dict, el: str, n_x: int, lo: float, hi: float, npar: int, mz: float, verdict: str,
               why: str, si: dict | None = None) -> str:
    lab = LOCK_OFFSET[el]
    if not np.isfinite(st["ra"]):
        txt = f"no line at the {lab} offset in {npar} spectra" if npar else ""
    else:
        ppm = st["off"] / mz * 1e3 if np.isfinite(mz) and mz > 0 else np.nan
        txt = (f"a {st['ra']:.2f}x line at the {lab} offset ({n_x} {el}: {lo:.2f}-{hi:.2f}), r {st['r']:.2f}, "
               f"in {st['frac']:.0%} of {npar} spectra, {ppm:+.2f} ppm")
    if verdict == "no_lock" and np.isfinite(st["ra"]):
        miss = []
        if st["frac"] < LOCK_FRAC:
            miss.append(f"present in < {LOCK_FRAC:.0%}")
        if not st["r"] >= LOCK_RMIN:
            miss.append(f"r < {LOCK_RMIN:g}")
        if not lo <= st["ra"] <= hi:
            miss.append(f"outside the {n_x} {el} window")
        why = "; ".join(miss)
    elif verdict == "lock":
        why = "no lighter line makes the M0 a heavy isotopologue"
        if si:
            why += f"; no 29Si line of a {_si_text(si)}"
    return "; ".join(x for x in (txt, why) if x)


# --------------------------------------------------------------------------- the table
def measure(ts: pd.DataFrame | None, frames: dict, prof=None, *, resolution=None, mass_scale=None,
            x_edge: float = 1.0, context: str | None = None, edge_cps: float | None = None,
            log=print) -> pd.DataFrame:
    """The isotope-check table: one row per tested pooled pair and check (module
    docstring). `resolution` is the batch's width model (chem.resolution.Resolution
    or its as_dict; it decides the instrument class and REQ's observable lines),
    `mass_scale` the batch's traces.MassScale (sigma_ppm, stamp_ppm), `x_edge` the
    batch's height_cutoff_x_edge (the height gate each spectrum's floor is: its
    noise edge, the 1st-percentile height, x x_edge -- the per-file
    height_gate_cps where a file is also a ledger), `context` the batch's
    assignment context (rule H's `budget_ok`: the element budget each file's
    plausibility stage demoted against; None = no budget), `edge_cps` the
    batch's typical detection edge (PassConfig.noise_edge_batch_cps: the TOF
    REQ test's floor is k_detect x it; None = the median of the spectra's own
    edges). `prof` is the batch's reagent profile (rule C's 14N exemption, rule
    H's reagent supply). Empty with a header when the batch has no time series
    or no committed pair."""
    if ts is None or not len(ts):
        return _empty()
    pooled = _pooled(frames)
    if pooled.empty:
        return _empty()
    klass = instrument_class(resolution)
    rp = _resolution(resolution)
    sigma, stamp = _scale(mass_scale)
    S = _Series(ts)
    if not S.n:
        return _empty()
    try:
        x_edge = float(x_edge)
    except (TypeError, ValueError):     # an unresolved 'auto' multiple reads as the edge itself
        x_edge = 1.0
    x_edge = x_edge if np.isfinite(x_edge) and x_edge > 0 else 1.0
    parts = []
    if klass == "orbitrap":
        parts.append(_rule_c(S, pooled, prof))
    if rp is not None:
        parts.append(_req_tof(S, pooled, rp, sigma, stamp, edge_cps, log=log) if klass == "tof"
                     else _req(S, pooled, klass, rp, sigma, stamp, x_edge, log=log))
    parts.append(_high(S, pooled, klass))
    parts.append(_lock(S, pooled, klass, prof, context, rp))
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return _empty()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["instrument"] = klass
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["veto"] = out["veto"].fillna(False).astype(bool)
    out["stamped"] = out["stamped"].fillna(False).astype(bool)
    out["lock"] = out["lock"].fillna(False).astype(bool)
    out["note"] = out["note"].fillna("")
    order = {c: i for i, c in enumerate(CHECKS)}
    out = out[list(TABLE_COLUMNS)]
    out = out.assign(__o=out["check"].map(order)).sort_values(["__o", "neutral_formula", "adduct"], kind="mergesort")
    out = out.drop(columns="__o").reset_index(drop=True)
    log(f"[iso_checks] {klass}-class batch: "
        + "; ".join(f"{CHECK_NAME[c]} {int((out['check'] == c).sum())} tested, "
                    + (f"{int((out['veto'] & (out['check'] == c)).sum())} refuted" if c in VETO_CHECKS
                       else f"{int((out['lock'] & (out['check'] == c)).sum())} locked") for c in CHECKS)
        + f" -> {len(veto(out))} pair(s) vetoed, {len(lock(out))} locked"
        + ("" if klass == "orbitrap" else " (TOF-class: no rule C)")
        + ("" if rp is not None else " (no width model: no REQ)"))
    return out


def veto(table: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): note} the checks refute -- one joined note per pair
    ('rule C: ...; REQ: ...') in the order of CHECKS."""
    if table is None or not len(table) or "veto" not in table.columns:
        return {}
    t = table[table["veto"].map(_truth)]
    out: dict = {}
    order = {c: i for i, c in enumerate(CHECKS)}
    t = t.assign(__o=t["check"].map(lambda c: order.get(str(c), len(order)))).sort_values("__o", kind="mergesort")
    for n, a, c, x in zip(t["neutral_formula"], t["adduct"], t["check"], t["note"]):
        k = (str(n), str(a))
        piece = f"{CHECK_NAME.get(str(c), str(c))}: {x}" if isinstance(x, str) and x else CHECK_NAME.get(str(c), str(c))
        out[k] = f"{out[k]}; {piece}" if k in out else piece
    return out


def lock(table: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): {'element', 'n', 'budget_ok', 'note'}} of the rule H
    rows that lock -- {} for a table written before rule H (no `lock` column,
    no H row)."""
    if table is None or not len(table) or "lock" not in table.columns or "check" not in table.columns:
        return {}
    t = table[(table["check"].astype(str) == "H") & table["lock"].map(_truth)]
    out: dict = {}
    for r in t.itertuples(index=False):
        n = pd.to_numeric(getattr(r, "n_halogen", np.nan), errors="coerce")
        el = getattr(r, "element", "")
        out[(str(r.neutral_formula), str(r.adduct))] = {
            "element": el if isinstance(el, str) else "", "n": int(n) if pd.notna(n) else 0,
            "budget_ok": _truth(getattr(r, "budget_ok", False)),
            "note": str(r.note) if isinstance(r.note, str) else ""}
    return out


def facts(table: pd.DataFrame | None) -> dict | None:
    """What the pooled pair facts read (evidence._level_pairs, its `iso=`): {'veto': {(n, a): note},
    'lock': {(n, a): {...}}} -- the refutations and rule H's locks; None for an
    empty table (no time series, nothing committed)."""
    if table is None or not len(table):
        return None
    return {"veto": veto(table), "lock": lock(table)}


def _truth(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False


def summary(table: pd.DataFrame | None, resolution=None) -> dict:
    """The funnel, for batch_summary.json: per check the pairs tested and each
    verdict's count (the vetoes of C / REQ / HIGH, the locks of rule H), and the
    pairs vetoed and locked in all."""
    klass = instrument_class(resolution)
    if table is None or not len(table):
        return {"instrument": klass, "tested": 0, "vetoed_pairs": 0, "locked_pairs": 0}
    out = {"instrument": klass, "tested": int(len(table)), "vetoed_pairs": len(veto(table)),
           "locked_pairs": len(lock(table))}
    for c in CHECKS:
        t = table[table["check"] == c]
        v = t["verdict"].astype(str)
        head = ({"vetoed": int(t["veto"].map(_truth).sum())} if c in VETO_CHECKS else
                {"locked": int(t["lock"].map(_truth).sum()) if "lock" in t.columns else 0})
        out[c] = {"tested": int(len(t)), **head, **{k: int((v == k).sum()) for k in VERDICTS[c]}}
    return out


# --------------------------------------------------------------------------- the TOF merged-row gates
#: the merged row's tier_reason mark of a species lock_known_species decided
KNOWN_LOCK_MARK = "known species decided once for the batch"
#: ... and of a lock that displaced the vote's winner (its note goes on "; kept
#: over the N-file X reading", the displaced reading heading `alternatives`)
_KEPT_OVER = re.compile(re.escape(KNOWN_LOCK_MARK) + r"[^|]*; kept over the \d+-file ")


def _merge_ppm(mass_scale) -> float:
    """The batch's merge window (traces.MassScale.merge_ppm, else its tol_ppm, else 6)."""
    if mass_scale is None:
        return 6.0
    get = mass_scale.get if isinstance(mass_scale, dict) else (lambda k, d=None: getattr(mass_scale, k, d))
    for k in ("merge_ppm", "tol_ppm"):
        try:
            v = float(get(k))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            return v
    return 6.0


def _tallest_height(S: _Series, codes, targets, ppm) -> np.ndarray:
    """The tallest line within +-ppm of each (spectrum, target), 0 where none (vectorised)."""
    lo, hi = S.window(codes, targets, ppm)
    k = hi - lo
    out = np.zeros(len(lo))
    for off in range(int(k.max()) if len(k) else 0):
        sel = k > off
        out[sel] = np.maximum(out[sel], S.h[lo[sel] + off])
    return out


def doublets(S: _Series, mzs, ppm: float) -> pd.DataFrame:
    """Per merged line (m/z): over the spectra showing it (a line within `ppm`),
    how often it stands at TOF_DBL_LO..TOF_DBL_HI x the line one 79Br -> 81Br
    spacing below it (LOCK_D['Br']) -- the 81Br partner of a Br1 ion, not an
    M0. Columns n_spectra, n_band, share (n_band / n_spectra; 0 where none),
    median_ratio."""
    mzs = np.asarray(mzs, dtype=float)
    m = len(mzs)
    if not m or not S.n:
        return pd.DataFrame({"n_spectra": np.zeros(m, int), "n_band": np.zeros(m, int),
                             "share": np.zeros(m), "median_ratio": np.full(m, np.nan)})
    codes = np.repeat(np.arange(S.n), m)
    tg = np.tile(mzs, S.n)
    P = _tallest_height(S, codes, tg, ppm).reshape(S.n, m)
    Q = _tallest_height(S, codes, tg - LOCK_D["Br"], ppm).reshape(S.n, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((P > 0) & (Q > 0), P / np.where(Q > 0, Q, 1.0), np.nan)
    n_p = (P > 0).sum(axis=0)
    band = ((r >= TOF_DBL_LO) & (r <= TOF_DBL_HI)).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(r, axis=0) if S.n else np.full(m, np.nan)
    return pd.DataFrame({"n_spectra": n_p, "n_band": band, "share": np.where(n_p > 0, band / np.maximum(n_p, 1), 0.0),
                         "median_ratio": med})


def tof_m2_gates(merged: pd.DataFrame, table: pd.DataFrame | None, ts: pd.DataFrame | None, *,
                 resolution=None, mass_scale=None, log=print) -> dict:
    """The TOF ion-M+2 gates on the merged ledger, after the stamp (in place).

    TOF-class batches only (a width model whose class is 'tof'; without one
    nothing runs). Two demotions, Assigned -> Candidate, no re-vote (the winner
    and its reading stay; the row says why):

      * REQ: the merged winner's pair is refuted by REQ's TOF branch -- the
        ion's own M+2 line absent over the batch (`_req_tof`). This includes a
        species lock_known_species decided (it runs before the stamp, so it
        cannot read REQ): a known reading whose own envelope the batch refutes
        is not Assigned. A lock that displaced the vote's winner is demoted,
        not undone -- the row keeps the known reading, now Candidate, and the
        vote's reading the lock put at the head of `alternatives` stays there
        (the row's note names it). The per-file test (tiers.apply_tof_m2)
        usually gets there first: a 'known' C30 chlorinated paraffin [M+Br]-
        on a bromide TOF batch (one Br's M+2 line where BrCl4 predicts 2.3x,
        no 13C line: the reagent's water cluster at the same nominal mass) was
        Candidate in every file, so the vote already made it Candidate.
      * the doublet: the row's line stands at TOF_DBL_LO..TOF_DBL_HI x the line
        one 81Br spacing below it in >= TOF_DBL_SHARE of the spectra showing it
        (`doublets`, within the batch's merge window) -- it is that line's 81Br
        partner, not an M0 -- unless its reading's ion carries Br / Cl whose own
        M+2 REQ sees in >= TOF_DBL_OWN_SEEN of its testable spectra. A known
        lock is left to REQ.

    Returns counts for batch_summary['merge_gates']['tof_m2']."""
    out = {"ran": False, "req_demoted": 0, "known_demoted": 0, "doublet_demoted": 0, "doublet_exempt": 0}
    rp = _resolution(resolution)
    if rp is None or instrument_class(rp) != "tof":
        out["skipped"] = "no width model" if rp is None else "not a TOF-class batch"
        return out
    out["ran"] = True
    if merged is None or not len(merged) or "tier" not in merged.columns:
        return out
    from peaky.assignment.cleanup import _note
    from peaky.assignment.tiers import _ion_counts
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    req = (table[table["check"].astype(str) == "REQ"] if table is not None and len(table) and "check" in table.columns
           else pd.DataFrame(columns=list(TABLE_COLUMNS)))
    refuted = {(str(n), str(a)): str(x) for n, a, v, x in zip(req["neutral_formula"], req["adduct"],
                                                               req["veto"].map(_truth), req["note"]) if v}
    seen = {(str(n), str(a)): (float(f) if pd.notna(f) else np.nan)
            for n, a, f in zip(req["neutral_formula"], req["adduct"], pd.to_numeric(req["det_frac"], errors="coerce"))}

    def _assigned(i):
        return str(merged.at[i, "tier"]) == "Assigned"

    def _locked(i):
        return KNOWN_LOCK_MARK in str(merged.at[i, "tier_reason"])

    def _overruled(i) -> str:
        """The note of a REQ-refuted known-species decision; a lock that displaced
        the vote's winner names the reading it put at the head of `alternatives`."""
        why = "; this overrules the known-species decision"
        if not _KEPT_OVER.search(str(merged.at[i, "tier_reason"])) or "alternatives" not in merged.columns:
            return why
        alt = merged.at[i, "alternatives"]
        alt = "" if alt is None or (not isinstance(alt, str) and pd.isna(alt)) else str(alt)
        head = alt.split("; ")[0].strip()
        return why + (f" (demoted, not undone: the vote's reading it was kept over, {head}, heads "
                      "`alternatives`)" if head else "")

    for i in merged.index:
        k = (str(merged.at[i, "neutral_formula"]), str(merged.at[i, "adduct"]))
        if not _assigned(i) or k not in refuted:
            continue
        locked = _locked(i)
        merged.at[i, "tier"] = "Candidate"
        _note(merged, i, "Candidate: the batch refutes the ion's own M+2 line (REQ: " + refuted[k] + ")"
              + (_overruled(i) if locked else ""))
        out["req_demoted"] += 1
        out["known_demoted"] += int(locked)
    if ts is None or not len(ts):
        return out
    S = _Series(ts)
    idx = [i for i in merged.index if _assigned(i) and not _locked(i)]
    if not idx or not S.n:
        return out
    ppm = _merge_ppm(mass_scale)
    mzs = pd.to_numeric(merged.loc[idx, "mz"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(mzs)
    idx = [i for i, g in zip(idx, ok) if g]
    d = doublets(S, mzs[ok], ppm)
    for i, r in zip(idx, d.itertuples(index=False)):
        if not (r.n_spectra > 0 and r.share >= TOF_DBL_SHARE):
            continue
        k = (str(merged.at[i, "neutral_formula"]), str(merged.at[i, "adduct"]))
        ion = _ion_counts(*k) or {}
        own = seen.get(k, np.nan)
        if (ion.get("Br", 0) or ion.get("Cl", 0)) and np.isfinite(own) and own >= TOF_DBL_OWN_SEEN:
            out["doublet_exempt"] += 1
            continue
        merged.at[i, "tier"] = "Candidate"
        _note(merged, i, f"Candidate: the line is the 81Br partner of the line {LOCK_D['Br']:.4f} Da below it "
                         f"({TOF_DBL_LO:g}-{TOF_DBL_HI:g}x it in {int(r.n_band)} of the {int(r.n_spectra)} spectra "
                         f"showing it, median {r.median_ratio:.2f}x), not an M0")
        out["doublet_demoted"] += 1
    log(f"[tof_m2] TOF ion-M+2 gates on the merged ledger: {out['req_demoted']} Assigned -> Candidate by REQ "
        f"({out['known_demoted']} known-species decisions overruled), {out['doublet_demoted']} by the 81Br doublet "
        f"({out['doublet_exempt']} halogen readings exempt: their own M+2 seen)")
    return out
