"""Fluorine by evidence: the CF2 ladder, and the two lines that refute it.

19F is the only stable fluorine isotope, so no isotope line can confirm a
fluorine count, and two refusals guard F readings for that reason: the tier
engine caps a partially fluorinated reading (F < 2H) at Candidate
(tiers.F_H_COHERENCE), and cleanup.demote_unconfirmed_fluorine flags an F>=4
reading below assignability. What a mass spectrum CAN show for fluorine is a
CF2 ladder: the same remaining composition one, two, three CF2 (49.99681 Da)
lighter or heavier. A fluorinated homologous series is what a PTFE / PFAS source
emits, and between two members a CF2 step pins two fluorines: at Orbitrap
accuracy no other composition difference sits within the window (the nearest,
CH3Cl and O3H2, are 4.5 and 3.6 mDa away). This module reads that ladder per
file and hands the two refusals a verdict:

* LADDER: the row's ion formula +/- k CF2 (k = 1..LADDER_STEPS, every member
  still fluorinated) is looked for as a picked line at the row's own measured
  m/z +/- k x 49.99681 (the step is measured between two lines of one spectrum,
  so the file's calibration offset cancels), within `window_ppm` and at a height
  within LADDER_HEIGHT_X of the row's either way. A line the ledger commits as an
  M0 of exactly that ion composition is a COMMITTED member.
* lift: >= LADDER_MIN_NEIGHBOURS such lines, at least one of them an adjacent
  member (+/- 1 CF2) -- the reading is not refused. It stays Candidate unless
  >= LADDER_MIN_COMMITTED members of its ladder (itself included) are committed
  and, on a 15N-nitrate ledger, its 15N alias has been tested (`full`): then the
  normal tier rules apply.
* veto: the ladder proves a fluorinated CF2 series, not the composition of its
  end group, and two lines refute a reading on it (the refusal then stands):
    - the 14N twin of its 15N alias. On a 15N-nitrate run C2F3 and H2^NO4 differ
      by 0.21 mDa (unresolvable), so every F reading with C >= 2 and F >= 3 has a
      same-mass reading with one more ^N, and the swap preserves the ladder. The
      line one 15N-14N step (0.99703 Da) below carries the 14N share of a
      labelled ion: f = (1 - purity) / purity per labelled atom from the reagent,
      plus whatever 14N nitrate the air brings. Only a reading that carries NO ^N
      is tested: it has no such line of its own, so a line there inside
      ALIAS_TWIN_BAND of the alias's f is the alias's twin. A reading that carries
      n ^N has its own twin at ~n f, and the share is not known well enough to
      tell n f from the alias's (n + 1) f: on the campaign below the own twin of
      the truth's one-^N F readings read 0.01-0.1 of the line (up to 5x the
      98 % purity's f, 40 % of them above 2 f): a test against the nominal f
      refuted 52 truth readings to catch one alias, which the test on ^N-free
      readings catches too;
    - the 13C line: the row's own 13C line (its child, or an unclaimed line at
      +1.00336), at or above the file's detection floor, whose carbon count
      contradicts the formula's (iso_response.contradicts, the test the isotope
      audit clears on). It cannot split the 15N alias: the swap moves two carbons,
      inside contradicts' tolerance (max(2.5, 0.35 n)) at every n.
* the alias left open: on a 15N-nitrate ledger a reading with C >= 2 and F >= 3
  whose 15N alias no line has tested -- it carries ^N (its own twin hides the
  alias's), or its alias's twin would read under the floor (lower band edge) --
  is never `full`: it is lifted (no refusal) but stays Candidate. An alias that
  cannot exist is not open: the swap lowers the DBE by one, so a saturated
  reading (DBE 0, the CnH(2n+2-f)Ff telomer chains) has none (alias_admissible:
  the closed-shell DBE and structural-O gates the grid applies). On the
  campaign measured below the untested alias was the commonest wrong full-ladder
  reading: 8 of the 15 rows a full ladder made wrong-Assigned were exactly the
  C2F3 <-> H2^NO4 swap, and their twins were predicted at 1.7-10 cps against
  floors of 62-68.

The series hold (`series_alias_holds`) is the same alias test turned on a series
step. On a 15N-nitrate run C8S and H ^N O7 differ by 0.27 mDa (LABEL_ALIASES), so
every organosulfur reading with C >= 8 has a same-ion reading with one more ^N --
an [M-H]- organosulfur ion is the [M+^NO3]- cluster of its neutral minus C8S plus
O4 -- and both swaps keep every CH2 / O / H2O / CF2 step: a homologous series
reads the alias exactly as it reads the row. A series step or a ladder anchor
(`anchor_peak_id` / `series_unit`) is therefore no corroboration of a reading
whose F count (never testable) or S count (its 34S line neither seen nor
refuted) its alias could be standing in for, while that alias can exist and is
untested or its 14N twin is seen. The tier engine drops the row's series credit
and appends the reason (SERIES_ALIAS_MARK); the evidence scale's `anchor` axis
reads the mark, so the merge vote's class does not count the step either. On the
15N-nitrate field batch above the 128-600 window had made seven C8S aliases of
the truth's [M+^NO3]- readings Assigned that way (a pass-6 gap-fill or pass-4
series step off a pass-3 organosulfate Candidate, class 1 in one file outvoting
the truth's reading in 3-5 files at class 0).

The ladder's 13C counterpart in arbitration -- a 13C carbon-count term between
sub-mDa aliases of the fluorinated family (C8 vs H1F5, 0.16 mDa) -- is
`c13_lines` + `c13_alias_penalties`, called by pass 3 (directors.run_pass3) and
arbitrate (passes/core.py). It is COMPARATIVE: an alias pays when the line reads
its carbon count clearly farther off than the nearest alias's (by >= 2 carbons
and >= 2 sigma of the count read). The absolute refutation (contradicts, whose
tolerance grows as 0.35 n) cannot do this at high C: for C15 vs C23 a perfect
line never refutes the C23 alias.

Measured on the stored per-file ledgers of a 15N-nitrate field campaign
(128-600 window): 52 truth lines on CF2 ladders were refused as 'unconfirmed
fluorine'; their ladder neighbours sit at 0.03-10x the line's height (1st-99th
percentile), so LADDER_HEIGHT_X = 30 leaves out only the extremes. The line test
is specific: at decoy repeat units (CF2 +/- 3-10 mDa) >= 2 ladder lines are found
for 0-1.2 % of the F readings against 76 % at the CF2 step (on a ~10k TOF, with
the 2 ppm window: <= 0.4 % against 42 %).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.assignment import iso_response as IR
from peaky.assignment import ledger as L
from peaky.assignment import satellites as SAT

__version__ = "0.1.0"

#: the repeat unit and its exact mass
CF2 = {"C": 1, "F": 2}
D_CF2 = C.M["C"] + 2 * C.M["F"]            # 49.99681 Da
LADDER_STEPS = 3                           # CF2 steps looked for each way
LADDER_MIN_NEIGHBOURS = 2                  # ladder lines (the row excluded) that lift the refusal ...
LADDER_ADJACENT = 1                        # ... at least one of them this many CF2 steps away (+/-1: a neighbour)
LADDER_MIN_COMMITTED = 3                   # committed members (the row included): the normal tier rules apply
LADDER_PPM = 2.0                           # the step window where no exact-offset one holds (max with 2 x cfg.ppm)
LADDER_HEIGHT_X = 30.0                     # a ladder line reads within 1/30..30x the row's height


class Alias(NamedTuple):
    """A 15N alias: on a 15N-nitrate run the reading's `out` composition and
    the ^N-carrying one the `swap` puts in its place weigh the same to a
    fraction of a mDa, so every reading that carries `out` has a same-ion
    reading with one more ^N."""
    name: str           # the swap as the reasons print it
    element: str        # the element the alias trades away: the count it stands in for
    out: dict           # what a reading must carry for the alias to exist
    swap: dict          # the alias's composition = the reading's + swap

    @property
    def mda(self) -> float:
        """|the swap's mass| in mDa."""
        return abs(sum(C.M[el] * n for el, n in self.swap.items())) * 1e3


#: the 15N aliases: C2F3 out, H2 ^N O4 in (0.21 mDa) -- the ladder's (`verdicts`);
#: C8S out, H ^N O7 in (0.27 mDa) -- an organosulfur reading's (`series_alias_holds`)
ALIAS_C2F3 = Alias("C2F3 -> H2^NO4", "F", {"C": 2, "F": 3}, {"C": -2, "F": -3, "H": 2, "^N": 1, "O": 4})
ALIAS_C8S = Alias("C8S -> H^NO7", "S", {"C": 8, "S": 1}, {"C": -8, "S": -1, "H": 1, "^N": 1, "O": 7})
LABEL_ALIASES = (ALIAS_C2F3, ALIAS_C8S)
ALIAS_OUT = ALIAS_C2F3.out
ALIAS_SWAP = ALIAS_C2F3.swap
D_15N = C.M["^N"] - C.M["N"]               # 0.99703 Da
ALIAS_TWIN_BAND = (0.5, 5.0)               # the twin reads within this x the alias's nominal share f
#: an untested 15N alias holds a full ladder at Candidate (False: the ladder alone
#: decides, as the first build did -- on the 15N-nitrate campaign below, against
#: the hold, that made 366 more merged rows Assigned: 231 more the truth's reading,
#: 16 more against it; against the code before this module, 15 new rows against
#: it, 8 of them exactly the alias)
ALIAS_UNTESTED_HOLDS = True
#: a series step or a ladder anchor does not corroborate a reading whose 15N alias
#: is untested, or whose alias's 14N twin is seen (`series_alias_holds`; False: it
#: does, as before -- see the module note for the batch it was measured on)
ALIAS_SERIES_HOLDS = True
#: the tier_reason mark of a row whose series credit the hold withheld: the tier
#: engine writes it, the evidence scale's `anchor` axis reads it
SERIES_ALIAS_MARK = "series step not counted"
C13_PPM = 5.0                              # the 13C line's window where no exact-offset one holds (audit_sat_ppm)
#: the 13C carbon-count term between sub-mDa aliases of the fluorinated family
ALIAS_MDA = 1.0                            # aliases: ion masses within this of the arbitration's top reading
C13_ALIAS_PENALTY = 0.30                   # eff-score charge on an alias its 13C line reads clearly farther from
C13_REL = 0.12                             # relative read error of a 13C carbon count (contradicts' 0.35 n is ~3x it;
                                           # measured robust sd 0.08 at 3-10x the floor, 0.04 at 10-30x)
C13_K = 2.0                                # the margin, in sigma of the count read ...
C13_MIN_GAP = 2.0                          # ... and never under this many carbons


class Ladder(NamedTuple):
    """One F reading's ladder verdict (`verdicts`)."""
    lines: int          # ladder lines found, the row included
    committed: int      # of them committed as the ladder's own ion composition, the row included
    steps: tuple        # ((k, committed), ...) of the lines found, k in CF2 steps
    veto: str | None    # why the ladder does not lift the refusal (None: it does, given enough lines)
    alias_open: str | None = None   # why its 15N alias is untested (15N-nitrate ledger; None: tested or none)

    @property
    def lift(self) -> bool:
        """The ladder lifts the F refusal: enough lines, one of them an adjacent
        member, no refuting line."""
        return (self.veto is None and self.lines - 1 >= LADDER_MIN_NEIGHBOURS
                and any(abs(k) <= LADDER_ADJACENT for k, _c in self.steps))

    @property
    def full(self) -> bool:
        """... and enough committed members, and no untested 15N alias, for the
        normal tier rules to apply."""
        return (self.lift and self.committed >= LADDER_MIN_COMMITTED
                and (self.alias_open is None or not ALIAS_UNTESTED_HOLDS))

    def text(self) -> str:
        st = ", ".join(f"{k:+d}{'*' if c else ''}" for k, c in self.steps)
        return (f"CF2 ladder of {self.lines} lines ({self.committed} committed; steps {st or 'none'}, "
                "* = committed)")


def _key(counts: dict) -> str:
    return C.format_formula({k: v for k, v in counts.items() if v})


def _ion(ion, neutral=None, adduct=None) -> dict:
    """An ion composition (labelled atoms kept: '^N' is its own key) from the
    row's ion formula, else from its neutral + adduct."""
    cnt = C.parse_formula(ion) if isinstance(ion, str) and ion.strip() else {}
    if not cnt:
        from peaky.assignment import tiers as T     # lazy: tiers imports this module
        cnt = T._ion_counts(neutral, adduct, labelled=True) or {}
    return {k: v for k, v in cnt.items() if v}


def _height_floor(cfg) -> float | None:
    if cfg is None:
        return None
    try:
        f = cfg.audit_floor_cps if getattr(cfg, "audit_floor_cps", None) is not None else cfg.height_cutoff
    except RuntimeError:
        return None
    try:
        f = float(f)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def window_ppm(cfg=None) -> float:
    """The ladder step's window: the exact-offset window on a calibrated
    Orbitrap-class file (satellites.exact_offset_ppm), else max(LADDER_PPM,
    2 x cfg.ppm) -- the link tolerance the GKA chains are detected with."""
    wide = max(LADDER_PPM, 2.0 * float(getattr(cfg, "ppm", 1.0) or 1.0)) if cfg is not None else LADDER_PPM
    return float(SAT.exact_offset_ppm(cfg, wide))


def _c13_ppm(cfg=None) -> float:
    wide = float(getattr(cfg, "audit_sat_ppm", C13_PPM) or C13_PPM) if cfg is not None else C13_PPM
    return float(SAT.exact_offset_ppm(cfg, wide))


class _Lines:
    """Every real picked line of a ledger (synthetic sub-peaks and artefacts
    out), sorted by m/z, with its ledger index."""

    def __init__(self, ledger: pd.DataFrame):
        mz = pd.to_numeric(ledger["mz"], errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(ledger["height"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(mz) & np.isfinite(h) & (h > 0)
        if "synthetic" in ledger.columns:
            ok &= ~ledger["synthetic"].map(L._truthy).to_numpy(dtype=bool)
        if "role" in ledger.columns:
            ok &= ledger["role"].astype(str).to_numpy() != L.ROLE_ARTIFACT
        o = np.argsort(mz[ok], kind="mergesort")
        self.mz, self.h = mz[ok][o], h[ok][o]
        self.idx = np.asarray(ledger.index)[ok][o]

    def near(self, target: float, ppm: float):
        """(ledger index, height) of the line nearest `target` within `ppm`, else None."""
        if not len(self.mz):
            return None
        tol = abs(target) * ppm * 1e-6
        j = int(np.searchsorted(self.mz, target))
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(self.mz) and abs(self.mz[k] - target) <= tol:
                if best is None or abs(self.mz[k] - target) < abs(self.mz[best] - target):
                    best = k
        return None if best is None else (self.idx[best], float(self.h[best]))


def _carries(counts: dict, alias: Alias) -> bool:
    """`counts` hold the alias's `out` composition (the alias can be read off them)."""
    return all(counts.get(el, 0) >= n for el, n in alias.out.items())


def alias_admissible(neutral: dict | None, alias: Alias = ALIAS_C2F3) -> bool:
    """Can the reading's 15N alias exist? Its neutral (`alias`: by default C2F3
    out, H2 ^N O4 in) must pass the closed-shell gate the grid and the labelled
    rescue apply (chemistry.dbe_ok, oxygen_ok). The C2F3 swap lowers the DBE by
    one, so a saturated reading (DBE 0, e.g. a CnH(2n+2-f)Ff telomer chain) has
    no alias; the C8S swap lowers it by eight. True where the neutral is unknown
    or lacks the alias's composition (nothing to rule out)."""
    if not neutral or not _carries(neutral, alias):
        return True
    a = dict(neutral)
    for el, d in alias.swap.items():
        a[el] = a.get(el, 0) + d
    a = {k: v for k, v in a.items() if v}
    return bool(C.dbe_ok(a)[0] and C.oxygen_ok(a)[0])


def _alias_twin(ion: dict, h0: float, lines: _Lines, mz0: float, ppm: float, f: float,
                floor: float | None = None, neutral: dict | None = None,
                alias: Alias = ALIAS_C2F3) -> tuple[str | None, str | None]:
    """The reading's 15N alias (`alias`: by default C2F3 -> H2^NO4) on a
    15N-nitrate ledger (`f` > 0, the reagent's 14N share): (veto, open). veto:
    a line one 15N-14N step below inside ALIAS_TWIN_BAND of f -- the alias's 14N
    twin, where the reading carries no ^N. open: why the alias is UNTESTED --
    the reading carries ^N (its own twin cannot be told from its alias's), or
    the alias's twin would read under the detection floor at the band's lower
    edge. (None, None): no alias (the ion short of the alias's composition, an
    alias whose neutral cannot exist -- `neutral`, alias_admissible --, or not a
    labelled ledger), or tested and absent."""
    if not _carries(ion, alias) or not (f > 0):
        return None, None
    if not alias_admissible(neutral, alias):
        return None, None
    if ion.get("^N", 0):
        return None, (f"the reading carries ^N{ion['^N']}: its own 14N twin (~{ion['^N']} f) "
                      f"cannot be told from its alias's (~{ion['^N'] + 1} f)")
    lo, hi = ALIAS_TWIN_BAND
    hit = lines.near(mz0 - D_15N, ppm)
    if hit is not None and lo * f <= hit[1] / h0 <= hi * f:
        return (f"the 14N twin of its 15N alias ({alias.name}, ^N1) reads {hit[1] / h0:.3f} of the line "
                f"(the alias's share ~{f:.3f}) where the reading carries no ^N"), None
    if floor is None or lo * f * h0 < floor:
        return None, (f"its alias's 14N twin would read ~{lo * f * h0:.0f} cps, under the floor"
                      + (f" ({floor:.0f})" if floor is not None else ""))
    return None, None


def _c13_refutes(ledger: pd.DataFrame, i, ion: dict, lines: _Lines, cfg, kids: dict) -> str | None:
    """The row's 13C line, at or above the detection floor, contradicts its
    carbon count: a reason, else None. `kids`: {parent peak_id: (height,
    area)} of the ledger's 13C children."""
    n_c = ion.get("C", 0)
    floor = _height_floor(cfg)
    if n_c < 1 or floor is None:
        return None
    h0 = float(ledger.at[i, "height"])
    a0 = _num(ledger.at[i, "area"]) if "area" in ledger.columns else None
    k = kids.get(ledger.at[i, "peak_id"])
    if k is not None:
        h_sat, a_sat = k
    else:
        hit = lines.near(float(ledger.at[i, "mz"]) + IR.D13C, _c13_ppm(cfg))
        if hit is None or str(ledger.at[hit[0], "role"]) != L.ROLE_UNEXPLAINED:
            return None
        h_sat = hit[1]
        a_sat = _num(ledger.at[hit[0], "area"]) if "area" in ledger.columns else None
    if not (h0 > 0 and h_sat >= floor):
        return None
    snr0 = _num(ledger.at[i, "signal_to_noise"]) if "signal_to_noise" in ledger.columns else None
    bad, c_est = IR.contradicts(getattr(cfg, "iso_response_fit", None), n_c, h0=h0, h_sat=h_sat,
                                a0=a0, a_sat=a_sat, snr0=snr0, area_mode=bool(getattr(cfg, "iso_area", False)))
    return f"its 13C line reads ~C{c_est:.0f} for C{n_c}" if bad else None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def labelled_nitrate(ledger: pd.DataFrame) -> bool:
    """The ledger commits on the labelled nitrate cluster ([M+^NO3]-): the run
    whose ^N makes the C2F3 / H2^NO4 alias (a 15N-ammonium run's [M+^NH4]+
    does not: its alias would need a ^N with four O)."""
    if "adduct" not in ledger.columns:
        return False
    return bool(ledger["adduct"].astype(str).str.contains(r"\^NO3", regex=True).any())


def verdicts(ledger: pd.DataFrame, *, cfg=None) -> dict:
    """{peak_id: Ladder} for every committed M0 row whose ion carries fluorine.
    Empty when the ledger has no m/z / height / ion columns (a bare formula
    table: nothing to read a ladder from, the refusals act as before). `cfg`
    (the file's PassConfig) supplies the windows, the 13C floor and response;
    without it the 13C veto is off (no floor to test a line against)."""
    need = {"mz", "height", "peak_id", "role"}
    if ledger is None or not len(ledger) or not need.issubset(ledger.columns):
        return {}
    m0 = ledger.index[ledger["role"] == L.ROLE_M0]
    if not len(m0):
        return {}
    ions, neutrals = {}, {}
    col = lambda c: ledger.loc[m0, c].tolist() if c in ledger.columns else [None] * len(m0)  # noqa: E731
    for i, ion, nf, ad in zip(m0, col("ion_formula"), col("neutral_formula"), col("adduct")):
        cnt = _ion(ion, nf, ad)
        if cnt:
            ions[i] = cnt
            neutrals[i] = C.parse_formula(nf) if isinstance(nf, str) and nf.strip() else None
    frows = [i for i, c in ions.items() if c.get("F", 0) >= 1]
    if not frows:
        return {}
    lines = _Lines(ledger)
    ppm = window_ppm(cfg)
    key_of = {i: _key(c) for i, c in ions.items()}
    f14 = (1.0 - ISO.label_purity()) / ISO.label_purity() if labelled_nitrate(ledger) else 0.0
    kids: dict = {}
    if {"parent_peak_id", "iso_label"}.issubset(ledger.columns):
        k13 = ledger[(ledger["role"] == L.ROLE_ISO) & (ledger["iso_label"].astype(str) == "13C")]
        for pid, h, a in zip(k13["parent_peak_id"], k13["height"],
                             k13["area"] if "area" in k13.columns else [None] * len(k13)):
            kids.setdefault(pid, (_num(h) or 0.0, _num(a)))
    floor = _height_floor(cfg)
    out: dict = {}
    for i in frows:
        mz0, h0 = _num(ledger.at[i, "mz"]), _num(ledger.at[i, "height"])
        if mz0 is None or h0 is None or h0 <= 0:
            continue
        ion = ions[i]
        steps = []
        for k in [s for n in range(1, LADDER_STEPS + 1) for s in (-n, n)]:
            nb = dict(ion)
            nb["C"] = nb.get("C", 0) + k * CF2["C"]
            nb["F"] = nb.get("F", 0) + k * CF2["F"]
            if nb["C"] < 1 or nb["F"] < 1:
                continue                       # every ladder member is itself fluorinated
            hit = lines.near(mz0 + k * D_CF2, ppm)
            if hit is None or not (1.0 / LADDER_HEIGHT_X <= hit[1] / h0 <= LADDER_HEIGHT_X):
                continue
            j = hit[0]
            steps.append((k, str(ledger.at[j, "role"]) == L.ROLE_M0 and key_of.get(j) == _key(nb)))
        steps.sort()
        veto = alias_open = None
        if len(steps) >= LADDER_MIN_NEIGHBOURS:
            # only a ladder that would lift the refusal is tested for what refutes it
            veto, alias_open = (_alias_twin(ion, h0, lines, mz0, ppm, f14, floor, neutrals.get(i))
                                if f14 > 0 else (None, None))
            veto = veto or _c13_refutes(ledger, i, ion, lines, cfg, kids)
        out[ledger.at[i, "peak_id"]] = Ladder(1 + len(steps), 1 + sum(c for _, c in steps), tuple(steps), veto,
                                              alias_open)
    return out


def lift_note(v: Ladder) -> str:
    """The commentary / reason a lifted reading carries."""
    if v.full:
        return (f"fluorine read off its {v.text()}: a fluorinated homologous series, so the "
                "F count is not refused; the normal tier rules apply")
    if v.committed >= LADDER_MIN_COMMITTED and v.alias_open:
        return (f"fluorine read off its {v.text()}: a fluorinated homologous series, not refused, "
                f"but Candidate while its 15N alias (C2F3 -> H2^NO4, 0.21 mDa) is untested: {v.alias_open}")
    return (f"fluorine read off its {v.text()}: a fluorinated homologous series, not refused, "
            f"but Candidate until {LADDER_MIN_COMMITTED} members of the ladder are committed"
            + (f" and its 15N alias (C2F3 -> H2^NO4) is tested ({v.alias_open})" if v.alias_open else ""))


def veto_note(v: Ladder) -> str:
    """The note a ladder that a line refutes adds to the refusal."""
    return f"its {v.text()} does not lift the refusal: {v.veto}"


# ---------------------------------------------------------------------------
# the series hold: a series step is the alias's step too
# ---------------------------------------------------------------------------
def _s_untestable(ledger: pd.DataFrame, i, ion: dict, neutral: dict | None, cfg) -> bool:
    """The row's sulfur count is untestable in this file: the tier engine's own
    34S verdict (satellites.twin_verdict at its floor and window) reads neither
    a line that supports it nor one that refutes it."""
    try:
        floor = cfg.height_cutoff if cfg is not None else None
    except RuntimeError:
        floor = None
    counts = {k: v for k, v in ion.items() if k != "^N"}
    v = SAT.twin_verdict(ledger, ledger.at[i, "peak_id"], counts, floor, element="S",
                         masked_by=SAT.reagent_masks("S", neutral or counts, counts), ppm=SAT.twin_ppm(cfg))
    return v["kind"] == "untestable"


def series_alias_holds(ledger: pd.DataFrame, *, cfg=None) -> dict:
    """{peak_id: reason} of the committed M0 rows of a 15N-nitrate ledger whose
    series credit (an `anchor_peak_id` or a `series_unit`: a pass-6 ladder fill,
    a pass-4 series step) does not count. The row's ion carries a LABEL_ALIASES
    composition whose element count nothing else tests -- fluorine (no minor
    isotope), or sulfur whose 34S line the file can neither show nor refute
    (`_s_untestable`) -- and its alias can exist (alias_admissible) and is
    untested, or its 14N twin is seen (`_alias_twin`, the ladder's own test).
    The swap keeps every CH2 / O / H2O / CF2 step, so a series reads the alias
    exactly as it reads the row: the step corroborates neither. Each reason
    starts with SERIES_ALIAS_MARK (the tier engine appends it to the row's
    reason; the evidence scale's `anchor` axis reads it). Empty off a
    15N-nitrate ledger, without the columns, or with ALIAS_SERIES_HOLDS off.
    `cfg` (the file's PassConfig) supplies the floors and windows; without it
    nothing is testable and every alias stays open."""
    need = {"mz", "height", "peak_id", "role"}
    if not ALIAS_SERIES_HOLDS or ledger is None or not len(ledger) or not need.issubset(ledger.columns):
        return {}
    credit = pd.Series(False, index=ledger.index)
    for c in ("anchor_peak_id", "series_unit"):
        if c in ledger.columns:
            credit |= ledger[c].notna()
    rows = ledger.index[(ledger["role"] == L.ROLE_M0) & credit]
    if not len(rows) or not labelled_nitrate(ledger):
        return {}
    col = lambda i, c: ledger.at[i, c] if c in ledger.columns else None  # noqa: E731
    f14 = (1.0 - ISO.label_purity()) / ISO.label_purity()
    floor, ppm = _height_floor(cfg), window_ppm(cfg)
    lines, out = None, {}
    for i in rows:
        mz0, h0 = _num(ledger.at[i, "mz"]), _num(ledger.at[i, "height"])
        if mz0 is None or h0 is None or h0 <= 0:
            continue
        nf = col(i, "neutral_formula")
        ion = _ion(col(i, "ion_formula"), nf, col(i, "adduct"))
        neutral = C.parse_formula(nf) if isinstance(nf, str) and nf.strip() else None
        for alias in LABEL_ALIASES:
            if not _carries(ion, alias) or not alias_admissible(neutral, alias):
                continue
            if alias.element == "S" and not _s_untestable(ledger, i, ion, neutral, cfg):
                continue
            if lines is None:
                lines = _Lines(ledger)
            veto, open_ = _alias_twin(ion, h0, lines, mz0, ppm, f14, floor, neutral, alias)
            if veto or open_:
                out[ledger.at[i, "peak_id"]] = (
                    f"{SERIES_ALIAS_MARK} for the {alias.element} count: its 15N alias ({alias.name}, "
                    f"{alias.mda:.2f} mDa) takes the same series steps and "
                    + (f"is untested -- {open_}" if open_ else f"is seen -- {veto}"))
                break
    return out


# ---------------------------------------------------------------------------
# the 13C carbon-count term between sub-mDa aliases (pass 3, fluorinated family)
# ---------------------------------------------------------------------------
def c13_lines(ledger: pd.DataFrame, cfg=None) -> dict:
    """{peak_id: (h0, h_sat, a0, a_sat, snr0)} for every unexplained line that
    has a picked line at +1.00336 (its 13C line, if it is one): what
    `c13_alias_penalties` reads a carbon count from. The line is read when it
    is unexplained or an earlier pass's M0 -- an F line's own 13C line is often
    committed as a CHON(F) fit before the family reaches its parent: replayed on
    the stored pass-3 rows of the 15N-nitrate campaign, 254 of the term's lines
    were a pass-1/2 M0 (median reading C10.6), and reading them took the term
    from 192 to 262 per-file flips onto the truth reading, none off it. A
    foreign line that reads no alias is caught by the term's own test (the
    nearest alias must not be contradicted). A line another row claims as ITS
    isotope is not read. (fluorine's ladder veto, a refusal, reads only an
    unexplained line or the row's own child.) Empty without the columns or a
    detection floor (nothing is testable then)."""
    need = {"mz", "height", "peak_id", "role"}
    if ledger is None or not len(ledger) or not need.issubset(ledger.columns) or _height_floor(cfg) is None:
        return {}
    lines = _Lines(ledger)
    ppm = _c13_ppm(cfg)
    out = {}
    for i in ledger.index[ledger["role"] == L.ROLE_UNEXPLAINED]:
        mz0, h0 = _num(ledger.at[i, "mz"]), _num(ledger.at[i, "height"])
        if mz0 is None or h0 is None or h0 <= 0:
            continue
        hit = lines.near(mz0 + IR.D13C, ppm)
        if hit is None or str(ledger.at[hit[0], "role"]) not in (L.ROLE_UNEXPLAINED, L.ROLE_M0):
            continue
        out[ledger.at[i, "peak_id"]] = (
            h0, hit[1],
            _num(ledger.at[i, "area"]) if "area" in ledger.columns else None,
            _num(ledger.at[hit[0], "area"]) if "area" in ledger.columns else None,
            _num(ledger.at[i, "signal_to_noise"]) if "signal_to_noise" in ledger.columns else None)
    return out


def c13_reads(line, resp=None, area_mode: bool = False) -> list[float]:
    """The carbon counts a 13C line (`line`: a c13_lines entry) reads: the raw
    height reading (over R13C_RAW, as iso_response.contradicts reads it) and,
    where the run reads its lines by area or has a response, the corrected one
    (iso_response.carbon_from_13c) -- the last is the count reported."""
    h0, h_sat, a0, a_sat, snr0 = line
    out = [(h_sat / h0) / IR.R13C_RAW]
    if resp or area_mode:
        out.append(IR.carbon_from_13c(resp, h0=h0, h_sat=h_sat, a0=a0, a_sat=a_sat, snr0=snr0))
    return out


def c13_sigma(c: float, line) -> float:
    """The error of a carbon count c read off `line`: C13_REL of it, with the
    13C line's own S/N (snr0 x h_sat / h0) added in quadrature where known."""
    h0, h_sat, _a0, _a_sat, snr0 = line
    rel2 = C13_REL ** 2
    if snr0 and snr0 > 0 and h_sat > 0 and h0 > 0:
        rel2 += (h0 / (snr0 * h_sat)) ** 2
    return abs(c) * math.sqrt(rel2)


def c13_alias_penalties(grp: pd.DataFrame, line, cfg) -> tuple[dict, str | None]:
    """The 13C term for ONE peak's scored readings (`grp`: arbitrate's base rows
    of the peak, best eff_score first; `line` its c13_lines entry). The aliases
    are the readings whose ion mass lies within ALIAS_MDA of the top reading's.
    Where the 13C line is testable -- predicted above the detection floor for
    the alias of fewest carbons -- and reads a count c that the NEAREST alias's
    carbon count does not contradict, each alias that sits at least
    max(C13_MIN_GAP, C13_K x sigma_c) carbons farther from c than the nearest
    pays C13_ALIAS_PENALTY. A run that reads by area or has a response reads c
    both raw and corrected (c13_reads); an alias pays only if both readings
    charge it (as contradicts clears only on both). Comparative, so the
    carbon-rich alias gets no wider tolerance than the carbon-poor one.
    Returns ({row index: penalty}, a note for the winner) -- ({}, None) when
    the term does not apply."""
    floor = _height_floor(cfg)
    if line is None or floor is None or len(grp) < 2:
        return {}, None
    h0, h_sat, a0, a_sat, snr0 = line
    if not (h0 and h0 > 0):
        return {}, None
    top = grp.iloc[0]
    mz = _num(top.get("sample_peak_mz")) or _num(top.get("theo_mz"))
    p_top = _num(top.get("ppm_error"))
    if mz is None or p_top is None:
        return {}, None
    alias = {}
    for idx, r in grp.iterrows():
        p = _num(r.get("ppm_error"))
        if p is None or abs(p - p_top) * mz * 1e-3 > ALIAS_MDA:
            continue
        n_c = C.parse_formula(str(r.get("ion_formula"))).get("C", 0)
        if n_c >= 1:
            alias[idx] = n_c
    if len(alias) < 2 or len(set(alias.values())) < 2:
        return {}, None
    resp = getattr(cfg, "iso_response_fit", None)
    area_mode = bool(getattr(cfg, "iso_area", False))
    c_min = min(alias.values())
    x = math.log10(snr0 * IR.R13C * c_min) if (snr0 and snr0 > 0) else None
    if h0 * IR.R13C * c_min * IR.scale(resp, x) < floor:
        return {}, None                        # untestable: the line could be missing for any of them
    reads = c13_reads(line, resp, area_mode)
    c_est = reads[-1]
    n_near = min(alias.values(), key=lambda n: abs(c_est - n))
    if IR.contradicts(resp, n_near, h0=h0, h_sat=h_sat, a0=a0, a_sat=a_sat, snr0=snr0, area_mode=area_mode)[0]:
        return {}, None                        # the line reads none of them
    bad = None
    for c in reads:
        near = min(abs(c - n) for n in alias.values())
        gap = max(C13_MIN_GAP, C13_K * c13_sigma(c, line))
        hit = {idx for idx, n in alias.items() if abs(c - n) - near >= gap}
        bad = hit if bad is None else bad & hit
    if not bad:
        return {}, None
    names = sorted({str(grp.at[idx, "compound_formula"]) for idx in bad})
    note = (f"; 13C reads ~C{c_est:.0f}: the sub-mDa alias(es) {', '.join(names)} sit "
            f"clearly farther from that carbon count (-{C13_ALIAS_PENALTY:.2f} effective score)")
    return {idx: C13_ALIAS_PENALTY for idx in bad}, note
