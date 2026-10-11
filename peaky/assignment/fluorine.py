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
* lift: >= LADDER_MIN_NEIGHBOURS such lines -- the reading is not refused. It
  stays Candidate unless >= LADDER_MIN_COMMITTED members of its ladder (itself
  included) are committed (`full`): then the normal tier rules apply.
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
      audit clears on).

The ladder's 13C counterpart in arbitration -- a 13C carbon-count term between
sub-mDa aliases of the fluorinated family (C8 vs H1F5, 0.16 mDa) -- is
`c13_lines` + `c13_alias_penalties`, called by pass 3 (directors.run_pass3) and
arbitrate (passes/core.py).

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
LADDER_MIN_NEIGHBOURS = 2                  # ladder lines (the row excluded) that lift the refusal
LADDER_MIN_COMMITTED = 3                   # committed members (the row included): the normal tier rules apply
LADDER_PPM = 2.0                           # the step window where no exact-offset one holds (max with 2 x cfg.ppm)
LADDER_HEIGHT_X = 30.0                     # a ladder line reads within 1/30..30x the row's height
#: the 15N alias: C2F3 out, H2 ^N O4 in (0.21 mDa)
ALIAS_OUT = {"C": 2, "F": 3}
D_15N = C.M["^N"] - C.M["N"]               # 0.99703 Da
ALIAS_TWIN_BAND = (0.5, 5.0)               # the twin reads within this x the alias's nominal share f
C13_PPM = 5.0                              # the 13C line's window where no exact-offset one holds (audit_sat_ppm)
#: the 13C carbon-count term between sub-mDa aliases of the fluorinated family
ALIAS_MDA = 1.0                            # aliases: ion masses within this of the arbitration's top reading
C13_ALIAS_PENALTY = 0.30                   # eff-score charge on a reading whose carbon count its 13C line contradicts


class Ladder(NamedTuple):
    """One F reading's ladder verdict (`verdicts`)."""
    lines: int          # ladder lines found, the row included
    committed: int      # of them committed as the ladder's own ion composition, the row included
    steps: tuple        # ((k, committed), ...) of the lines found, k in CF2 steps
    veto: str | None    # why the ladder does not lift the refusal (None: it does, given enough lines)

    @property
    def lift(self) -> bool:
        """The ladder lifts the F refusal: enough lines, no refuting line."""
        return self.veto is None and self.lines - 1 >= LADDER_MIN_NEIGHBOURS

    @property
    def full(self) -> bool:
        """... and enough committed members for the normal tier rules to apply."""
        return self.lift and self.committed >= LADDER_MIN_COMMITTED

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


def _alias_twin(ion: dict, h0: float, lines: _Lines, mz0: float, ppm: float, f: float) -> str | None:
    """The 14N twin of the reading's C2F3 -> H2^NO4 alias, read as the alias's:
    a reason, else None (no alias, a reading that carries ^N -- its own twin
    cannot be told from its alias's --, no twin line, or one outside the band)."""
    if ion.get("C", 0) < ALIAS_OUT["C"] or ion.get("F", 0) < ALIAS_OUT["F"] or not (f > 0):
        return None
    if ion.get("^N", 0):
        return None
    hit = lines.near(mz0 - D_15N, ppm)
    if hit is None:
        return None
    r = hit[1] / h0
    lo, hi = ALIAS_TWIN_BAND
    if not (lo * f <= r <= hi * f):
        return None
    return (f"the 14N twin of its 15N alias (C2F3 -> H2^NO4, ^N1) reads {r:.3f} of the line "
            f"(the alias's share ~{f:.3f}) where the reading carries no ^N")


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
    whose ^N makes the C2F3 / H2^NO4 alias."""
    if "adduct" not in ledger.columns:
        return False
    return bool(ledger["adduct"].astype(str).str.contains(r"\^N", regex=True).any())


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
    ions = {}
    col = lambda c: ledger.loc[m0, c].tolist() if c in ledger.columns else [None] * len(m0)  # noqa: E731
    for i, ion, nf, ad in zip(m0, col("ion_formula"), col("neutral_formula"), col("adduct")):
        cnt = _ion(ion, nf, ad)
        if cnt:
            ions[i] = cnt
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
        veto = None
        if len(steps) >= LADDER_MIN_NEIGHBOURS:
            # only a ladder that would lift the refusal is tested for what refutes it
            veto = (_alias_twin(ion, h0, lines, mz0, ppm, f14) if f14 > 0 else None) \
                or _c13_refutes(ledger, i, ion, lines, cfg, kids)
        out[ledger.at[i, "peak_id"]] = Ladder(1 + len(steps), 1 + sum(c for _, c in steps), tuple(steps), veto)
    return out


def lift_note(v: Ladder) -> str:
    """The commentary / reason a lifted reading carries."""
    if v.full:
        return (f"fluorine read off its {v.text()}: a fluorinated homologous series, so the "
                "F count is not refused; the normal tier rules apply")
    return (f"fluorine read off its {v.text()}: a fluorinated homologous series, not refused, "
            f"but Candidate until {LADDER_MIN_COMMITTED} members of the ladder are committed")


def veto_note(v: Ladder) -> str:
    """The note a ladder that a line refutes adds to the refusal."""
    return f"its {v.text()} does not lift the refusal: {v.veto}"


# ---------------------------------------------------------------------------
# the 13C carbon-count term between sub-mDa aliases (pass 3, fluorinated family)
# ---------------------------------------------------------------------------
def c13_lines(ledger: pd.DataFrame, cfg=None) -> dict:
    """{peak_id: (h0, h_sat, a0, a_sat, snr0)} for every unexplained line that
    has a picked line at +1.00336 (its 13C line, if it is one): what
    `c13_alias_penalties` reads a carbon count from. Empty without the columns
    or a detection floor (nothing is testable then)."""
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
        if hit is None:
            continue
        out[ledger.at[i, "peak_id"]] = (
            h0, hit[1],
            _num(ledger.at[i, "area"]) if "area" in ledger.columns else None,
            _num(ledger.at[hit[0], "area"]) if "area" in ledger.columns else None,
            _num(ledger.at[i, "signal_to_noise"]) if "signal_to_noise" in ledger.columns else None)
    return out


def c13_alias_penalties(grp: pd.DataFrame, line, cfg) -> tuple[dict, str | None]:
    """The 13C term for ONE peak's scored readings (`grp`: arbitrate's base rows
    of the peak, best eff_score first; `line` its c13_lines entry). The aliases
    are the readings whose ion mass lies within ALIAS_MDA of the top reading's;
    where the 13C line is testable -- predicted above the detection floor for
    the alias of fewest carbons -- and contradicts the carbon count of some of
    them but not of all, each contradicted alias pays C13_ALIAS_PENALTY.
    Returns ({row index: penalty}, a note for the winner) -- ({}, None) when
    the term does not apply."""
    floor = _height_floor(cfg)
    if line is None or floor is None or len(grp) < 2:
        return {}, None
    h0, h_sat, a0, a_sat, snr0 = line
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
    c_min = min(alias.values())
    x = math.log10(snr0 * IR.R13C * c_min) if (snr0 and snr0 > 0) else None
    if h0 * IR.R13C * c_min * IR.scale(resp, x) < floor:
        return {}, None                        # untestable: the line could be missing for any of them
    verdict = {idx: IR.contradicts(resp, n_c, h0=h0, h_sat=h_sat, a0=a0, a_sat=a_sat, snr0=snr0,
                                   area_mode=bool(getattr(cfg, "iso_area", False)))
               for idx, n_c in alias.items()}
    bad = {idx for idx, (b, _c) in verdict.items() if b}
    if not bad or len(bad) == len(alias):
        return {}, None
    c_est = next(c for idx, (_b, c) in verdict.items() if idx not in bad)
    names = sorted({str(grp.at[idx, "compound_formula"]) for idx in bad})
    note = (f"; 13C reads ~C{c_est:.0f}: its carbon count contradicts the sub-mDa alias(es) "
            f"{', '.join(names)} (-{C13_ALIAS_PENALTY:.2f} effective score)")
    return {idx: C13_ALIAS_PENALTY for idx in bad}, note
