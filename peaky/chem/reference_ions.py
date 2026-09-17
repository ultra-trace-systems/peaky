"""Reference ions -- the external mass yardstick for `peaky mass-qc`.

A reference ion is a (neutral, channel) pair whose FORMULA is certain, so its
theoretical m/z can be used to measure a batch's mass axis instead of the
engine's own output. Pass-1 self-calibration fits the Assigned backbone, which
is circular: an axis 30 ppm wrong yields a self-consistent calibration and a
batch of confident, wrong formulas, and nothing in the run log says the axis
moved. Probing formula-certain ions breaks the circle.

Tables:

  NO3       the nitrate-mode core: 30 ions present in every nitrate-CIMS
            dataset probed while building the list (six instruments, three
            sites, TOF and Orbitrap), shipped as data/reference_ions/nitrate.csv
            with a grade (A / A- / B-TOF ...), an anchor flag (the 11 purest,
            the ones a calibration should rest on) and a blend flag.
  NO3_15N   the same chemistry with the cluster channel on the heavy reagent.
  Br        a PROVISIONAL bromide list: the reagent ladder plus the small acids
            a Br- source always sees. The nitrate list was earned by probing
            seven datasets; this one has not been, so mass-qc validates every
            entry against the batch it is given and reports which qualified.

Masses go through `chem.ion_mz`, so the 15N-reagent variant is a parameter and
not a second table. Every Br adduct comes with its 81Br twin (+1.9979521 Da at
~0.97 of its height): the two sit on the same axis, so their disagreement is a
free internal check that the position is one clean ion.

The CSV carries no site, campaign or instrument name: it is generic nitrate-CIMS
chemistry, and the repository is public.
"""
from __future__ import annotations

import pandas as pd

from peaky.chem import chemistry as C
from peaky.paths import pkg_data

__version__ = "0.1.0"
__all__ = ["nitrate", "bromide", "get", "available", "ion_mz", "COLUMNS"]

COLUMNS = ["reagent", "neutral", "channel", "identity", "grade", "anchor", "blend_flag",
           "mz", "iso", "twin_key"]

_M_81BR = 80.9162897                      # AME2020
_D_81BR = _M_81BR - C.M["Br"]             # +1.9979521 Da
# heavy-bromine channels: peaky's ADDUCT_SHIFTS carries the 79Br adduct only
EXTRA_SHIFTS = {
    "[M+81Br]-": _M_81BR + C.M_E,
    "[M+81Br-H]-": _M_81BR - C.M["H"] + C.M_E,
}


def ion_mz(neutral: str, channel: str) -> float:
    """m/z of `neutral` on `channel`, including the heavy-bromine channels."""
    if channel in EXTRA_SHIFTS:
        return C.neutral_mass(neutral) + EXTRA_SHIFTS[channel]
    return C.ion_mz(neutral, channel)


def _finish(t: pd.DataFrame, reagent: str) -> pd.DataFrame:
    t = t.copy()
    t["reagent"] = reagent
    for c in ("anchor", "blend_flag"):
        t[c] = t[c].map(lambda v: str(v).strip().lower() in ("true", "1", "yes")) \
            if t[c].dtype == object else t[c].astype(bool)
    t["iso"] = t.get("iso", pd.Series("", index=t.index)).fillna("")
    if "twin_key" not in t.columns:
        t["twin_key"] = t["neutral"] + "|" + t["channel"]
    return (t[COLUMNS].sort_values(["mz", "channel"], kind="mergesort")
            .reset_index(drop=True))


def nitrate(label_15n: bool = False) -> pd.DataFrame:
    """The nitrate-mode core. `label_15n` puts the cluster channel on the heavy
    reagent ([M+^NO3]-, +0.99703 Da); [M-H]- is reagent-independent."""
    t = pd.read_csv(pkg_data("reference_ions", "nitrate.csv"))
    if label_15n:
        t["channel"] = t["channel"].replace({"[M+NO3]-": "[M+^NO3]-"})
    t["mz"] = [ion_mz(n, c) for n, c in zip(t["neutral"], t["channel"])]
    return _finish(t, "NO3_15N" if label_15n else "NO3")


# neutral, channel, identity, grade, anchor -- the reagent ladder first (Br- IS
# the [M-H]- ion of HBr, Br-.H2O of HBr.H2O and HBr.Br- of (HBr)2, so the ladder
# needs no special case, only honesty about which neutral each ion is the
# conjugate base of), then the small acids. `anchor` is a PRIOR here, confirmed
# or withdrawn by mass-qc on the batch.
_BR = [
    ("HBr",     "[M-H]-",  "Br- reagent ion (79Br)",       "A", True),
    ("H3BrO",   "[M-H]-",  "Br-.H2O (79Br)",               "A", True),
    ("H2Br2",   "[M-H]-",  "HBr.Br- (79Br2)",              "A", True),
    ("HNO3",    "[M+Br]-", "nitric acid . Br-",            "A", True),
    ("HNO2",    "[M+Br]-", "nitrous acid . Br-",           "B", False),
    ("CH2O2",   "[M+Br]-", "formic acid . Br-",            "A", True),
    ("C2H4O2",  "[M+Br]-", "acetic acid . Br-",            "A", True),
    ("C2H2O4",  "[M+Br]-", "oxalic acid . Br-",            "A", True),
    ("C3H4O4",  "[M+Br]-", "malonic acid . Br-",           "B", False),
    ("C4H6O4",  "[M+Br]-", "succinic acid . Br-",          "B", False),
    ("C2HF3O2", "[M-H]-",  "trifluoroacetic acid [M-H]-",  "A", True),
    ("C2HF3O2", "[M+Br]-", "trifluoroacetic acid . Br-",   "A", True),
    ("HNO3",    "[M-H]-",  "nitrate NO3-",                 "B", False),
]


def bromide() -> pd.DataFrame:
    """Provisional Br- reference ions, each Br adduct with its 81Br twin (same
    `twin_key`, iso '81Br'), plus the heavy twins of the reagent ladder."""
    rows = []
    for neutral, ch, ident, grade, anch in _BR:
        key = f"{neutral}|{ch}"
        rows.append(dict(neutral=neutral, channel=ch, identity=ident, grade=grade,
                         anchor=anch, blend_flag=False, mz=ion_mz(neutral, ch),
                         iso="", twin_key=key))
        if ch == "[M+Br]-":
            rows.append(dict(neutral=neutral, channel="[M+81Br]-",
                             identity=ident.replace("Br-", "81Br-"), grade=grade,
                             anchor=False, blend_flag=False,
                             mz=ion_mz(neutral, "[M+81Br]-"), iso="81Br", twin_key=key))
    for ident, base, n81 in (("81Br- reagent ion", "HBr", 1),
                             ("81Br-.H2O", "H3BrO", 1),
                             ("HBr.81Br- / H81Br.Br-", "H2Br2", 1),
                             ("H81Br.81Br-", "H2Br2", 2)):
        rows.append(dict(neutral=base, channel=f"[M-H]-(+{n81}x81Br)", identity=ident,
                         grade="A", anchor=False, blend_flag=False,
                         mz=C.ion_mz(base, "[M-H]-") + n81 * _D_81BR, iso="81Br",
                         twin_key=f"{base}|[M-H]-" if n81 == 1 else f"{base}|[M-H]-(2x81Br)"))
    return _finish(pd.DataFrame(rows), "Br")


_ALIASES = {
    "no3": "NO3", "nitrate": "NO3", "no3-": "NO3", "nitrate-cims": "NO3",
    "no3_15n": "NO3_15N", "15no3": "NO3_15N", "15no3-": "NO3_15N", "^no3": "NO3_15N",
    "no3-15n": "NO3_15N", "[15n]o3-": "NO3_15N",
    "br": "Br", "bromide": "Br", "br-": "Br", "br-cims": "Br",
}


def available() -> list[str]:
    return ["NO3", "NO3_15N", "Br"]


def get(reagent: str) -> pd.DataFrame:
    """The reference table for a reagent name or alias. KeyError otherwise."""
    key = _ALIASES.get((reagent or "").strip().lower())
    if key == "NO3":
        return nitrate()
    if key == "NO3_15N":
        return nitrate(label_15n=True)
    if key == "Br":
        return bromide()
    raise KeyError(f"no reference-ion table for {reagent!r}; available: {available()}")
