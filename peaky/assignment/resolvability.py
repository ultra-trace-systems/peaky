"""Stage `resolvability`: how separable each committed M0 peak is from its
nearest picked neighbour, from the run's width model (chem.resolution).

Measured for every run that has a model -- a batch measures one from the raw
profile of a middling spectrum (`batch.tracefirst.measure_resolution`) and hands
it to every per-file run; a single-sample run measures its own when a server is
there; `--resolving-power R` declares one; an offline run with no model skips
the stage and the columns stay NA (the pair fact `res_ok` then reads "not measured", never
"blended": docs/EVIDENCE_LEVELS.md section 3.2). Until this stage existed the
flag was trace-first only, so the cover path -- every baseline run -- counted
its peaks as separable and `Assigned` without knowing whether the picked
centroid was the ion's own.

What it stamps, on M0 rows only (NA elsewhere): `resolvability` in `isolated`
/ `resolved` / `blended` / `unresolvable`, `sep_hwhm` (distance to the nearest
picked peak, in HWHM at that mass) and `d_crit_hwhm` (the separation at which
two peaks of that height ratio become bimodal). The neighbour pool is every
picked peak of the sample whatever its role -- a reagent ion or an unexplained
peak displaces a centroid as surely as an assigned one -- minus the synthetic
sub-peaks of the composite de-blending, which are not picked peaks.

Read by `tiers` (a blended / unresolvable M0 with no isotope, second-channel or
series corroboration is capped at Candidate: the mass the formula was fitted to
is not the ion's own) and by `evidence`'s pair facts (`res_ok`: the merge vote's
private class wants a separable peak, docs/EVIDENCE_LEVELS.md section 13).
Never a filter: no row is removed or re-assigned here.

Measured before the stage was written (two same-air batches, per-file Assigned
M0 rows, nearest picked neighbour within 8 HWHM): on a ~10k TOF 55 % of the
Assigned rows are blended (1 FWHM ~ 30 mDa at m/z 300) and 11 % blended with
no corroboration; on an Orbitrap (R ~120k at m/z 200) 13 % and 1 %.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.chem import resolution as RES

COLUMNS = ("resolvability", "sep_hwhm", "d_crit_hwhm")


def stamp_resolvability(ledger: pd.DataFrame, resolving_power, *, log=print) -> dict:
    """Stamp the three columns on the ledger's M0 rows in place; returns the
    model and the class counts. `resolving_power` is a number (constant R) or a
    `Resolution`; None is a caller error (the stage is skipped upstream)."""
    res = RES.Resolution.coerce(resolving_power)
    for c in COLUMNS:
        if c not in ledger.columns:
            ledger[c] = pd.NA if c == "resolvability" else np.nan
    if "role" not in ledger.columns or not len(ledger):
        return {"model": res.as_dict(), "counts": {}, "n_pool": 0}
    synthetic = (ledger["synthetic"].fillna(False).astype(bool)
                 if "synthetic" in ledger.columns else pd.Series(False, index=ledger.index))
    mz = pd.to_numeric(ledger["mz"], errors="coerce")
    pool = ledger.index[mz.notna() & ~synthetic]
    if not len(pool):
        return {"model": res.as_dict(), "counts": {}, "n_pool": 0}
    heights = (pd.to_numeric(ledger.loc[pool, "height"], errors="coerce")
               if "height" in ledger.columns else pd.Series(0.0, index=pool))
    out = RES.nearest_neighbour_classes(mz.loc[pool].to_numpy(), heights.to_numpy(), res)
    cls = pd.Series(out["resolvability"], index=pool, dtype=object)
    sep = pd.Series(out["sep_hwhm"], index=pool, dtype=float)
    dcr = pd.Series(out["d_crit_hwhm"], index=pool, dtype=float)
    m0 = ledger.index[(ledger["role"] == L.ROLE_M0)].intersection(pool)
    ledger.loc[m0, "resolvability"] = cls.loc[m0].to_numpy()
    ledger.loc[m0, "sep_hwhm"] = sep.loc[m0].to_numpy()
    ledger.loc[m0, "d_crit_hwhm"] = dcr.loc[m0].to_numpy()
    counts = {k: int(v) for k, v in cls.loc[m0].value_counts().to_dict().items()}
    log(f"[resolvability] {res.describe()}: M0 rows {counts} "
        f"(nearest picked neighbour within {RES.NEIGHBOUR_WINDOW_HWHM:g} HWHM, {len(pool)} peaks in the pool)")
    return {"model": res.as_dict(), "counts": counts, "n_pool": int(len(pool))}


def already_stamped(ledger: pd.DataFrame) -> bool:
    """True when the ledger carries a resolvability value already -- the
    trace-first synthetic sample brings its own from the trace build, and a
    second stamp from the same model would only redo it."""
    return "resolvability" in ledger.columns and bool(ledger["resolvability"].notna().any())
