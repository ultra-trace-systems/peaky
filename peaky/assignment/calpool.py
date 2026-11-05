"""The tier calibration pooled over a batch, for the files that cannot calibrate alone.

The tier engine calibrates each file on its own core (`tiers._calibrate`): at
least `tiers.CAL_MIN_N` (20) High/Good, pure-organic, isotope-backed M0 rows. A
small scan window never gets there -- a 15N-nitrate 40-600 or 65-120 window, an
NO+ 40-160 window: its carbon-bearing lines put their 13C under the noise edge,
and most of what IS isotope-backed are carbon-free reagent lines (HNO3 clusters
seen with their own 15N / 18O lines). Such a file runs uncalibrated: the
degeneracy audit and the mass z-tail gate are skipped, its Assigned rows rest on
"degeneracy not measured", and the evidence scale has no window -- a batch none
of whose files calibrated levels nothing.

The files of one batch can reach together what none reaches alone. The pooled
core (`pool_calibration`):

  * the rows `tiers.core_point` takes from each file, restricted to a
    carbon-bearing neutral (C >= 1), to a reading that is not a reagent identity
    (a neutral made of the reagent's own molecules on a reagent adduct,
    `levels.decide.reagent_identity`) and to a line off every reagent-library
    ion mass;
  * one row per distinct ION (its ion formula), the tallest of the files' rows
    for it: a pool of the per-file cores as they stand repeats the same few
    bright ions in every file, and on a 40-600 nitrate zone collapsed onto the
    HNO3 lines alone (23 rows, sigma on the floor);
  * an ion counts only where it is isotope-backed in at least POOL_MIN_FILES
    files (one file's isotope line is that file's luck).

It calibrates when it holds at least POOL_MIN_IONS distinct ions spanning at
least POOL_MIN_SPAN_FRAC of the m/z range the batch's files committed; (mu,
sigma) as the file's own (`tiers.robust_centre`: the median, the scaled MAD
over the errors within 2 ppm of it, the 0.15-ppm floor), the constant model
only -- its points come from different files, so no mass trend is fitted on
them. Otherwise the batch stays uncalibrated, as before, and its record says
why.

What a file is lent (`lent_calibration`) is a WIDTH about the file's OWN
centre: the median ppm error of its Assigned M0 rows (the centre its tier stage
already stamped as `ppm_error_cal`'s, `tiers.stamp_calibrated_ppm`), when it has
at least CENTRE_MIN_ROWS of them, else the batch's centre. The files of one
batch sit apart by up to a few tenths of a ppm (per-calibration offsets), as
much as the pooled sigma itself: centred on the pool, a file 1 sigma off would
put about 6 % of its correct uncorroborated rows past the z-tail instead of
1 %. The width is the RUN sigma when any file of the batch calibrated alone
(the median of those files' sigma, `Pool.run_window`; the batch's centre is
then their median mu): a file-centred window wants a single file's width, which
those files measure directly, while the pooled sigma also carries the spread
between the files its ions came from. The evidence levels already window an
uncalibrated file of a calibrated batch that way (its centre, the run sigma).
With no file calibrated alone the pooled sigma (and mu) is all there is.

A TOF-class batch is not pooled (`pool_calibration(klass='tof')`): a TOF file's
own calibration fits a mass trend and its tier stage reads a TOF floor; the
pooled core's constant model has not been tested there, so such a file stays
uncalibrated, as before, and the record says why.

`apply_pool` gives one finished, uncalibrated per-file ledger the pooled
calibration WITHOUT re-running the file: no enumeration, no scoring, no winner
changes. It re-applies the steps the calibration decides, in their pipeline
order, on the ledger as the file's own degeneracy stage found it (the run keeps
that copy for an uncalibrated file: `PoolInputs.at_tiers`):

  1. the degeneracy audit (`degeneracy.apply_degeneracy`) in the pooled
     window, over the space the file's own stage counted (its channels, the
     families it opened, the curated formulas); the stamps are carried onto the
     finished ledger by peak, as the stage's own would have stood there;
  2. the tier stage (`tiers.compute_tiers` with the pooled calibration: the
     z-tail gate, the degeneracy cap, the reasons that name the window); a
     finished row whose verdict is still the one the uncalibrated tier stage
     gave it takes the calibrated one (`tiers.retier_calibrated`; a row a later
     stage only demoted takes the calibrated words), a row a later stage
     rewrote keeps what that stage left;
  3. the two degeneracy-keyed flags on the finished ledger: below
     assignability (O >= 11 on a mass-saturated window,
     `tiers.flag_below_assignability`; a tier-stage flag, so only on the rows
     still committing their tier-time reading) and the oxygen-monster demote
     (`plausibility.demote_oxygen_monsters`);
  4. the file's own evidence level (`evidence.apply_levels`) in the pooled
     window.

The NOx-skeleton cap's record (stats `nox_skeleton_gate`) is re-taken where the
run takes it: on the tier-time ledger with the re-run tier stage's verdicts,
before any later stage rewrote a row.

Not re-applied (they would need the file re-run): the off-calibration
re-arbitration (`passes.rearbitrate_offcal_degenerate`, which picks the
winner), the pass engine's own scoring calibration (a Low / Suspect confidence
stays what the passes gave it), and the post-tier stages that read a row's tier
or flags (the ion-only bucket's parents, the cluster re-read's skeleton
re-judgement). A row a post-tier re-read rewrote (a nitrate-cluster, radical-
anion or reagent-N re-read) keeps its tier: the tier stage judged the reading
it replaced.

Pure on its inputs; the batch (`assign_batch.run`) decides the pool once, after
the first per-file stage, and applies it to every uncalibrated file before the
merge.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.assignment import masscal as MC
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C

__version__ = "0.1.1"  # + the per-file isotope-contradiction guard on the pooled re-level (isoguard.guard_ledger)
#                        0.1.0 the batch-pooled tier calibration (REPORT #1 / B7)

#: distinct ions the pooled core needs (the per-file core needs tiers.CAL_MIN_N rows)
POOL_MIN_IONS = 15
#: an ion enters the pool only where it is isotope-backed in at least this many files
POOL_MIN_FILES = 2
#: the core's m/z span, as a fraction of the batch's committed M0 m/z range
POOL_MIN_SPAN_FRAC = 0.25
#: a core line within this many ppm of a reagent-library ion is the reagent's
#: (the reagent sweep's own window, reagents.reclaim_reagent_clusters). Only a
#: reagent with a cluster library has masses to test (halide, urea, ammonium,
#: NO+: reagents.reagent_for_adducts); a nitrate set has none, and there C >= 1
#: and the reagent-identity test are the reagent exclusions
REAGENT_PPM = 12.0
#: a file lent the pooled calibration is centred on its own Assigned M0 rows when
#: it has at least this many (`lent_calibration`), else on the batch's centre
CENTRE_MIN_ROWS = 15

CORE_COLUMNS = ["sid", "ion", "neutral_formula", "adduct", "mz", "ppm", "height"]


def _ion_key(r) -> str:
    """The row's ion: its `ion_formula`, else the ion composition of the reading."""
    ion = T._txt(r.get("ion_formula")).strip()
    if ion:
        return ion
    cnt = T._ion_counts(r.get("neutral_formula"), r.get("adduct"), labelled=True)
    return "|".join(f"{k}{v}" for k, v in sorted(cnt.items())) if cnt else (
        f"{T._txt(r.get('neutral_formula'))} {T._txt(r.get('adduct'))}")


def core_rows(ledger: pd.DataFrame, *, polarity: str = "negative", reagent_masses=(), sid: str = "") -> pd.DataFrame:
    """The rows of one file's ledger the pooled core may take (CORE_COLUMNS):
    `tiers.core_point`'s rows whose neutral carries carbon, whose reading is no
    reagent identity on the run's `polarity` ('negative' / 'positive') and whose
    line sits off every `reagent_masses` ion (REAGENT_PPM)."""
    from peaky.assignment.levels.decide import reagent_identity
    if ledger is None or not len(ledger) or "role" not in ledger.columns:
        return pd.DataFrame(columns=CORE_COLUMNS)
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    kids_of = ledger.loc[ledger["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
    masses = np.sort(np.asarray([float(m) for m in reagent_masses], dtype=float))
    recs = []
    for _, r in m0.iterrows():
        pt = T.core_point(r, kids_of)
        if pt is None or not np.isfinite(pt[1]):
            continue
        nf, ad = T._txt(r.get("neutral_formula")), T._txt(r.get("adduct"))
        if C.parse_formula(nf).get("C", 0) < 1:
            continue
        if reagent_identity(nf, ad, polarity):
            continue
        if len(masses) and np.any(np.abs(masses - pt[1]) <= masses * REAGENT_PPM * 1e-6):
            continue
        h = pd.to_numeric(r.get("height"), errors="coerce")
        recs.append({"sid": str(sid), "ion": _ion_key(r), "neutral_formula": nf, "adduct": ad,
                     "mz": pt[1], "ppm": pt[0], "height": float(h) if pd.notna(h) else 0.0})
    return pd.DataFrame(recs, columns=CORE_COLUMNS)


@dataclasses.dataclass
class Pool:
    """A batch's pooled tier calibration: `cal` (a tiers._Cal, source
    CAL_SOURCE_POOLED) or None with the `reason` it is not used, and what it
    was built from."""
    cal: object = None
    reason: str | None = None
    n_files_read: int = 0      # per-file ledgers the pool read
    n_files: int = 0           # files holding a pooled ion's row
    n_ions: int = 0            # distinct ions in the core
    n_candidate_ions: int = 0  # distinct carbon-bearing core ions before the file-count minimum
    mz_span: tuple = (float("nan"), float("nan"))     # the core's m/z span
    mz_range: tuple = (float("nan"), float("nan"))    # the batch's committed M0 m/z range
    core: pd.DataFrame = dataclasses.field(default_factory=lambda: pd.DataFrame(columns=CORE_COLUMNS))
    #: (mu, sigma, n) -- the medians of the per-file calibrations of the files that
    #: calibrated alone (n of them), set by the batch; None when none did. A
    #: file is then lent the run sigma, not the pooled one (`lent_calibration`)
    run_window: tuple | None = None

    def lent_width(self) -> tuple[float, float, str] | None:
        """(centre, sigma, 'run' | 'pool') a file is lent about its own centre
        (the centre: a file with too few Assigned rows of its own); None
        without a calibration."""
        if self.cal is None:
            return None
        if self.run_window is not None:
            return float(self.run_window[0]), float(self.run_window[1]), "run"
        return float(self.cal[0]), float(self.cal[1]), "pool"

    def record(self) -> dict:
        """The batch_summary record (JSON-safe)."""
        def _f(x, nd=3):
            return None if x is None or not np.isfinite(x) else round(float(x), nd)
        lw = self.lent_width()
        return {"used": self.cal is not None,
                "mu": _f(self.cal[0], 4) if self.cal is not None else None,
                "sigma": _f(self.cal[1], 4) if self.cal is not None else None,
                # the width a file is lent (the run sigma when any file calibrated
                # alone, else the pooled one) and where it came from
                "lent_sigma": _f(lw[1], 4) if lw is not None else None,
                "lent_width": lw[2] if lw is not None else None,
                "n_files_calibrated": int(self.run_window[2]) if self.run_window is not None else 0,
                "n_ions": int(self.n_ions), "n_candidate_ions": int(self.n_candidate_ions),
                "n_files": int(self.n_files), "n_files_read": int(self.n_files_read),
                "mz_span": [_f(self.mz_span[0], 4), _f(self.mz_span[1], 4)],
                "mz_range": [_f(self.mz_range[0], 4), _f(self.mz_range[1], 4)],
                "min_ions": POOL_MIN_IONS, "min_files": POOL_MIN_FILES, "min_span_frac": POOL_MIN_SPAN_FRAC,
                "reason": self.reason}

    def describe(self) -> str:
        """One log line."""
        if self.cal is None:
            return f"no pooled tier calibration: {self.reason}"
        lw = self.lent_width()
        lent = (f"; a file is lent the run sigma {lw[1]:.3f} ppm of the {int(self.run_window[2])} file(s) "
                f"that calibrated alone" if lw[2] == "run" else "; a file is lent the pooled sigma")
        return (f"pooled tier calibration {self.cal[0]:+.3f} ± {self.cal[1]:.3f} ppm from {self.n_ions} distinct "
                f"ions isotope-backed in >= {POOL_MIN_FILES} files ({self.n_files} of {self.n_files_read} files; "
                f"m/z {self.mz_span[0]:.1f}-{self.mz_span[1]:.1f} of the committed {self.mz_range[0]:.1f}-"
                f"{self.mz_range[1]:.1f}){lent}")


def pool_calibration(ledgers: dict, *, polarity: str = "negative", reagent_masses=(),
                     abs_floor_mda: float = MC.ABS_FLOOR_MDA, min_ions: int = POOL_MIN_IONS,
                     min_files: int = POOL_MIN_FILES, min_span_frac: float = POOL_MIN_SPAN_FRAC,
                     klass: str | None = None, carry=None) -> Pool:
    """The pooled core of a batch's per-file ledgers ({sid: ledger}, read in
    sorted sid order) and its calibration (module note). `klass`: the batch's
    instrument class (PassConfig.instrument_class); a 'tof' batch gets the core
    counted and no calibration (module note). `carry`: the batch axis model's edge
    carry (`mass_axis.edge_carry`; masscal.edge_carry_at): a core row past the
    model's edge is read on the carried scale -- its ppm less the model's edge
    value, as the file's own tier calibration reads it -- and `core` keeps it so.
    Never raises on thin data: a pool that cannot calibrate returns `cal` None and
    its reason."""
    sids = sorted(ledgers, key=str)
    parts = [core_rows(ledgers[s], polarity=polarity, reagent_masses=reagent_masses, sid=s) for s in sids]
    parts = [p for p in parts if len(p)]
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CORE_COLUMNS)
    if carry and len(rows):
        rows["ppm"] = rows["ppm"].astype(float) - MC.edge_carry_at(carry, rows["mz"].to_numpy(dtype=float))
    mzs = []
    for s in sids:
        led = ledgers[s]
        if led is not None and len(led) and {"role", "mz"} <= set(led.columns):
            mzs.append(pd.to_numeric(led.loc[led["role"] == L.ROLE_M0, "mz"], errors="coerce").to_numpy(dtype=float))
    allm = np.concatenate(mzs) if mzs else np.empty(0)
    allm = allm[np.isfinite(allm)]
    pool = Pool(n_files_read=len(sids),
                mz_range=(float(allm.min()), float(allm.max())) if len(allm) else (float("nan"), float("nan")))
    if not len(rows):
        pool.reason = ("no file holds a carbon-bearing, isotope-backed High/Good core row off the reagent "
                       "(too few 13C lines above the noise edge)")
        return pool
    nfiles = rows.groupby("ion")["sid"].nunique()
    pool.n_candidate_ions = int(len(nfiles))
    keep = set(nfiles[nfiles >= min_files].index)
    best = (rows[rows["ion"].isin(keep)]
            .sort_values(["height", "sid"], ascending=[False, True], kind="mergesort")
            .drop_duplicates("ion"))
    pool.core = best.reset_index(drop=True)
    pool.n_ions = int(len(best))
    pool.n_files = int(rows.loc[rows["ion"].isin(keep), "sid"].nunique())
    if len(best):
        pool.mz_span = (float(best["mz"].min()), float(best["mz"].max()))
    if klass == "tof":
        pool.reason = ("a TOF-class batch: the pooled core's constant model is not lent (a TOF file's own "
                       "calibration fits a mass trend; not tested on TOF)")
        return pool
    if len(best) < min_ions:
        pool.reason = (f"{len(best)} distinct carbon-bearing ions isotope-backed in >= {min_files} files "
                       f"({pool.n_candidate_ions} in any file); the pooled core needs {min_ions}")
        return pool
    rng = pool.mz_range[1] - pool.mz_range[0]
    span = pool.mz_span[1] - pool.mz_span[0]
    if not (np.isfinite(rng) and rng > 0 and span >= min_span_frac * rng):
        pool.reason = (f"the core's {len(best)} ions span m/z {pool.mz_span[0]:.1f}-{pool.mz_span[1]:.1f}, under "
                       f"{min_span_frac:g} of the committed range {pool.mz_range[0]:.1f}-{pool.mz_range[1]:.1f}")
        return pool
    mu, sigma, _keep = T.robust_centre(best["ppm"].to_numpy(dtype=float), min_ions)
    pool.cal = T._Cal(mu, sigma, abs_floor_mda=abs_floor_mda, source=T.CAL_SOURCE_POOLED)
    return pool


@dataclasses.dataclass
class PoolInputs:
    """What a file's own run used for the steps `apply_pool` re-applies
    (assign.run returns it, as `pool_inputs`, for an uncalibrated file): the
    ledger as its degeneracy stage found it (`at_tiers`), its final cfg
    (PassConfig), its context profile, its channels, the contaminant families
    its degeneracy stage opened, the curated formulas, and the run inputs its
    evidence stage levelled it with (None: the stage did not run)."""
    at_tiers: object = None
    cfg: object = None
    profile: object = None
    adducts: list = dataclasses.field(default_factory=list)
    families: tuple = ()
    curated: frozenset = frozenset()
    level_inputs: object = None


def lent_calibration(pool: Pool, assigned_ppm=(), *, min_rows: int = CENTRE_MIN_ROWS):
    """(cal, centre) the pool lends one file: a tiers._Cal (source
    CAL_SOURCE_POOLED) of the lent width (`Pool.lent_width`: the run sigma when
    any file of the batch calibrated alone, else the pooled sigma) about the
    file's own centre -- the median of `assigned_ppm`, the ppm errors of the M0
    rows its own tier stage made Assigned (the centre that stage stamped as
    `ppm_error_cal`'s), `centre` 'file' -- when there are at least `min_rows`
    of them, else about the batch's centre (the run mu, else the pooled mu;
    `centre` 'run' / 'pool'). (None, None) when the pool has no calibration."""
    lw = pool.lent_width()
    if lw is None:
        return None, None
    p = pd.to_numeric(pd.Series(list(assigned_ppm), dtype=object), errors="coerce").dropna()
    mu, centre = (float(p.median()), "file") if len(p) >= min_rows else (lw[0], lw[2])
    return T._Cal(mu, lw[1], abs_floor_mda=getattr(pool.cal, "abs_floor_mda", MC.ABS_FLOOR_MDA),
                  source=T.CAL_SOURCE_POOLED), centre


def _level_inputs_with(ri, cal):
    """The file's evidence-stage run inputs with the pooled window as its degeneracy calibration."""
    summ = copy.deepcopy(ri.summary)
    pf = summ.get("per_file") or [{}]
    pf[0] = dict(pf[0], degeneracy_cal={"mu": float(cal[0]), "sigma": float(cal[1])},
                 tier_calibration=T.CAL_SOURCE_POOLED)
    summ["per_file"] = pf
    return dataclasses.replace(ri, summary=summ)


def apply_pool(ledger: pd.DataFrame, pool: Pool, inputs: PoolInputs, *, audit=None, log=print) -> dict:
    """Give one finished, uncalibrated per-file ledger the batch's pooled
    calibration, in place (module note, steps 1-4). `inputs.at_tiers` is the
    ledger as the file's degeneracy stage found it (read, never changed);
    `audit`: the file's plausibility audit list (the oxygen-monster demote
    appends to it). The calibration is the one the pool lends this file
    (`lent_calibration`: the lent width about the file's own centre). Returns
    {cal, degeneracy, tiers, nox_skeleton_gate, below, o_monster, levels}
    (+ ``isotope_guard`` when the file was levelled: the per-file
    isotope-contradiction guard, isoguard.guard_ledger, on the facts of THIS
    re-level -- the file the per-file stage could not level is guarded here, at
    the same point relative to the vote; any calibration lent through this
    function, pooled or a supplied one, takes it)
    (`cal`: {mu, sigma, centre, width}; `nox_skeleton_gate`: the NOx-skeleton
    cap's record as the re-run tier stage leaves it, tiers.nox_skeleton_summary
    on the tier-time ledger, None when the profile reads no skeletons); {} when
    the pool has no calibration or there is no tier-time ledger."""
    from peaky.assignment import degeneracy as DG
    from peaky.assignment import evidence as EV
    from peaky.assignment import plausibility as PL
    if pool.cal is None or ledger is None or not len(ledger) or inputs.at_tiers is None:
        return {}
    cfg, profile = inputs.cfg, inputs.profile
    at = inputs.at_tiers.copy(deep=True)
    # 1-2. the degeneracy and tier stages again on the ledger they saw: the
    # verdicts the uncalibrated stage gave, the stamps in the pooled window, the
    # verdicts the calibrated stage gives
    base = T.compute_tiers(at, cfg=cfg, profile=profile, cal=None)
    # the window: the pooled width about the centre of the rows that stage made Assigned
    won = set(base.loc[base["tier"] == T.TIER_ASSIGNED, "peak_id"]) if len(base) else set()
    won_rows = at.loc[(at["role"] == L.ROLE_M0) & at["peak_id"].isin(won)]
    won_ppm = pd.to_numeric(won_rows["ppm_error"], errors="coerce")
    if getattr(cfg, "axis_edge_carry", None):
        # the file's own centre on the carried scale: a row past the batch axis model's edge
        # less the model's edge value (the scale the lent window adds it back on)
        won_ppm = won_ppm - MC.edge_carry_at(
            cfg.axis_edge_carry, pd.to_numeric(won_rows["mz"], errors="coerce").to_numpy(dtype=float))
    cal, centre = lent_calibration(pool, won_ppm)
    if cal is not None and getattr(cfg, "axis_edge_carry", None):
        # the batch axis model's edge carry: the file's lines just outside the model's scope
        # are judged in the lent window at its centre plus the model's edge (degeneracy, tiers)
        cal = T._with_carry(cal, cfg.axis_edge_carry)
    DG.apply_degeneracy(at, cal=cal, context=profile if profile is not None else "ambient-air",
                        adducts=list(inputs.adducts or []) or None, families=tuple(inputs.families or ()),
                        curated=frozenset(inputs.curated or ()), log=(lambda *_: None))
    new = T.compute_tiers(at, cfg=cfg, profile=profile, cal=cal)
    # the stamps onto the finished ledger, by peak (a row re-read or cleared later
    # keeps the stamp its tier-time reading got, as the stage's own does)
    for col in ("degeneracy_density", "degeneracy_note"):
        if col not in ledger.columns:
            ledger[col] = pd.Series(pd.NA, index=ledger.index, dtype="object")
        elif ledger[col].dtype != object:
            ledger[col] = ledger[col].astype("object")
    am = at[at["role"] == L.ROLE_M0]
    dens = dict(zip(am["peak_id"], am["degeneracy_density"]))
    notes = dict(zip(am["peak_id"], am["degeneracy_note"]))
    for i in ledger.index[ledger["peak_id"].isin(dens)]:
        pid = ledger.at[i, "peak_id"]
        ledger.at[i, "degeneracy_density"] = dens[pid]
        ledger.at[i, "degeneracy_note"] = notes[pid]
    d = pd.to_numeric(am["degeneracy_density"], errors="coerce")
    out = {"cal": {"mu": float(cal[0]), "sigma": float(cal[1]), "centre": centre, "width": pool.lent_width()[2]},
           "degeneracy": {"measured": int(d.notna().sum()),
                          "degenerate": int(sum(T._degeneracy(r)[1] for _, r in am.iterrows()))}}
    out["tiers"] = T.retier_calibrated(ledger, base, new, source=T.CAL_SOURCE_POOLED)
    # the skeleton cap's record where the run takes it (assign._stage_tiers):
    # the tier stage's verdicts, before later stages rewrite rows
    rec = at[["peak_id", "role"]].copy()
    rec["tier_reason"] = rec["peak_id"].map(dict(zip(new["peak_id"], new["tier_reason"]))) if len(new) else pd.NA
    out["nox_skeleton_gate"] = T.nox_skeleton_summary(rec, profile)
    # 3. the degeneracy-keyed flags on the finished ledger. The O >= 11 flag is
    # the tier stage's: only on a row still committing its tier-time reading (a
    # later re-read's commit resets the flags in a run, so its row has none);
    # the oxygen-monster demote is the plausibility stage's, which runs after the
    # re-reads on the stamp carried by peak -- the finished ledger as it is
    tier_time = dict(zip(am["peak_id"], zip(am["neutral_formula"].map(T._txt), am["adduct"].map(T._txt))))
    same = ((ledger["role"] == L.ROLE_M0)
            & pd.Series([tier_time.get(p) == (T._txt(n), T._txt(a)) for p, n, a in
                         zip(ledger["peak_id"], ledger["neutral_formula"], ledger["adduct"])], index=ledger.index))
    L.ensure_flags(ledger)
    sub = ledger.loc[same].copy()
    out["below"] = int(T.flag_below_assignability(sub))
    for col in (L.FLAG_BELOW, "tier_reason"):
        ledger.loc[same, col] = sub[col]
    out["o_monster"] = int(PL.demote_oxygen_monsters(ledger, audit=audit, log=(lambda *_: None),
                                                     profile=profile)["o_demoted"])
    # 4. the file's own evidence level, in the pooled window
    out["levels"] = None
    if inputs.level_inputs is not None:
        pairs: list = []
        s = EV.apply_levels(ledger, cfg=cfg, run_inputs=_level_inputs_with(inputs.level_inputs, cal), pairs_out=pairs)
        out["levels"] = {"n_levelled": int(s.get("n_levelled", 0)), "levels": s.get("levels")}
        # 5. the per-file isotope-contradiction guard on those facts (the stage the file's own run
        # ran without facts: it was not levelled there), before the batch folds the file into its vote
        from peaky.assignment import isoguard as IG
        g = IG.guard_ledger(ledger, pairs[-1] if pairs else None, log=log)
        g["rows"] = g["rows"][:IG.ROWS_KEPT]
        out["isotope_guard"] = {**g, "source": T.CAL_SOURCE_POOLED}
    t = out["tiers"]
    log(f"[calpool]   window {cal[0]:+.3f} ± {cal[1]:.3f} ppm (centre: the {centre}'s; "
        f"width: the {out['cal']['width']}'s); "
        f"degeneracy measured on {out['degeneracy']['measured']} M0 rows "
        f"({out['degeneracy']['degenerate']} degenerate); tiers: {t['retiered']} re-tiered ({t['demoted']} "
        f"Assigned -> Candidate, {t['reworded']} reworded), {t['kept']} kept as a later stage left them; "
        f"{out['below']} below assignability, {out['o_monster']} oxygen monsters"
        + (f"; {out['levels']['n_levelled']} rows levelled" if out["levels"] else ""))
    return out
