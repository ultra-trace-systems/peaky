"""Top-level orchestrator + CLI.

Wires the spine (chemistry, contexts, ledger), the oracle (io_mascope), the
prescan (isotopes), and the three-pass director (passes) into one run and
records a reproducibility manifest with every locked module version.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field

import pandas as pd

from peaky.assignment import admission
from peaky.assignment import cleanup
from peaky.chem import contexts
from peaky.assignment import degeneracy
from peaky.io import io_mascope
from peaky.chem import isotopes
from peaky.assignment import labeled
from peaky.assignment import ladders
from peaky.assignment import ledger
from peaky.assignment import masscal
from peaky.assignment import passes
from peaky.assignment import evidence
from peaky.assignment import plausibility
from peaky.chem import reagents
from peaky.chem import resolution as RES
from peaky.assignment import reflists
from peaky.assignment import resolvability
from peaky.assignment import residual
from peaky.assignment import siloxane
from peaky.assignment import solvent_clusters
from peaky.assignment import tiers
from peaky.batch import timeseries

__version__ = "0.6.1"  # the v2 fit (0.6.0) with main's solvent_clusters stage
#                        (0.5.2) under it: a run stamped 0.6.0 was scored before
#                        that stage existed, so the two are told apart.
#                        0.6.0: every candidate scored with the v2 fit at the
#                        sample's own mass width, so no run of this version is
#                        comparable with a 0.5.x one; it is what a published run
#                        stamps.
#                        0.5.2: + solvent_clusters stage (source-solvent cluster
#                        ladders, pass-0 slot)
#                        0.5.1: a batch defers the hydrocarbon-on-N-cluster
#                        re-read to its merged ledger (run(reagent_n_relabel=False))


def _parsed_version(path: "Path") -> str | None:
    """A module's ``__version__`` string, read from its AST -- never imported.

    Importing 38 modules just to read a string would drag matplotlib, the Mascope
    SDK and `pipeline` into every run, and `assign.py` cannot import the batch
    orchestrator that CALLS it without inverting the dependency. Parsing sidesteps
    both, and is why every module can be registered instead of the subset this
    file happens to import.
    """
    import ast

    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError):              # unreadable / not valid python
        return None
    for node in tree.body:                      # module scope only, like a reader
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__version__" for t in targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _scan_module_versions() -> dict:
    """Walk the package and collect every module `__version__`. Cached: the files
    cannot change mid-process, and `run()` asks once per SAMPLE (a 230-spectrum
    batch would otherwise re-parse the tree 230 times)."""
    from pathlib import Path

    from peaky import paths
    root = Path(paths.PKG_ROOT)
    out: dict[str, str] = {}
    for py in sorted(root.rglob("*.py")):
        if py.name == "__init__.py":
            if py.parent == root:
                continue        # peaky/__init__.py holds the PACKAGE version, which
                                # the manifest records as code.package_version
            name = py.parent.name              # a sub-package (assignment/passes)
        else:
            name = py.stem
        version = _parsed_version(py)
        if version is None:
            continue
        if name in out:         # two files share a stem -> keep BOTH, the second
            name = py.relative_to(root).with_suffix("").as_posix()   # keyed by path
        out[name] = version
    return out


_MODULE_VERSIONS_CACHE: dict | None = None


def module_versions() -> dict:
    """Every versioned peaky module, `name -> __version__`, for the run manifest.

    DERIVED, not declared. The hand-written dict this replaces listed only the
    modules `assign.py` imports at module level, so it had gone 17 modules stale
    -- `traces` and `pipeline`, carrying the newest per-peak admission and
    set-cover selection behaviour, reported `None` in every manifest. A registry
    read off the package tree cannot drift from it.

    Keys are module names (`assign`, `traces`), a sub-package taking its own name
    (`passes`); `code.module_hashes` pins the same files by sha1 under their
    package-relative paths. Returns a fresh dict -- the cache is not the caller's
    to mutate.
    """
    global _MODULE_VERSIONS_CACHE
    if _MODULE_VERSIONS_CACHE is None:
        _MODULE_VERSIONS_CACHE = _scan_module_versions()
    return dict(_MODULE_VERSIONS_CACHE)


def _degen_summary(led, channels=None, families=None) -> dict:
    """Compact manifest summary of the degeneracy audit: how many M0 rows it
    measured, how many it could only bound (the committed formula outside the
    run's space) or not measure, and the channels / opened families it counted."""
    m0 = led[led["role"] == ledger.ROLE_M0]
    note = m0["degeneracy_note"].astype(str) if "degeneracy_note" in m0.columns else pd.Series(dtype=str)
    out = {"not_measured": int(note.str.startswith("not measured").sum()),
           "lower_bound": int(note.str.contains("count is a lower bound", regex=False).sum())}
    if channels is not None:
        out["channels"] = list(channels)
    if families is not None:
        out["families"] = list(families)
    d = pd.to_numeric(m0["degeneracy_density"], errors="coerce").dropna() \
        if "degeneracy_density" in m0.columns else pd.Series(dtype=float)
    if not len(d):
        return {"measured": 0, **out}
    d = d.astype(int)
    return {"measured": int(len(d)), "degenerate_ge2": int((d >= 2).sum()),
            "max_density": int(d.max()), **out}


def _stage_degeneracy(st):
    """The honest mass-degeneracy count (degeneracy.py) over what THIS run could
    commit: its channels (the adducts it scored), the context's element budget
    widened by the contaminant families this file opened -- declared by the
    context, the reagent's organohalogen family, the GKA evidence pass 3 carried --
    and the curated formulas (the pass-0 registry, the active reference lists)."""
    polarity = getattr(st.profile, "polarity", "negative")
    curated = passes.known_formulas(polarity, getattr(st.profile, "label", None))
    curated = curated | frozenset(getattr(st.cfg, "reflist_formulas", None) or ())
    carry = st.series_carry or {}
    families = degeneracy.opened_families(
        st.profile if st.do_pass3 else None, st.reagent if st.do_pass3 else None,
        carry.get("evidence"), st.led)
    # the calibration the count's window is centred on (tiers._calibrate on the
    # ledger as it stands here; None = uncalibrated, the count is skipped) --
    # kept on the run state: the per-file evidence level's step-1 window is this
    # SAME (mu, sigma), and the run's stats persist it (`degeneracy_cal`)
    cal = tiers._calibrate(st.led[st.led["role"] == ledger.ROLE_M0],
                       st.led.loc[st.led["role"] == ledger.ROLE_ISO, "parent_peak_id"].value_counts())
    st.degeneracy_cal = None if cal is None else (float(cal[0]), float(cal[1]))
    degeneracy.apply_degeneracy(st.led, cal=cal, context=st.profile, adducts=st.adducts,
                                families=families, curated=curated, log=st.log)
    return _degen_summary(st.led, st.adducts, families)


def _module_hashes() -> dict:
    """sha1 of each module file -- the manifest must pin the EXACT code of a
    run; static version strings are not bumped on every edit (v15-vs-v16
    lesson: two runs differed only via an un-versioned passes.py edit).

    Anchored at the package root and RECURSIVE (`rglob`) so it keeps pinning every
    module regardless of sub-package nesting; keys are package-relative POSIX paths
    (e.g. `chem/chemistry.py`) so they stay unique across sub-packages."""
    import hashlib
    from pathlib import Path

    from peaky import paths
    d = Path(paths.PKG_ROOT)
    return {p.relative_to(d).as_posix(): hashlib.sha1(p.read_bytes()).hexdigest()[:12]
            for p in sorted(d.rglob("*.py"))}


@dataclass
class _RunState:
    """Mutable run context threaded to every pipeline stage."""

    client: object
    sample_id: str
    led: object
    profile: object
    pre: object
    cfg: object
    adducts: list
    reagent: object
    has_halogen: bool
    do_pass2: bool
    do_pass3: bool
    do_pass4: bool
    do_pass5: bool
    do_pass_certified: bool
    reflists_active: object
    ts_peaks: object
    label_isotope: object
    label_max: int
    log: object
    checkpoint_dir: object
    # False on a batch: the hydrocarbon-on-N-cluster re-read is decided once on
    # the merged ledger (assign_batch.run), not per file -- its skip key is a
    # presence test that flips with each file's S/N.
    reagent_n_relabel: bool = True
    # GKA series evidence measured in the pass-3 CURATED phase and handed to the
    # late `pass3_series` stage (a DataFrame + member sets; never serialized).
    series_carry: object = None
    # the neutral formulas a --corroborate source holds: accepted for the
    # caller's record only -- the per-file evidence level takes no partners
    # (a batch's merge vote reads the cross set in the parent)
    corroborate: set = field(default_factory=set)
    # the peak-width model the `resolvability` stage reads (chem.resolution);
    # None = no model, the stage is skipped and its columns stay NA; it also
    # sets the evidence level's instrument class (no model: not assessed)
    resolving_power: object = None
    # the reagent halogen the evidence levels read (evidence.channel_halogen of
    # the declared channels, C43); DETECT_HALOGEN = count the committed clusters
    reagent_halogen: object = evidence.DETECT_HALOGEN
    # the reagent profile's NAME (chem.profiles; the evidence level's
    # enumeration space); None = found from the run's adducts
    reagent_profile: object = None
    # the degeneracy stage's calibration (mu, sigma) ppm; None = uncalibrated;
    # "absent" = the stage did not run (stats then record no `degeneracy_cal`)
    degeneracy_cal: object = "absent"
    # how the run's reference lists were activated (levels.lists.activation_record:
    # {tags, matched}); None = not recorded (the context source then says so)
    reflists_context: object = None
    summaries: dict = field(default_factory=dict)
    plaus_audit: list = field(default_factory=list)


@dataclass
class _Stage:
    """One pipeline step. ``fn(st)`` does the work; ``when(st)`` gates it; ``safe``
    wraps it so a failure can't lose prior work; ``store`` keeps its summary."""

    name: str
    fn: object
    when: object = (lambda st: True)
    safe: bool = True
    store: bool = True


def _checkpoint(st, tag):
    if st.checkpoint_dir:
        from pathlib import Path
        Path(st.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        st.led.to_csv(
            Path(st.checkpoint_dir) / f"{st.sample_id}_ledger_{tag}.csv", index=False)


def _safe(st, tag, fn):
    """Run a stage; a failure (e.g. a server 500) must not lose prior passes' work."""
    import time as _time

    t0 = _time.time()
    try:
        s = fn()
    except Exception as e:  # noqa: BLE001
        st.log(f"[run] {tag} FAILED: {type(e).__name__}: {e}")
        s = {"committed": 0, "locked": 0, "iso_attached": 0, "error": str(e)}
    s["elapsed_s"] = round(_time.time() - t0, 1)
    st.log(f"[run] {tag} took {s['elapsed_s']}s")
    _checkpoint(st, tag)
    return s


def _stage_pass3_curated(st):
    """Pass 3, first half: cluster resolution + the profile's own families. The
    GKA-opened families are held back to `pass3_series` (see run_pass3.__doc__)."""
    late = getattr(st.cfg, "pass3_series_late", True)
    res = passes.run_pass3(
        st.client, st.sample_id, st.led, st.profile, st.pre, st.cfg, st.adducts,
        log=st.log, phase="curated" if late else "all")
    # only the curated phase emits a carry, so `pass3_series` self-disables when
    # the knob is off -- the stage's `when` needs no second condition.
    st.series_carry = res.pop("_carry", None)
    return res


def _stage_pass3_series(st):
    """Pass 3, second half: the evidence-opened families, claiming from whatever
    passes 4/5/7 could not explain. Evidence comes from the curated phase."""
    return passes.run_pass3(
        st.client, st.sample_id, st.led, st.profile, st.pre, st.cfg, st.adducts,
        log=st.log, phase="series", carried=st.series_carry)


def _stage_composite(st):
    """Composite detection + de-blend. Halide-CIMS only -- in positive urea mode
    the even-shift residual is ordinary 13C2/18O/34S structure, not a co-component,
    so the test is skipped. split_composites appends rows, so rebind st.led."""
    if st.has_halogen:
        st.summaries["composite"] = _safe(
            st, "composite", lambda: passes.detect_composites(st.led, st.cfg, log=st.log))
        n_before = len(st.led)
        st.led = passes.split_composites(st.led, st.cfg, log=st.log)
        st.summaries["composite_split"] = {"split": len(st.led) - n_before}
    else:
        st.summaries["composite"] = {"flagged": 0, "skipped": "no halogen adduct"}
        st.summaries["composite_split"] = {"split": 0}
        st.log("[run] composite test skipped (no halogen adduct -- even-shift "
               "residual is isotope structure, not a co-component signature)")
    _checkpoint(st, "audit")


def _stage_reagent_post(st):
    """Final reagent sweep: catch cluster peaks the passes left unexplained, then
    AUTHORITATIVELY reclaim any reagent-cluster mass a pass committed an analyte M0
    onto (the urea `[R_n+H]+`/`[R_n+NH4]+` == `CHNO`/`CH4N2O` degeneracy)."""
    n = reagents.label_reagents(st.led, st.reagent, ppm=12.0)
    if n:
        st.log(f"[run] post-labeled {n} more reagent-cluster peaks")
    reagents.reclaim_reagent_clusters(st.led, st.reagent, ppm=12.0, log=st.log)
    # claim the bright ¹³C/¹⁵N (and halide heavy-isotope) satellites of the reagent
    # ions -- they otherwise dominate the unexplained residual (the urea-dimer ¹³C/¹⁵N
    # at 122.075/122.069 were the two biggest 'unexplained' peaks of the batch).
    reagents.label_reagent_isotopologues(st.led, log=st.log)


def _stage_timeseries(st):
    """Time-resolved disposition when a batch TS is supplied (runs last)."""
    ts_reagent_mzs = None
    if st.reagent in reagents._POSITIVE_REAGENTS or st.reagent == "EasyIC":
        ts_reagent_mzs = [m for (_l, m, _f) in reagents.build_library(st.reagent)]
    return timeseries.apply_timeseries(
        st.led, st.ts_peaks, reagent_mzs=ts_reagent_mzs, log=st.log)


def _stage_plausibility(st):
    """The shared-oracle demotes (O-monster, carbon cluster) and the element-budget
    demote: a commit whose neutral lies outside the run context's element budget
    (contexts.element_budget) and that no curated list names -- the pass-0
    registry for this polarity/context or an active reference list -- is
    Candidate + tentative_lead (plausibility.demote_off_budget)."""
    label = getattr(st.profile, "label", None)
    curated = passes.known_formulas(getattr(st.profile, "polarity", "negative"), label)
    curated = curated | frozenset(getattr(st.cfg, "reflist_formulas", None) or ())
    return plausibility.demote_implausible(
        st.led, audit=st.plaus_audit, log=st.log, context=label, curated=curated)


def _profile_name_for(adducts) -> str | None:
    """The registered reagent profile whose analyte channels are exactly
    `adducts` (the run's forced or detected list, before the side channels
    join): its name, or None when no profile -- or more than one -- matches."""
    from peaky.chem import profiles as PR
    want = {str(a) for a in (adducts or ())}
    if not want:
        return None
    names = sorted({p.name for p in PR._BY_ALIAS.values() if set(p.adducts) == want})
    return names[0] if len(names) == 1 else None


def _stage_evidence(st):
    """The evidence level of every committed M0 row on the scale of peaky
    0.10.0 (docs/EVIDENCE_LEVELS.md), the file levelled ALONE in "adapted"
    mode (evidence.apply_levels): every file-count minimum 1, no time series,
    no merged ledger, no partners; the step-1 window is the degeneracy stage's
    own calibration, the height gate this run's resolved gate. The instrument
    class comes from the run's width model: without one (or a TOF-class one)
    every row reads NA. Runs after every tier and demote stage, the reflist
    rescue and the final envelope sweep, before `timeseries`; it changes no
    tier. The claim each level supports is stamped beside it and tallied on
    the log line. A batch's merged ledger is levelled again on the POOLED
    files (assign_batch.run); the per-file ledgers keep this per-file level."""
    try:
        gate = float(st.cfg.height_cutoff) if st.cfg is not None else None
    except Exception:  # noqa: BLE001 -- an unresolved gate falls back to the noise edge
        gate = None
    reagent = getattr(st, "reagent_profile", None) or _profile_name_for(getattr(st, "adducts", None))
    ri = evidence.file_run_inputs(
        sample_id=getattr(st, "sample_id", "") or "file", reagent=reagent,
        context=getattr(getattr(st, "profile", None), "label", None) or "ambient-air",
        resolution=getattr(st, "resolving_power", None),
        reflists_active=reflists.active_versions(getattr(st, "reflists_active", None)),
        height_gate_cps=gate, noise_edge_cps=getattr(st.cfg, "noise_edge_cps", None) if st.cfg is not None else None,
        degeneracy_cal=getattr(st, "degeneracy_cal", "absent"),
        activation=getattr(st, "reflists_context", None),
        reagent_halogen=getattr(st, "reagent_halogen", evidence.DETECT_HALOGEN))
    s = evidence.apply_levels(st.led, cfg=st.cfg, run_inputs=ri)
    claims = s.get("claims") or {}
    inst = s.get("instrument") or {}
    st.log(f"[run] evidence levels {s['levels']} on {s['n_levelled']} M0 rows "
           f"({s['n_pairs']} neutral/adduct pairs; levelled alone, instrument class "
           f"{inst.get('class') or 'unknown'}); "
           "claims " + " | ".join(f"{k} {claims.get(k, 0)}" for k in evidence.CLAIM_KEYS))
    return s


def _stage_resolvability(st):
    """Nearest-neighbour separability of every M0 peak from the run's width
    model (assignment/resolvability.py): a tier input (a blended peak with no
    isotope / second-channel / series corroboration is capped at Candidate) and
    a fact of the merge vote's class. Skipped without a model, and
    when the ledger already carries the flag (the trace-first synthetic sample
    stamps its own at the trace build)."""
    return resolvability.stamp_resolvability(st.led, st.resolving_power, log=st.log)


def _width_model(resolving_power, client, sample_id, raw, log):
    """The peak-width model the `resolvability` stage reads (chem.resolution):
    None -> None; 'auto' -> measured from this sample's raw profile when a
    server is there, None offline; a number -> constant R; a Resolution as is.
    A measurement that cannot be made is a log line, never a failed run: the
    stage then skips and the level is computed without it."""
    if resolving_power is None:
        return None
    if isinstance(resolving_power, RES.Resolution):
        return resolving_power
    if isinstance(resolving_power, str):
        if resolving_power.strip().lower() == "none":
            return None
        if resolving_power.strip().lower() != "auto":
            return RES.Resolution.from_r(float(resolving_power))
        if client is None:
            log("[resolvability] no server to measure the peak width from (offline sample); "
                "pass resolving_power=<R> to declare one")
            return None
        from peaky.batch import tracefirst as TFT   # lazy: the batch package imports this module
        return TFT.measure_resolution(client, sample_id, peaks=raw, log=log)
    return RES.Resolution.coerce(resolving_power)


# The assignment pipeline AS DATA -- read top to bottom to see exactly what runs,
# in what order, under what condition. `safe` wraps a stage so a failure can't lose
# prior work; `store` keeps its summary. Authoritative stage table: ARCHITECTURE.md §4.
_STAGES = [
    _Stage("pass0", lambda st: passes.run_pass0_known(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts, log=st.log)),
    # Source-solvent CLUSTER ladders ([S_n+H]+ / [S_n-H]+ and their -H2O
    # condensation rung). Pass-0 style and in pass 0's slot -- offline (the ion
    # masses are exact and the evidence is the ladder, so no scorer is asked),
    # BEFORE pass 1 and locked, so the grid never re-reads a cluster mass as a
    # covalent molecule. Self-gating: a no-op unless the context declares source
    # solvents AND this spectrum shows their monomer ions.
    _Stage("solvent_clusters", lambda st: solvent_clusters.assign_solvent_clusters(
        st.led, st.profile, st.cfg, log=st.log),
           when=lambda st: bool(solvent_clusters.solvents_for(st.profile))),
    _Stage("pass1", lambda st: passes.run_pass1(
        st.client, st.sample_id, st.led, st.profile, st.pre, st.cfg, st.adducts, log=st.log)),
    # self-calibrate the mass gate on the pass-1 backbone, then re-grade pass-1's
    # pre-calibration confidence labels against the fitted center.
    _Stage("calibrate", lambda st: passes.calibrate(st.led, st.cfg, log=st.log),
           safe=False, store=False),
    _Stage("relabel", lambda st: passes.relabel_confidence(st.led, st.cfg, log=st.log),
           safe=False, store=False),
    _Stage("pass2", lambda st: passes.run_pass2(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts, log=st.log),
           when=lambda st: st.do_pass2),
    _Stage("pass3", _stage_pass3_curated, when=lambda st: st.do_pass3),
    # claim each committed peak's full M+2/M+4 envelope BEFORE pass 4, then free the
    # bright low-carbon CHON mass-fits whose 13C contradicts the carbon count.
    _Stage("iso_env_pre4",
           lambda st: passes.complete_isotope_envelopes(st.led, st.cfg, log=st.log),
           when=lambda st: st.do_pass4),
    _Stage("carbon_clamp_pre4",
           lambda st: {"demoted": passes.demote_carbon_inconsistent(st.led, st.cfg, log=st.log)},
           when=lambda st: st.do_pass4),
    _Stage("pass4", lambda st: residual.explain_residual(
        st.client, st.sample_id, st.led, st.profile, st.pre, st.cfg, st.adducts,
        reagent=st.reagent, log=st.log), when=lambda st: st.do_pass4),
    _Stage("pass5", lambda st: passes.run_pass5_completion(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts, log=st.log),
           when=lambda st: st.do_pass5),
    # Pass 7: certified-neutral discovery over the residual -- multi-channel /
    # cluster-ladder convergence licenses off-grid (P/S/Cl) formula space that
    # the per-peak grid forbids. The pass-5 INVERSE: from unknown peak groups
    # to a licensed neutral, not from known neutrals to their partners. Before
    # the audits so the calibrated mass gate judges its commits like any other.
    _Stage("pass_certified", lambda st: passes.run_pass_certified(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts,
        reagent=st.reagent, ts_peaks=st.ts_peaks, log=st.log),
           when=lambda st: st.do_pass_certified),
    # Pass 3, LATE half: families opened by detected GKA series structure claim
    # only what passes 4/5/7 left behind. Ordering matters -- run before pass 3
    # these out-competed the better-evidenced passes for the same unexplained
    # peaks (2.0 % of their commits reached `Assigned` against a 41 % baseline).
    _Stage("pass3_series", _stage_pass3_series,
           when=lambda st: st.do_pass3 and st.series_carry is not None),
    # post-run audits: apply the calibrated mass gate to pre-calibration commits.
    _Stage("audit_iso", lambda st: passes.audit_isotopes(st.led, st.cfg, log=st.log), safe=False),
    _Stage("audit", lambda st: passes.audit_mass_gate(st.led, st.cfg, log=st.log), safe=False),
    _Stage("iso_env_post",
           lambda st: passes.complete_isotope_envelopes(st.led, st.cfg, log=st.log)),
    _Stage("composite", _stage_composite, safe=False, store=False),
    _Stage("reagent_post", _stage_reagent_post,
           when=lambda st: bool(st.reagent), safe=False, store=False),
    # Pass 6: anchored ladder gap-fill -- AFTER the audits so their gates don't clear
    # its pattern-evidenced completions. Then a 3rd envelope sweep for the di-bromide
    # SOA cores that commit only in pass 6.
    _Stage("pass6_ladder", lambda st: ladders.run_ladder_gapfill(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts, log=st.log),
           when=lambda st: st.do_pass5),
    _Stage("iso_env_post6",
           lambda st: passes.complete_isotope_envelopes(st.led, st.cfg, log=st.log)),
    _Stage("cleanup", lambda st: cleanup.run_cleanup(
        st.client, st.sample_id, st.led, st.profile, st.cfg, log=st.log)),
    _Stage("siloxane", lambda st: siloxane.assign_siloxane_ladder(
        st.client, st.sample_id, st.led, st.profile, st.cfg, adducts=st.adducts, log=st.log)),
    # labelled-reagent covalent heavy-isotope rescue (e.g. 15N-organonitrate
    # products) -- runs BEFORE degeneracy/tiers so the filled/re-read peaks are
    # tiered normally. No-op unless the profile declares a label_isotope.
    _Stage("labeled_15n", lambda st: labeled.rescue_labeled(
        st.client, st.sample_id, st.led, st.profile, st.cfg, adducts=st.adducts,
        label_isotope=st.label_isotope, label_max=st.label_max, log=st.log),
        when=lambda st: bool(st.label_isotope)),
    # re-arbitrate off-calibration, uncorroborated aromatic-monster winners against
    # their stored on-cal plausible alternatives -- applies the tier engine's
    # calibration-sigma + corroboration gate AT WINNER-SELECTION (before degeneracy /
    # tiers see the committed formula), so a degenerate competitor the local scorer
    # over-ranked can't keep the M0 slot it will only ever be tier-demoted out of.
    _Stage("rearbitrate", lambda st: passes.rearbitrate_offcal_degenerate(
        st.led, st.cfg, log=st.log)),
    # separability of each M0 peak from its nearest picked neighbour -- MUST precede
    # tiers (a blended, uncorroborated peak is capped) and the merge vote's class (reads it).
    _Stage("resolvability", _stage_resolvability,
           when=lambda st: st.resolving_power is not None and not resolvability.already_stamped(st.led),
           safe=False),
    # honest mass-degeneracy measurement -- MUST precede tiers (the tier engine reads it).
    _Stage("degeneracy", _stage_degeneracy),
    # report tier, then the post-tier de-risking demotes (each gets the last word).
    _Stage("tiers", lambda st: tiers.apply_tiers(st.led, cfg=st.cfg), safe=False, store=False),
    _Stage("demote_fluorine",
           lambda st: cleanup.demote_unconfirmed_fluorine(st.led, log=st.log),
           safe=False, store=False),
    _Stage("demote_carbon",
           lambda st: cleanup.demote_implausible_carbon(st.led, log=st.log),
           safe=False, store=False),
    # relabel hydrocarbon FG-cluster anions (C6H6 [M+CO3]-) as radical anions M-.
    # of the closed-shell oxygenated neutral BEFORE the hydrocarbon demote, so a
    # real M-. (corroborated by [M-H]-/[M+Br]-) is shown as its true neutral.
    _Stage("relabel_radicals",
           lambda st: cleanup.relabel_radical_anions(st.led, log=st.log),
           safe=False, store=False),
    # positive-mode arbitration: a pure hydrocarbon via an N-carrying reagent
    # cluster ([M+NH4]+ / uronium) is re-read as [M+H]+ of an N-heterocycle.
    # Skipped on a batch (reagent_n_relabel=False): the rule's skip key -- does
    # the hydrocarbon show its own [M+H]+ -- is a presence test that flips with
    # each file's S/N, so per file it split one ion into two readings across the
    # batch; assign_batch.run applies the same rule ONCE to the merged ledger.
    _Stage("relabel_reagent_n",
           lambda st: cleanup.relabel_reagent_n_adducts(st.led, log=st.log),
           when=lambda st: bool(st.reagent_n_relabel),
           safe=False, store=False),
    # ¹⁵N-ammonium in-source DECLUSTERING cascade: [M+^NH4]+ -> [M+H]+ ->
    # [M+H-H2O]+, and [M+^NH4]+ -> [M+^NH4-H2O]+ (both product routes seen in
    # MS2 of the 2026-09-10 exploratory file). The dehydration ions are ion-
    # identical to the alkene/enone X-H2O on [M+H]+ / [M+^NH4]+; re-read them
    # onto the hydrate X when X is corroborated on its own labelled channel.
    _Stage("nh4_dehydration",
           lambda st: cleanup.relabel_ammonium_dehydration(st.led, log=st.log),
           when=lambda st: any("^NH4" in str(a) for a in st.adducts),
           safe=False, store=False),
    # EasyIC⁺ fragmentation ambiguity: relabel corroborated alcohol-dehydration
    # ions ([CnH2n+H]+ -> [CnH2n+2O+H-H2O]+) and stamp the MS1-irreducible
    # carbonyl-vs-alcohol / fragment-vs-intact / cluster-vs-covalent dual
    # readings into commentary. The cluster note is the complement of the
    # `solvent_clusters` stage above: that stage COMMITS a cluster reading when
    # the whole ladder is behind it, this one records the reading beside the
    # covalent one on the rows it could not take.
    # With a batch TS, the dehydration pair (ion and its +O hydride partner)
    # can ALSO be corroborated by time-correlation -- one spectrum cannot tell
    # butene+MEK from dehydrated butanol, but the batch can.
    _Stage("easyic_ambiguity",
           lambda st: cleanup.annotate_easyic_ambiguity(
               st.led, ts_peaks=st.ts_peaks, profile=st.profile, log=st.log),
           when=lambda st: getattr(st.profile, "label", "") == "easyic",
           safe=False, store=False),
    # ¹⁵N-nitrate isobar arbitration: a covalent organonitrate [Y−H]- whose cluster
    # parent X = Y−HNO₃ is independently detected is really the chamber-¹⁴NO₃ cluster
    # [X+NO₃]- (exact isobar; ¹⁴NO₃ is off the labelled scoring grid). Tier preserved.
    # No-op unless the run is the labelled-nitrate profile (label_isotope '^N').
    _Stage("relabel_nitrate_clusters",
           lambda st: cleanup.relabel_nitrate_clusters(st.led, log=st.log),
           when=lambda st: st.label_isotope == "^N", safe=False, store=False),
    _Stage("demote_ionization",
           lambda st: cleanup.demote_implausible_ionization(st.led, log=st.log),
           safe=False, store=False),
    _Stage("demote_speculative",
           lambda st: cleanup.demote_speculative_residual(st.led, st.cfg, log=st.log),
           safe=False, store=False),
    _Stage("plausibility", _stage_plausibility, safe=False),
    # rescue-verify the still-unexplained residual against active reference peaklists.
    _Stage("reflist_rescue", lambda st: reflists.rescue_unexplained_by_reflist(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.reflists_active,
        st.adducts, log=st.log), when=lambda st: bool(st.reflists_active)),
    # ION-ONLY rows: the +1.0078 Da electron-attachment line beside each committed
    # [M-H]- acid, committed as a Candidate "[M]-." whose composition is pinned
    # (exact mass, own 13C) while the ionization process and the neutral stay
    # open. Post-tier (it sets its own tier, like reflist_rescue), BEFORE the
    # final envelope sweep (which then claims the new row's own 13C), `evidence`
    # (which reads its channel as the ion only and never lets it corroborate
    # its parent) and `timeseries` (which stamps it). Only where the profile
    # opened the channel (cfg.ion_only_channels); not `safe`: a bucket that
    # cannot be filled is a bug, not a lost stage.
    _Stage("ion_only", lambda st: cleanup.commit_ion_only_electron_attachment(
        st.led, st.cfg, log=st.log),
           when=lambda st: bool(getattr(st.cfg, "ion_only_channels", None)), safe=False),
    # final envelope sweep: an M0 committed AFTER the three earlier sweeps (a pass-8
    # reflist rescue, or a parent whose earlier assignment was cleared and re-won,
    # orphaning its children) still owes its satellites -- without this its bright
    # 13C line sits in the residual (the NBBS urea-adduct M+1 was the single
    # brightest "unexplained" peak of a positive urea-CIMS ambient batch run).
    _Stage("iso_env_final",
           lambda st: passes.complete_isotope_envelopes(st.led, st.cfg, log=st.log)),
    # evidence levels -- not `safe`: a level that cannot be computed is a bug,
    # not a lost stage. The stage order above is the design (spec §6.1).
    _Stage("evidence", _stage_evidence, safe=False),
    _Stage("timeseries", _stage_timeseries,
           when=lambda st: st.ts_peaks is not None and len(st.ts_peaks)),
]


#: C42: the scoring trend is the calibration's own when the two would put a line
#: within this many ppm of each other anywhere either was fitted (masscal.trend_shift)
TREND_AGREE_PPM = 0.05
#: ... and the file is re-run at most this many times to get there
MAX_TREND_RERUNS = 2


def _trend_step(scoring_trend, fitted, *, local: bool, reruns: int) -> str:
    """What a run does after a calibrate stage (C42), given the trend the file
    is scored at (`scoring_trend`, None = the constant offset) and the trend
    this pass's calibration accepted (`fitted`, None = rejected):

    'rerun'     -- score at `fitted` and re-run from pass 0: no scoring trend
                   yet, or the fit moved more than TREND_AGREE_PPM since the
                   last one (the first fit misses the low masses the constant
                   offset left unexplained), while re-runs remain;
    'keep_gates'-- the file is scored at a trend this calibration rejects: the
                   gates take the scoring trend, so one centre judges the file;
    'network'   -- a trend was accepted but the network scorer judges one
                   constant offset: nothing to re-run (the gates use the fit);
    'done'      -- nothing to change."""
    if fitted is None:
        return "keep_gates" if scoring_trend is not None else "done"
    if not local:
        return "network"
    if reruns >= MAX_TREND_RERUNS:
        return "done"
    if scoring_trend is None or masscal.trend_shift(scoring_trend, fitted) > TREND_AGREE_PPM:
        return "rerun"
    return "done"


def run(sample_id: str, context: str = "ambient-air", *,
        cfg: passes.PassConfig | None = None, use_cache: bool = True,
        do_pass2: bool = True, do_pass3: bool = True, do_pass4: bool = True,
        do_pass5: bool = True, do_pass_certified: bool = True,
        ts_peaks=None, adducts=None, reflists_active=None,
        label_isotope=None, label_max=2, label_purity=None, occurrence=None,
        reagent_n_relabel: bool = True, peaks=None, corroborate=None,
        resolving_power=None, log=print, checkpoint_dir=None, scoring=None,
        reagent_profile=None, reflists_context=None) -> dict:
    """Assign one sample. `peaks` (a DataFrame in the shape fetch_peaks returns:
    peak_id, mz, height, area ...) makes the run OFFLINE: the table is served
    as `sample_id` from memory, no server is contacted, the local scorer does
    the mass and isotope maths, and only the given `adducts` are open. The
    trace-first batch path and the tests use it. `corroborate` (a set of neutral
    formulas) is accepted for the caller's record and read by nothing here: the
    per-file evidence level takes no other-source partners, and a batch's merge
    vote reads its cross set in the parent (`evidence.vote_cross_neutrals`).
    `reagent_profile` names the reagent profile (chem.profiles) whose adducts span
    the evidence level's enumeration space (None: the profile whose analyte
    channels are exactly the run's adducts); `reflists_context` is how the
    caller's reference lists were activated (`levels.lists.activation_record`),
    printed as the evidence level's context source.
    `resolving_power` is the peak-width model of the `resolvability` stage (and
    the evidence level's instrument class: without one every row reads NA): None =
    no stage (its columns stay NA), 'auto' = measure it from this sample's raw
    profile when a server is there (an offline run cannot), a number = a constant
    R, or a `chem.resolution.Resolution`; a batch measures ONE model and hands it
    to every file. `scoring` (offline runs only) is what the offline sample is
    judged at -- a measured sample's `pattern_scoring` snapshot, a
    `PatternScoring` or an instrument class (`io_mascope.register_offline_sample`);
    None leaves an offline sample at the forgiving class fallback."""
    cfg = cfg or passes.PassConfig()
    # the reagent bottle's isotopic purity (ReagentProfile.purity), published for
    # the two consumers that model a '^X' ion's unlabelled impurity line: the
    # local scorer's predict_isotopes call and this package's own envelope
    # predictor (isotopes.isotope_pattern). None restores the default.
    purity = isotopes.set_label_purity(label_purity)
    # reference-list selection prior: a candidate neutral on an active reference
    # peaklist wins a near-tie over a mass coincidence (arbitrate reads this set).
    # The manifest records which list versions built it (`reflists_active`).
    reflist_versions = reflists.active_versions(reflists_active)
    if reflists_active:
        cfg.reflist_formulas = reflists.prior_formulas(reflists_active)
    profile = contexts.get_context(context)
    if scoring is not None and peaks is None:
        raise ValueError("scoring= is what an OFFLINE sample (peaks=) is judged at; a server sample is "
                         "judged at its own record and matches")
    if peaks is not None:
        if not io_mascope._local_scoring_enabled():
            raise RuntimeError("an offline sample (peaks=) needs the local scorer; "
                               "unset PEAKY_LOCAL_SCORING")
        io_mascope.register_offline_sample(sample_id, peaks, scoring=scoring)
        client = None
    else:
        client = io_mascope.connect()

    raw = io_mascope.fetch_peaks(client, sample_id, use_cache=use_cache)
    led = ledger.new_ledger(raw)
    width_model = _width_model(resolving_power, client, sample_id, raw, log)
    # The sample's noise edge (p1 of its picked heights) anchors every height
    # gate in the passes (cfg.height_cutoff = x_edge * edge): absolute cps
    # thresholds do not transfer between instruments/modes (see passes.config).
    cfg.noise_edge_cps = passes.noise_edge(led["height"]) if "height" in led.columns else None
    if cfg.noise_edge_cps is None and cfg.height_cutoff_cps is None:
        # fail closed here, with the sample named, rather than deep inside the
        # first height-gated pass (PassConfig.height_cutoff raises when unresolved)
        raise RuntimeError(
            f"sample {sample_id!r} has no finite peak heights, so its noise edge cannot "
            "be measured and the height gate cannot be resolved -- pass "
            "PassConfig(height_cutoff_cps=...) to gate on an absolute value")
    log(f"[run] noise edge {cfg.noise_edge_cps if cfg.noise_edge_cps is None else round(cfg.noise_edge_cps, 3)} cps "
        f"-> height_cutoff {cfg.height_cutoff:.3g} cps "
        + ("(absolute override)" if cfg.height_cutoff_cps is not None
           else f"({cfg.height_cutoff_x_edge_resolved:g}x edge)"))
    # Admission gate: brightness OR persistence. `occurrence` is the batch's
    # per-bin occurrence table (assign_batch computes it from the batch time
    # series); without it the gate is brightness alone (single-sample runs).
    adm = admission.stamp_admission(led, cfg, occurrence)
    _thr = adm.get("occurrence_threshold")
    log(f"[run] admission: {adm['height']} peaks by height, {adm['occurrence']} by "
        f"persistence only ("
        + (f"bin occurrence >= {_thr:.2f} [{adm.get('occurrence_min')}] of "
           f"{occurrence.attrs.get('n_samples', '?')} spectra" if _thr is not None
           else ("persistence path off" if occurrence is not None else "no batch table"))
        + f"), {adm['rejected']} not eligible")
    # Adducts are normally detected from the sample's own server matches (the
    # SKILL design rule for mixed-reagent datasets). But a batch with a KNOWN
    # reagent can pass `adducts=` to force the analyte channels: per-sample match
    # detection is unreliable when a sample has few/no server matches (a positive
    # sample with no urea-channel match then falls back to [M-H]- and the whole
    # spectrum is mis-assigned in the wrong polarity). The explicit list wins.
    adducts_given = bool(adducts)
    adducts = list(adducts) if adducts_given else io_mascope.detect_adducts(raw)
    if not adducts_given and not io_mascope.recognised_adducts(raw):
        # no channels given and none of the sample's own matches names one: the
        # [M-H]- detect_adducts returned is a DEFAULT, not a reading -- say so (the
        # CLI and MCP entry points stop before this; a library caller passes adducts=)
        log("[reagent] WARNING: no adducts given and no server match names a known "
            "channel; assuming [M-H]- (negative). Pass adducts= (or --reagent) for a "
            "positive or sparse-match sample.")
    analyte_adducts = list(adducts)     # before the side channels join (the profile match)
    # Polarity is read from the (detected or forced) adducts (cation forms end "+").
    polarity = "positive" if any(str(a).rstrip().endswith("+") for a in adducts) \
        else "negative"
    # SIDE CHANNELS: the adducts the reagent profile DECLARES beside its analyte
    # channels (ReagentProfile.side_channels: the uronium profile's [M+NH4]+), or
    # the caller's explicit set (`--side-channels`, PassConfig.side_channels),
    # scored only if the server has the mechanism registered (auto-selection
    # covers only the sample's own channels, so the ids must be passed
    # explicitly). Nothing is opened by polarity: a channel offered to every peak
    # doubles the alias space, and the polarity-wide set this replaced (carbonate
    # and di-bromide on every negative run, sodium and ammonium on every positive
    # one) read, on the measured batches, ions the run already held under another
    # reading or ions with no support of their own. Opt-in examples: [M+CO3]- on
    # a source with real CO3- chemistry (carbonyl.CO3- air ions), [M+Br2]- for
    # di-bromide clusters of analytes on a bromide source. NB: [M+Br3]- is a poor
    # choice -- as a blanket channel it lost 40 base M0s (incl. TFA) for 0 gains
    # (Br3- is the dominant reagent ion).
    # LABELLED-AMMONIUM run: the 14N [M+NH4]+ adduct is the reagent's ~2 % 14N
    # impurity satellite (-0.99703 Da, modelled by the scorer's ^N purity), not
    # an analyte channel -- enumerating it would only re-claim those satellites
    # as bogus M0s (see profiles.NH4_15N). ... and NO alkali channel either:
    # [X+Na]+ sits 0.2 mDa from [(X-O2+C2H4)+^NH4]+ (Na - ^NH4 = 3.9584 Da;
    # C2H4 - O2 = 3.9585 Da), so every ^NH4 adduct of an O>=2 neutral has a
    # Na-adduct hydrocarbon twin that the complexity prior then prefers. Both
    # stay closed on a labelled-ammonium run even when asked for.
    labelled_nh4 = any("^NH4" in str(a) for a in adducts)
    sign = "+" if polarity == "positive" else "-"
    requested = [str(a) for a in (getattr(cfg, "side_channels", None) or ())]
    side_refused = {}
    for a in requested:
        if a in adducts:
            continue                                  # already an analyte channel
        if a not in io_mascope.ADDUCT_TO_MECH:
            side_refused[a] = "no server mechanism"
        elif not a.rstrip(".").endswith(sign):
            side_refused[a] = f"not a {polarity} channel"
        elif labelled_nh4 and a in ("[M+NH4]+", "[M+Na]+"):
            side_refused[a] = "labelled-ammonium run"
    side = [a for a in requested if a not in adducts and a not in side_refused]
    if peaks is not None:
        # an offline sample resolves only the channels it registers: its own and
        # the side channels this run opens -- so an offline run (and a decoy arm,
        # which stamps the side channels its run recorded) opens what the run did
        io_mascope.register_offline_sample(
            sample_id, peaks,
            [io_mascope.ADDUCT_TO_MECH[a] for a in adducts + side if a in io_mascope.ADDUCT_TO_MECH],
            scoring=scoring)
    extra_channels = [a for a in side
                      if io_mascope.resolve_mechanism_ids(
                          client, [io_mascope.ADDUCT_TO_MECH[a]])]
    if requested or side_refused:
        _unres = [a for a in side if a not in extra_channels]
        log(f"[run] side channels: opened {extra_channels or 'none'}"
            + (f"; not resolved by the server {_unres}" if _unres else "")
            + "".join(f"; {a} skipped ({why})" for a, why in side_refused.items()))
    mech_names = [io_mascope.ADDUCT_TO_MECH[a]
                  for a in adducts + extra_channels
                  if a in io_mascope.ADDUCT_TO_MECH]
    mech_map = io_mascope.resolve_mechanism_ids(client, mech_names)
    # ABSTRACTION channels ([M-H]+, [M-CH3]+) have no deployment mechanism id, so
    # ADDUCT_TO_MECH cannot carry them and the loop above drops them. They ride
    # along as tagged tokens instead -- see io_mascope.LOCAL_MECH_PREFIX for why
    # this goes in mechanism_ids rather than a parallel argument.
    cfg.mechanism_ids = (list(mech_map.values())
                         + io_mascope.local_mechanism_tokens(adducts)) or None
    # the reagent halogen from the DECLARED channels, before the side channels
    # join: an [M+Br2]- side channel opened on a nitrate run does not make it a
    # bromide reagent
    reagent_halogen = evidence.channel_halogen(adducts)
    adducts = adducts + [a for a in extra_channels if a not in adducts]
    # ... and so is the composite (even-shift) de-blend's switch: it is the
    # halide reagents' test, so an opted-in [M+Br2]- side channel on a nitrate
    # run must not turn it on (a halogen side channel on a halide run changes
    # nothing: the declared channels already carry the halogen)
    has_halogen_adduct = any(h in str(a) for a in analyte_adducts
                             for h in ("Br", "Cl", "I"))
    # rough mass offset from the sample's own matches -> seeds the pre-calibration
    # pass-0 gate (the pass-1 self-calibration refines it). Without it a large
    # systematic offset is invisible to pass 0.
    prior = io_mascope.estimate_offset(raw)
    cfg.prior_offset = prior if prior is not None else 0.0
    log(f"[run] {len(led)} unique peaks; context={profile.label}; "
        f"polarity={polarity}; prior_offset={cfg.prior_offset:+.2f} ppm; "
        f"adducts={adducts}; mechanisms={sorted(mech_map)}"
        + (f"; label purity={purity:.3f}" if any("^" in str(a) for a in adducts) else ""))
    # What every candidate of this sample is scored at, resolved once and cached
    # for the passes. Logged because a run's assignments cannot be read without
    # it: the same envelope scores differently at 0.3 ppm and at 3.
    # a trend a previous run of this sample left (same process) is that run's;
    # this run fits its own (C42). A stand-in's inherited trend stays.
    io_mascope.reset_scoring_trend(sample_id)
    scoring = io_mascope.scoring_for_sample(client, sample_id, raw)
    scoring_snapshot = io_mascope.scoring_snapshot(client, sample_id, raw)
    log(f"[run] scoring {io_mascope.describe_scoring(scoring)}"
        f" ({scoring_snapshot['sigma_source']}, {scoring_snapshot['fitted_anchors']}"
        " anchors)")
    # C46: the tier pass keys its counting-detector floor on the class, and a
    # run whose peak table's signal-to-noise was replaced says so once
    cfg.instrument_type = scoring_snapshot.get("instrument_type")
    if scoring_snapshot.get("snr_source") == io_mascope.SNR_SOURCE_POISSON:
        _edge = scoring_snapshot.get("snr_edge")
        log(f"[run] signal-to-noise: the peak table's column does not track height "
            f"(Spearman {scoring_snapshot.get('snr_spearman')} over {scoring_snapshot.get('snr_n')} peaks)"
            f" -- lines judged at the counting-statistics SNR h/sqrt(h + edge^2), edge "
            f"{_edge if _edge is None else round(float(_edge), 4)} cps")
    _floor = tiers.tof_assign_floor(cfg)
    if _floor is not None:
        log(f"[run] tier floor (TOF): an M0 under {_floor:.3g} cps "
            f"({tiers.TOF_ASSIGN_FLOOR_X_EDGE:g}x the "
            + ("batch's typical" if cfg.noise_edge_batch_cps is not None else "file's own")
            + " detection edge) is Candidate")

    pre = isotopes.prescan(led)
    log(f"[run] prescan {pre.as_dict()}")

    # Label reagent-ion clusters BEFORE the passes so they are never assignment
    # candidates (e.g. [Br3]-, [Br+HBr]-, BrO- in a Br-CIMS sample; [urea_n+H]+
    # in a uronium sample).
    reagent = reagents.reagent_for_adducts(adducts)
    # The arbitration complexity prior is kept on a NEUTRAL halogen only -- a
    # molecular positive reagent (urea) puts no halogen in the neutral, so it has
    # no reagent_element (its [urea_n+H]+ clusters are still labelled via
    # `reagent`). This also keeps _prefer_adduct_reading / the di-bromide iso-pair
    # logic inert in positive mode.
    cfg.reagent_element = reagent if reagent in ("Br", "Cl", "I") else None
    if reagent:
        n_reag = reagents.label_reagents(led, reagent, ppm=12.0)
        log(f"[run] pre-labeled {n_reag} reagent-cluster peaks ({reagent})")

    # C42: the pass-1 commits are scored against the sample's ONE constant offset;
    # if calibrate() then accepts a 1/mz mass trend (fitted on the pattern-only,
    # isotope-tested backbone), the file is re-run from pass 0 -- from the state
    # the stages started from -- with every candidate line judged against the
    # trend's centre at its own m/z, until the trend it is scored at and its own
    # calibration's agree (_trend_step; at most MAX_TREND_RERUNS re-runs). A
    # stand-in that inherited its sample's trend (a decoy arm) is scored at it
    # from the start and never re-runs; with score_at_trend off a run fits none.
    led0, cfg0 = led.copy(deep=True), copy.deepcopy(cfg)
    inherited = io_mascope.scoring_trend(sample_id) is not None
    local = io_mascope._local_scoring_enabled()
    reruns = 0
    while True:
        st = _RunState(
            client=client, sample_id=sample_id, led=led, profile=profile, pre=pre,
            cfg=cfg, adducts=adducts, reagent=reagent, has_halogen=has_halogen_adduct,
            do_pass2=do_pass2, do_pass3=do_pass3, do_pass4=do_pass4, do_pass5=do_pass5,
            do_pass_certified=do_pass_certified,
            reflists_active=reflists_active, ts_peaks=ts_peaks,
            label_isotope=label_isotope, label_max=label_max, log=log,
            checkpoint_dir=checkpoint_dir, reagent_n_relabel=reagent_n_relabel,
            corroborate=set(corroborate or ()), resolving_power=width_model,
            reagent_halogen=reagent_halogen,
            reagent_profile=(getattr(reagent_profile, "name", None) or reagent_profile
                             or _profile_name_for(analyte_adducts)),
            reflists_context=reflists_context)
        restart = False
        for stg in _STAGES:
            if not stg.when(st):
                continue
            res = _safe(st, stg.name, (lambda s=stg: s.fn(st))) if stg.safe else stg.fn(st)
            if stg.store:
                st.summaries[stg.name] = res
            if stg.name == "calibrate" and not inherited and cfg.score_at_trend:
                fitted = (masscal.MassTrend(cfg.cal_a, cfg.cal_b, cfg.cal_sigma_trend,
                                            cfg.cal_trend_n or 0, cfg.cal_mz_lo, cfg.cal_mz_hi)
                          if cfg.cal_b is not None else None)
                current = io_mascope.scoring_trend(sample_id)
                step = _trend_step(current, fitted, local=local, reruns=reruns)
                if step == "rerun":
                    io_mascope.set_scoring_trend(sample_id, fitted)
                    log(f"[run] scoring centre -> the mass trend ppm = {fitted.a:+.3f} "
                        f"{fitted.b:+.3f}*1000/mz (m/z {fitted.mz_lo:.0f}-{fitted.mz_hi:.0f}); "
                        f"re-running from pass 0 at the per-line centre (re-run {reruns + 1})")
                    reruns += 1
                    restart = True
                    break
                if step == "keep_gates":
                    cfg.cal_a, cfg.cal_b = current.a, current.b
                    cfg.cal_sigma_trend, cfg.cal_trend_n = current.sigma, current.n
                    cfg.cal_mz_lo, cfg.cal_mz_hi = current.mz_lo, current.mz_hi
                    log("[run] this pass's calibration rejects the mass trend the file is "
                        "scored at; the gates keep the scoring trend")
                elif step == "network":
                    log("[run] mass trend accepted, but the network scorer judges one "
                        "constant offset: scored at it, the trend gates only")
                elif current is not None and fitted is not None:
                    log(f"[run] scoring trend and calibration agree within "
                        f"{masscal.trend_shift(current, fitted):.3f} ppm"
                        + ("" if masscal.trend_shift(current, fitted) <= TREND_AGREE_PPM
                           else f" (re-run limit {MAX_TREND_RERUNS} reached)"))
        if not restart:
            break
        led, cfg = led0.copy(deep=True), copy.deepcopy(cfg0)
    # the FINAL pass's state: its degeneracy calibration is the one the stats persist
    run_state = st
    led, summaries, plaus_audit = st.led, st.summaries, st.plaus_audit
    # the record of what the candidates were scored at, the trend included
    scoring_snapshot = io_mascope.scoring_snapshot(client, sample_id, raw)
    tc = led.loc[led["role"] == ledger.ROLE_M0, "tier"].value_counts().to_dict()
    log(f"[run] tiers {tc}")

    problems = ledger.validate(led)
    if problems:
        log(f"[run] LEDGER VALIDATION PROBLEMS: {problems}")
    st = ledger.stats(led)
    st["noise_edge_cps"] = cfg.noise_edge_cps
    # C46: the footing of the tier pass's counting-detector floor, the class it
    # keyed on, the floor in force (None off a TOF) and what SNR the lines were judged at
    st["noise_edge_batch_cps"] = getattr(cfg, "noise_edge_batch_cps", None)
    st["instrument_type"] = getattr(cfg, "instrument_type", None)
    st["tof_assign_floor_cps"] = tiers.tof_assign_floor(cfg)
    st["snr_source"] = scoring_snapshot.get("snr_source")
    st["height_gate_cps"] = cfg.height_cutoff     # RESOLVED gate (the knob is cfg.height_cutoff_cps)
    # the multiple the gate was resolved FROM (profile-supplied or the package
    # default) -- the gate in cps alone cannot be read back without it.
    st["height_cutoff_x_edge"] = cfg.height_cutoff_x_edge_resolved
    # the width model the resolvability stage used (None = not stamped) and its class counts
    st["resolution"] = width_model.as_dict() if width_model is not None else None
    st["resolvability"] = (summaries.get("resolvability") or {}).get("counts")
    # the degeneracy stage's own calibration (mu, sigma) ppm -- the step-1 window
    # of the evidence level, per file and pooled; null = uncalibrated (no key
    # when the stage did not run)
    if run_state.degeneracy_cal != "absent":
        cal = run_state.degeneracy_cal
        st["degeneracy_cal"] = None if cal is None else {"mu": float(cal[0]), "sigma": float(cal[1])}
    # the reagent halogen of the declared channels (C43) this file's evidence
    # read: a batch's merge vote computes the file's class in the parent with it
    st["reagent_halogen"] = reagent_halogen
    # the side channels this file OPENED (asked for and resolved by the server;
    # [] = none): two runs of one sample that differ here differ for this reason
    st["side_channels"] = list(extra_channels)
    st["admitted"] ={"height": adm["height"], "occurrence": adm["occurrence"],
                      "rejected": adm["rejected"]}
    log(f"[run] stats {json.dumps(st)}")
    return {"ledger": led, "stats": st, "summaries": summaries,
            "prescan": pre.as_dict(), "problems": problems,
            "plausibility_audit": plaus_audit,
            "module_versions": module_versions(),
            "module_hashes": _module_hashes(), "context": profile.label,
            "reflists_active": reflist_versions,     # [(id, data_version), ...]
            # What this sample's candidates were scored at. A run's assignments
            # cannot be read without it: the same envelope scores differently at
            # 0.3 ppm and at 3, and a published run carries it into the store.
            "pattern_scoring": scoring_snapshot,
            "sample_id": sample_id}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mascope multi-pass peak assignment")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--context", default="ambient-air")
    ap.add_argument("--ppm", type=float, default=1.0)
    ap.add_argument("--search-ppm", type=float, default=3.0)
    ap.add_argument("--height-cutoff", type=float, default=None,
                    help="absolute cps override; default = a multiple of the "
                         "sample's own noise edge")
    ap.add_argument("--height-cutoff-x-edge", type=float, default=None,
                    help="height gate as a multiple of the sample's own noise edge "
                         "(1.0 = keep every picked peak but the bottom 1%%); "
                         "default: the reagent profile's own multiple, else the "
                         "package default; ignored when --height-cutoff is given")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-pass2", action="store_true")
    ap.add_argument("--no-pass3", action="store_true")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args(argv)

    # no --reagent here (this entry point takes a context, not a profile), so the
    # resolution is the flag if given, else the package default.
    from peaky.chem import profiles as PR

    cfg = passes.PassConfig(ppm=args.ppm, search_ppm=args.search_ppm,
                            height_cutoff_cps=args.height_cutoff)
    PR.apply_height_cutoff_x_edge(cfg, None, explicit=args.height_cutoff_x_edge,
                                  log=print)
    PR.apply_ion_only_channels(cfg, None)        # no profile here: the bucket stays off
    PR.apply_side_channels(cfg, None)            # ... and so does every side channel
    out = run(args.sample_id, args.context, cfg=cfg, use_cache=not args.no_cache,
              do_pass2=not args.no_pass2, do_pass3=not args.no_pass3)
    # report.py will own file outputs; for now write the ledger + manifest
    from pathlib import Path
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    out["ledger"].to_csv(od / f"{args.sample_id}_ledger.csv", index=False)
    manifest = {k: v for k, v in out.items() if k != "ledger"}
    (od / f"{args.sample_id}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {args.sample_id}_ledger.csv + manifest")


if __name__ == "__main__":
    main()
