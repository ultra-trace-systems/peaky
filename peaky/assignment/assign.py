"""Top-level orchestrator + CLI.

Wires the spine (chemistry, contexts, ledger), the oracle (io_mascope), the
prescan (isotopes), and the three-pass director (passes) into one run and
records a reproducibility manifest with every locked module version.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from peaky.chem import chemistry
from peaky.assignment import cleanup
from peaky.chem import contexts
from peaky.assignment import degeneracy
from peaky.io import io_mascope
from peaky.chem import isotopes
from peaky.assignment import labeled
from peaky.assignment import ladders
from peaky.assignment import ledger
from peaky.assignment import passes
from peaky.assignment import plausibility
from peaky.chem import reagents
from peaky.assignment import reflists
from peaky.assignment import residual
from peaky.assignment import series_gka
from peaky.assignment import siloxane
from peaky.assignment import tiers
from peaky.batch import timeseries

__version__ = "0.4.0"

MODULE_VERSIONS = {
    "assign": __version__,
    "chemistry": chemistry.__version__,
    "contexts": contexts.__version__,
    "ledger": ledger.__version__,
    "io_mascope": io_mascope.__version__,
    "isotopes": isotopes.__version__,
    "series_gka": series_gka.__version__,
    "reagents": reagents.__version__,
    "passes": passes.__version__,
    "residual": residual.__version__,
    "ladders": ladders.__version__,
    "tiers": tiers.__version__,
    "degeneracy": degeneracy.__version__,
    "cleanup": cleanup.__version__,
    "plausibility": plausibility.__version__,
    "siloxane": siloxane.__version__,
    "timeseries": timeseries.__version__,
}


def _degen_summary(led) -> dict:
    """Compact manifest summary of the degeneracy audit."""
    m0 = led[led["role"] == ledger.ROLE_M0]
    d = m0["degeneracy_density"].dropna()
    if not len(d):
        return {"measured": 0}
    d = d.astype(int)
    return {"measured": int(len(d)), "degenerate_ge2": int((d >= 2).sum()),
            "max_density": int(d.max())}


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


# The assignment pipeline AS DATA -- read top to bottom to see exactly what runs,
# in what order, under what condition. `safe` wraps a stage so a failure can't lose
# prior work; `store` keeps its summary. Authoritative stage table: ARCHITECTURE.md §4.
_STAGES = [
    _Stage("pass0", lambda st: passes.run_pass0_known(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.adducts, log=st.log)),
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
    _Stage("pass3", lambda st: passes.run_pass3(
        st.client, st.sample_id, st.led, st.profile, st.pre, st.cfg, st.adducts, log=st.log),
           when=lambda st: st.do_pass3),
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
    # honest mass-degeneracy measurement -- MUST precede tiers (the tier engine reads it).
    _Stage("degeneracy", lambda st: _degen_summary(
        degeneracy.apply_degeneracy(st.led, context=st.profile.label, log=st.log))),
    # report tier, then the post-tier de-risking demotes (each gets the last word).
    _Stage("tiers", lambda st: tiers.apply_tiers(st.led), safe=False, store=False),
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
    _Stage("relabel_reagent_n",
           lambda st: cleanup.relabel_reagent_n_adducts(st.led, log=st.log),
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
    _Stage("plausibility", lambda st: plausibility.demote_implausible(
        st.led, audit=st.plaus_audit, log=st.log), safe=False),
    # rescue-verify the still-unexplained residual against active reference peaklists.
    _Stage("reflist_rescue", lambda st: reflists.rescue_unexplained_by_reflist(
        st.client, st.sample_id, st.led, st.profile, st.cfg, st.reflists_active,
        st.adducts, log=st.log), when=lambda st: bool(st.reflists_active)),
    # final envelope sweep: an M0 committed AFTER the three earlier sweeps (a pass-8
    # reflist rescue, or a parent whose earlier assignment was cleared and re-won,
    # orphaning its children) still owes its satellites -- without this its bright
    # 13C line sits in the residual (the NBBS urea-adduct M+1 was the single
    # brightest "unexplained" peak of a positive urea-CIMS ambient batch run).
    _Stage("iso_env_final",
           lambda st: passes.complete_isotope_envelopes(st.led, st.cfg, log=st.log)),
    _Stage("timeseries", _stage_timeseries,
           when=lambda st: st.ts_peaks is not None and len(st.ts_peaks)),
]


def run(sample_id: str, context: str = "ambient-air", *,
        cfg: passes.PassConfig | None = None, use_cache: bool = True,
        do_pass2: bool = True, do_pass3: bool = True, do_pass4: bool = True,
        do_pass5: bool = True, do_pass_certified: bool = True,
        ts_peaks=None, adducts=None, reflists_active=None,
        label_isotope=None, label_max=2,
        log=print, checkpoint_dir=None) -> dict:
    cfg = cfg or passes.PassConfig()
    # reference-list selection prior: a candidate neutral on an active reference
    # peaklist wins a near-tie over a mass coincidence (arbitrate reads this set).
    if reflists_active:
        cfg.reflist_formulas = frozenset().union(
            *(set(rl.formulas) for rl in reflists_active)) or frozenset()
    profile = contexts.get_context(context)
    client = io_mascope.connect()

    raw = io_mascope.fetch_peaks(client, sample_id, use_cache=use_cache)
    led = ledger.new_ledger(raw)
    # Adducts are normally detected from the sample's own server matches (the
    # SKILL design rule for mixed-reagent datasets). But a batch with a KNOWN
    # reagent can pass `adducts=` to force the analyte channels: per-sample match
    # detection is unreliable when a sample has few/no server matches (a positive
    # sample with no urea-channel match then falls back to [M-H]- and the whole
    # spectrum is mis-assigned in the wrong polarity). The explicit list wins.
    adducts = list(adducts) if adducts else io_mascope.detect_adducts(raw)
    # Polarity is read from the (detected or forced) adducts (cation forms end "+").
    polarity = "positive" if any(str(a).rstrip().endswith("+") for a in adducts) \
        else "negative"
    # opportunistic extra channels, scored only if the server has the mechanism
    # registered (auto-selection covers only the sample's own channels, so the
    # ids must be passed explicitly). Negative (halide-CIMS):
    #   +CO3-  carbonate adducts of aldehydes from lingering air ions
    #   +Br2-  di-bromide reagent-cluster adducts of analytes -- the n_Br=2
    #          "C/H lattice" residual is biogenic SOA seen this way (2026-06-12)
    # NB: +Br3- is intentionally NOT a blanket scoring channel -- adding it lost
    # 40 base M0s (incl. TFA) for 0 gains (heavier per-formula scoring times out
    # batches; Br3- is the dominant reagent ion). Positive (urea-CIMS): the
    # alkali / ammonium adducts the source also produces.
    opportunistic = (["[M+Na]+", "[M+NH4]+"] if polarity == "positive"
                     else ["[M+CO3]-", "[M+Br2]-"])
    extra_channels = [a for a in opportunistic
                      if io_mascope.resolve_mechanism_ids(
                          client, [io_mascope.ADDUCT_TO_MECH[a]])]
    mech_names = [io_mascope.ADDUCT_TO_MECH[a]
                  for a in adducts + extra_channels
                  if a in io_mascope.ADDUCT_TO_MECH]
    mech_map = io_mascope.resolve_mechanism_ids(client, mech_names)
    cfg.mechanism_ids = list(mech_map.values()) or None
    adducts = adducts + [a for a in extra_channels if a not in adducts]
    has_halogen_adduct = any(h in str(a) for a in adducts
                             for h in ("Br", "Cl", "I"))
    # rough mass offset from the sample's own matches -> seeds the pre-calibration
    # pass-0 gate (the pass-1 self-calibration refines it). Without it a large
    # systematic offset is invisible to pass 0.
    prior = io_mascope.estimate_offset(raw)
    cfg.prior_offset = prior if prior is not None else 0.0
    log(f"[run] {len(led)} unique peaks; context={profile.label}; "
        f"polarity={polarity}; prior_offset={cfg.prior_offset:+.2f} ppm; "
        f"adducts={adducts}; mechanisms={sorted(mech_map)}")

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

    st = _RunState(
        client=client, sample_id=sample_id, led=led, profile=profile, pre=pre,
        cfg=cfg, adducts=adducts, reagent=reagent, has_halogen=has_halogen_adduct,
        do_pass2=do_pass2, do_pass3=do_pass3, do_pass4=do_pass4, do_pass5=do_pass5,
        do_pass_certified=do_pass_certified,
        reflists_active=reflists_active, ts_peaks=ts_peaks,
        label_isotope=label_isotope, label_max=label_max, log=log,
        checkpoint_dir=checkpoint_dir)
    for stg in _STAGES:
        if not stg.when(st):
            continue
        res = _safe(st, stg.name, (lambda s=stg: s.fn(st))) if stg.safe else stg.fn(st)
        if stg.store:
            st.summaries[stg.name] = res
    led, summaries, plaus_audit = st.led, st.summaries, st.plaus_audit
    tc = led.loc[led["role"] == ledger.ROLE_M0, "tier"].value_counts().to_dict()
    log(f"[run] tiers {tc}")

    problems = ledger.validate(led)
    if problems:
        log(f"[run] LEDGER VALIDATION PROBLEMS: {problems}")
    st = ledger.stats(led)
    log(f"[run] stats {json.dumps(st)}")
    return {"ledger": led, "stats": st, "summaries": summaries,
            "prescan": pre.as_dict(), "problems": problems,
            "plausibility_audit": plaus_audit,
            "module_versions": MODULE_VERSIONS,
            "module_hashes": _module_hashes(), "context": profile.label,
            "sample_id": sample_id}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mascope multi-pass peak assignment")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--context", default="ambient-air")
    ap.add_argument("--ppm", type=float, default=1.0)
    ap.add_argument("--search-ppm", type=float, default=3.0)
    ap.add_argument("--height-cutoff", type=float, default=100.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-pass2", action="store_true")
    ap.add_argument("--no-pass3", action="store_true")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args(argv)

    cfg = passes.PassConfig(ppm=args.ppm, search_ppm=args.search_ppm,
                            height_cutoff=args.height_cutoff)
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
