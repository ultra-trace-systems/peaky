"""Offline tests for assignment/admission.py -- the persistence-OR-brightness
gate. Decoy control: recurrent-weak bins are admitted by occurrence, a cloud of
transient noise bins is not, and mass-shifted decoys admit ~nothing; with
occurrence_min=0 (or no batch context) the gate is exactly the height gate.
Run: python3 tests/test_admission.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.assignment import admission as ADM   # noqa: E402
from peaky.assignment import passes as P        # noqa: E402
from peaky.assignment import ledger as L        # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# synthetic batch: 100 spectra. 6 recurrent-weak ions (2 cps, in 95 % of spectra),
# 3 bright ions (500 cps, in half the spectra), and a cloud of 400 transient noise
# bins (1-3 cps, each in 1-3 spectra at a random m/z). Per-spectrum m/z jitter 2 ppm.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N = 100
RECUR = [264.0361, 278.0519, 294.0832, 310.0781, 326.0729, 342.0678]
BRIGHT = [201.0000, 250.5000, 333.3333]
rows = []
def jitter(mz):
    return mz * (1 + rng.normal(0, 2e-6))
for s in range(N):
    sid = f"s{s:03d}"
    for mz in RECUR:
        if rng.random() < 0.95:
            rows.append((sid, jitter(mz), 2.0 + rng.random()))
    for mz in BRIGHT:
        if rng.random() < 0.5:
            rows.append((sid, jitter(mz), 500.0 + 50 * rng.random()))
noise_mz = rng.uniform(200, 400, 400)
for mz in noise_mz:
    for s in rng.choice(N, size=rng.integers(1, 4), replace=False):
        rows.append((f"s{s:03d}", jitter(mz), 1.0 + 2 * rng.random()))
ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])

tab = ADM.bin_occurrence(ts)
check("bin_occurrence: one row per bin with mz/occurrence/n_samples + attrs",
      {"mz", "occurrence", "n_samples"} <= set(tab.columns) and tab.attrs["n_samples"] == N
      and tab.attrs["tol_ppm"] > 0, (list(tab.columns), tab.attrs))
occ_recur = ADM.lookup_occurrence(RECUR, tab)
occ_bright = ADM.lookup_occurrence(BRIGHT, tab)
check("recurrent-weak ions bin at ~0.95 occurrence", np.all(occ_recur > 0.85), occ_recur)
check("bright half-time ions bin at ~0.5", np.all((occ_bright > 0.35) & (occ_bright < 0.65)), occ_bright)
occ_noise = ADM.lookup_occurrence(noise_mz, tab)
check("transient noise bins stay far below 0.8 (max < 0.2)", np.nanmax(occ_noise) < 0.2, np.nanmax(occ_noise))
check("lookup: NaN outside the tolerance",
      np.isnan(ADM.lookup_occurrence([264.0361 * (1 + 50e-6), 150.0], tab)).all())
check("lookup: no table -> all NaN", np.isnan(ADM.lookup_occurrence([1.0, 2.0], None)).all())

# ---------------------------------------------------------------------------
# a single spectrum's ledger through the gate with an ABSOLUTE 100 cps height
# cutoff (the old behaviour) + occurrence 0.8: bright by height, recurrent-weak by
# persistence, noise rejected
# ---------------------------------------------------------------------------
one = ts[ts.sample_item_id == "s000"].copy()
one["peak_id"] = [f"p{i}" for i in range(len(one))]
led = L.new_ledger(one[["peak_id", "mz", "height"]])
cfg = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.8)
counts = ADM.stamp_admission(led, cfg, tab)
check("stamp_admission: occurrence + admitted_by columns added",
      {"occurrence", "admitted_by"} <= set(led.columns), list(led.columns)[-3:])
is_recur = led["mz"].apply(lambda m: any(abs(m - r) / r * 1e6 < 10 for r in RECUR))
is_bright = led["height"] >= 100
is_noise = ~is_recur & ~is_bright
check("bright peaks admitted by height", (led.loc[is_bright, "admitted_by"] == "height").all())
check("recurrent-weak peaks admitted by OCCURRENCE (below the height gate)",
      is_recur.sum() >= 4 and (led.loc[is_recur, "admitted_by"] == "occurrence").all(),
      led.loc[is_recur, ["mz", "height", "occurrence", "admitted_by"]].to_dict("records"))
check("transient noise peaks NOT admitted (specificity)",
      is_noise.sum() >= 3 and (led.loc[is_noise, "admitted_by"] == "").all(),
      led.loc[is_noise, ["mz", "height", "occurrence", "admitted_by"]].head().to_dict("records"))
check("counts agree with the columns",
      counts["height"] == int(is_bright.sum()) and counts["occurrence"] == int(is_recur.sum())
      and counts["rejected"] == int(is_noise.sum()), counts)
mask = ADM.admissible(led, cfg)
check("admissible == admitted_by != ''", (mask == (led["admitted_by"] != "")).all())

# decoy control: shift the SAME ledger by +0.35 / +0.50 / -0.35 Da -> nothing
# recurrent should be admitted by occurrence any more
for off in (0.35, 0.50, -0.35):
    d = led.copy(); d["mz"] = d["mz"] + off
    ADM.stamp_admission(d, cfg, tab)
    n_occ = int((d["admitted_by"] == "occurrence").sum())
    check(f"decoy {off:+.2f} Da: occurrence admits ~nothing ({n_occ})", n_occ <= 1, n_occ)

# ---------------------------------------------------------------------------
# no-regression: occurrence_min=0, or no table / no column -> exactly the height gate
# ---------------------------------------------------------------------------
cfg0 = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.0)
ADM.stamp_admission(led, cfg0, tab)
check("occurrence_min=0: gate == height gate, nothing by occurrence",
      (ADM.admissible(led, cfg0) == (led["height"] >= 100)).all()
      and (led["admitted_by"] != "occurrence").all())
led_nb = L.new_ledger(one[["peak_id", "mz", "height"]])            # no batch context
c = ADM.stamp_admission(led_nb, cfg, None)
check("no table: occurrence NaN, admitted_by height-only, n_peaks 0",
      led_nb["occurrence"].isna().all() and (ADM.admissible(led_nb, cfg) == (led_nb["height"] >= 100)).all()
      and c["n_peaks"] == 0)
plain = one[["peak_id", "mz", "height"]].copy()
check("admissible on a ledger without an occurrence column = height gate",
      (ADM.admissible(plain, cfg) == (plain["height"] >= 100)).all())
check("admissible: NaN heights count as 0", not ADM.admissible(pd.DataFrame({"height": [np.nan]}), cfg0).iloc[0])

# edge-relative default: x_edge 1.0 with a stamped edge -> the height path alone
# already admits nearly everything, the persistence path adds only sub-edge peaks
cfg_e = P.PassConfig(occurrence_min=0.8); cfg_e.noise_edge_cps = 1.5
ADM.stamp_admission(led, cfg_e, tab)
check("edge-relative gate: height_cutoff = 1x edge, recurrent 2-cps ions admitted by height",
      abs(cfg_e.height_cutoff - 1.5) < 1e-9 and (led.loc[is_recur, "admitted_by"] == "height").all())
cfg_e.height_cutoff_x_edge = 5.0
ADM.stamp_admission(led, cfg_e, tab)
check("x_edge 5: the same ions fall to the persistence path, noise below it is rejected",
      (led.loc[is_recur, "admitted_by"] == "occurrence").all()
      and (led.loc[is_noise & (led["height"] < 7.5), "admitted_by"] == "").all())

# ---------------------------------------------------------------------------
# the derived threshold: Otsu's split of the (bimodal) bin-occurrence distribution
# ---------------------------------------------------------------------------
auto = tab.attrs["auto_threshold"]
check("bin_occurrence stores an auto threshold within the clamp",
      auto is not None and ADM.AUTO_MIN <= auto <= ADM.AUTO_MAX, auto)
check("auto threshold separates the recurrent ions from the noise cloud",
      np.all(occ_recur >= auto) and np.nanmax(occ_noise) < auto, (auto, occ_recur.min(), np.nanmax(occ_noise)))
cfg_auto = P.PassConfig(height_cutoff_cps=100.0)          # occurrence_min 'auto' (default)
check("PassConfig default occurrence_min is 'auto'", cfg_auto.occurrence_min == "auto")
check("resolve_threshold('auto') == the table's auto threshold",
      ADM.resolve_threshold(cfg_auto, tab) == auto)
check("resolve_threshold(number) passes the number through",
      ADM.resolve_threshold(P.PassConfig(occurrence_min=0.6), tab) == 0.6)
check("resolve_threshold(0) / no table -> None (path off)",
      ADM.resolve_threshold(P.PassConfig(occurrence_min=0), tab) is None
      and ADM.resolve_threshold(cfg_auto, None) is None)
try:
    ADM.resolve_threshold(P.PassConfig(occurrence_min="sometimes"), tab)
    check("resolve_threshold rejects an unknown string", False)
except ValueError:
    check("resolve_threshold rejects an unknown string", True)
ADM.stamp_admission(led, cfg_auto, tab)
check("stamp_admission with 'auto': recurrent-weak by occurrence, threshold stored on cfg",
      (led.loc[is_recur, "admitted_by"] == "occurrence").all() and cfg_auto.occurrence_threshold == auto)
# too few spectra -> occurrence is coarse -> the path is off
small = ADM.bin_occurrence(ts[ts.sample_item_id.isin([f"s{i:03d}" for i in range(5)])])
check("fewer than MIN_SPECTRA spectra -> path off (resolve None)",
      small.attrs["n_samples"] == 5 and ADM.resolve_threshold(cfg_auto, small) is None)
check("otsu_threshold on a degenerate input is None", ADM.otsu_threshold([0.5]) is None)
o = ADM.otsu_threshold(np.r_[np.zeros(80) + 0.05, np.ones(20) * 0.95])
check("otsu on a clean two-cluster distribution splits between them", 0.1 < o < 0.9, o)

check("otsu on a clean two-cluster distribution lands mid-gap",
      0.4 < ADM.otsu_threshold(np.r_[np.zeros(80) + 0.05, np.ones(20) * 0.95]) < 0.6,
      ADM.otsu_threshold(np.r_[np.zeros(80) + 0.05, np.ones(20) * 0.95]))

# ---------------------------------------------------------------------------
# THE CLAMP. Otsu always returns a split, even for a distribution that has no
# two modes to separate -- it then cuts inside the single mode and the "derived"
# threshold is meaningless. [AUTO_MIN, AUTO_MAX] is what keeps such a batch from
# admitting (unimodal-low) or rejecting (unimodal-high) essentially everything.
# Both raw splits sit far OUTSIDE the clamp, so deleting the clamp fails here.
# ---------------------------------------------------------------------------
_r = np.random.default_rng(3)
uni_low = np.clip(_r.normal(0.05, 0.02, 400), 0, 1)     # nothing persists
uni_high = np.clip(_r.normal(0.95, 0.02, 400), 0, 1)    # everything persists
raw_low, raw_high = ADM.otsu_threshold(uni_low), ADM.otsu_threshold(uni_high)
check("unimodal LOW: the raw Otsu split is below the clamp, auto_threshold returns AUTO_MIN",
      raw_low < ADM.AUTO_MIN and ADM.auto_threshold(uni_low) == ADM.AUTO_MIN,
      (raw_low, ADM.auto_threshold(uni_low)))
check("unimodal HIGH: the raw Otsu split is above the clamp, auto_threshold returns AUTO_MAX",
      raw_high > ADM.AUTO_MAX and ADM.auto_threshold(uni_high) == ADM.AUTO_MAX,
      (raw_high, ADM.auto_threshold(uni_high)))
check("the clamp is what bounds it: every auto threshold lands inside [AUTO_MIN, AUTO_MAX]",
      all(ADM.AUTO_MIN <= ADM.auto_threshold(v) <= ADM.AUTO_MAX
          for v in (uni_low, uni_high, tab["occurrence"].to_numpy())))
# all-equal: there is no split to make. otsu returns None and the run must treat
# that as PATH OFF, not as a 0.0 threshold (which would admit every peak).
flat = np.full(200, 0.5)
check("all-equal occurrences: otsu None -> auto None -> resolve None (path off, not 0.0)",
      ADM.otsu_threshold(flat) is None and ADM.auto_threshold(flat) is None,
      (ADM.otsu_threshold(flat), ADM.auto_threshold(flat)))
_flat_tab = pd.DataFrame({"mz": np.linspace(200, 400, 200), "occurrence": flat,
                          "n_samples": np.full(200, 50)})
_flat_tab.attrs.update(tol_ppm=6.0, n_samples=50, auto_threshold=ADM.auto_threshold(flat))
check("...and a whole batch of all-equal bins resolves to None -> nothing admitted by persistence",
      ADM.resolve_threshold(cfg_auto, _flat_tab) is None, ADM.resolve_threshold(cfg_auto, _flat_tab))
_fl = L.new_ledger(pd.DataFrame({"peak_id": ["weak", "bright"], "mz": [250.0, 300.0],
                                 "height": [2.0, 500.0]}))
_fc = ADM.stamp_admission(_fl, P.PassConfig(height_cutoff_cps=100.0), _flat_tab)
_fby = _fl.set_index("peak_id")["admitted_by"]
check("...stamping on that batch admits by height only (the 0.5 bins admit nobody)",
      _fc["occurrence"] == 0 and _fc["occurrence_threshold"] is None
      and _fby["bright"] == "height" and _fby["weak"] == "", (_fc, _fby.to_dict()))

# ---------------------------------------------------------------------------
# WHY the path is off. Four different situations resolve to no threshold, and
# they call for opposite responses (raise the knob / pass --ts-batch / assign a
# longer batch / accept that this batch has no split). A log line saying only
# "path off" -- or worse, "X spectra < 10, OR occurrence_min=..." -- is
# unactionable, so every reporting caller goes through why_off().
# ---------------------------------------------------------------------------
check("why_off: a zero knob names the knob (not the table)",
      "occurrence_min" in ADM.why_off(P.PassConfig(occurrence_min=0), tab)
      and "knob" in ADM.why_off(P.PassConfig(occurrence_min=0), tab),
      ADM.why_off(P.PassConfig(occurrence_min=0), tab))
check("why_off: a zero knob is named even when there IS no table (the knob wins)",
      "occurrence_min" in ADM.why_off(P.PassConfig(occurrence_min=0), None),
      ADM.why_off(P.PassConfig(occurrence_min=0), None))
check("why_off: no table names the missing batch context (--ts-batch)",
      "--ts-batch" in ADM.why_off(cfg_auto, None), ADM.why_off(cfg_auto, None))
check("why_off: too few spectra names the count and the minimum",
      f"{ADM.MIN_SPECTRA}" in ADM.why_off(cfg_auto, small) and "5 spectra" in ADM.why_off(cfg_auto, small),
      ADM.why_off(cfg_auto, small))
check("why_off: a table with no split says so -- NOT 'too few spectra'",
      "no split" in ADM.why_off(cfg_auto, _flat_tab)
      and "spectra, fewer than" not in ADM.why_off(cfg_auto, _flat_tab),
      ADM.why_off(cfg_auto, _flat_tab))
check("why_off: the four reasons are four DIFFERENT strings",
      len({ADM.why_off(P.PassConfig(occurrence_min=0), tab), ADM.why_off(cfg_auto, None),
           ADM.why_off(cfg_auto, small), ADM.why_off(cfg_auto, _flat_tab)}) == 4)
check("why_off: empty when the path is ON (there is nothing to explain)",
      ADM.why_off(cfg_auto, tab) == "" and ADM.why_off(P.PassConfig(occurrence_min=0.6), tab) == "",
      (ADM.why_off(cfg_auto, tab), ADM.why_off(P.PassConfig(occurrence_min=0.6), tab)))
check("why_off is non-empty exactly when resolve_threshold is None",
      all((ADM.why_off(c, t) != "") == (ADM.resolve_threshold(c, t) is None)
          for c, t in ((cfg_auto, tab), (cfg_auto, None), (cfg_auto, small),
                       (cfg_auto, _flat_tab), (P.PassConfig(occurrence_min=0), tab),
                       (P.PassConfig(occurrence_min=0.6), tab))))

# ---------------------------------------------------------------------------
# THE BOUNDARY. The gate is `occurrence >= threshold`, inclusive: a bin sitting
# EXACTLY on the derived threshold is admitted. (A `>` gate passes every other
# test in this file; only this one sees the difference.)
# ---------------------------------------------------------------------------
THR = 0.44
bl = L.new_ledger(pd.DataFrame({"peak_id": ["on", "under", "over"],
                                "mz": [250.0, 260.0, 270.0],
                                "height": [2.0, 2.0, 2.0]}))     # all below the height gate
_btab = pd.DataFrame({"mz": [250.0, 260.0, 270.0],
                      "occurrence": [THR, np.nextafter(THR, 0.0), np.nextafter(THR, 1.0)],
                      "n_samples": [44, 43, 45]})
_btab.attrs.update(tol_ppm=6.0, n_samples=100)
cfg_b = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=THR)
ADM.stamp_admission(bl, cfg_b, _btab)
_by = bl.set_index("peak_id")["admitted_by"]
check("occurrence EXACTLY == threshold is admitted (>= not >)", _by["on"] == "occurrence", _by.to_dict())
check("one ulp under the threshold is not admitted", _by["under"] == "", _by.to_dict())
check("one ulp over the threshold is admitted", _by["over"] == "occurrence", _by.to_dict())
_am = ADM.admissible(bl, cfg_b)
check("admissible() agrees with the stamp at the boundary",
      (_am == (bl["admitted_by"] != "")).all() and bool(_am[bl["peak_id"] == "on"].iloc[0]))

# ...and they must still agree when the resolver switched the path OFF while the
# knob is a NUMBER. `admitted_by == ''` is DEFINED as "not admissible at the
# gated sites", so the gate may not fall back to the raw `occurrence_min` the
# resolver already rejected -- a 5-spectrum batch would otherwise re-admit every
# persistent-weak peak at all of them, silently bypassing the MIN_SPECTRA guard.
_ftab = pd.DataFrame({"mz": [250.0, 260.0], "occurrence": [0.93, 0.93], "n_samples": [5, 5]})
_ftab.attrs.update(tol_ppm=6.0, n_samples=5)               # < ADM.MIN_SPECTRA
cfg_f = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.6)
fl = L.new_ledger(pd.DataFrame({"peak_id": ["weak", "bright"], "mz": [250.0, 260.0],
                                "height": [2.0, 500.0]}))
_fc = ADM.stamp_admission(fl, cfg_f, _ftab)
check("too few spectra: the resolver gives no threshold even for a numeric knob",
      cfg_f.occurrence_threshold is None and cfg_f.occurrence_resolved is True
      and "spectra" in ADM.why_off(cfg_f, _ftab), (cfg_f.occurrence_threshold, _fc))
_fm = pd.Series(ADM.admissible(fl, cfg_f).to_numpy(), index=fl["peak_id"])
check("resolved-OFF path stays off in admissible(): the persistent weak peak is NOT eligible",
      (ADM.admissible(fl, cfg_f).to_numpy() == (fl["admitted_by"] != "").to_numpy()).all()
      and not bool(_fm["weak"]) and bool(_fm["bright"]),
      (list(fl["peak_id"]), list(fl["admitted_by"]), list(_fm)))
# the documented fallback survives where it is meant to: a cfg NEVER stamped has
# no resolved decision to honour, so a numeric knob is still read.
_un = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.6)
_um = pd.Series(ADM.admissible(fl, _un).to_numpy(), index=fl["peak_id"])
check("an UNSTAMPED cfg still honours a numeric occurrence_min",
      bool(_um["weak"]) and _un.occurrence_resolved is False)

# ---------------------------------------------------------------------------
# tiering: persistence gates ENTRY, corroboration gates the TIER. An
# occurrence-admitted M0 with no isotopologue / channel / series corroboration is
# capped at Candidate ('persistent-weak'); with a 13C child it is not.
# ---------------------------------------------------------------------------
from peaky.assignment import tiers as TR  # noqa: E402
tl = L.new_ledger(pd.DataFrame({"peak_id": ["W", "X", "Y", "Z"],
                                "mz": [264.0361, 265.0395, 278.0519, 300.0],
                                "height": [2.0, 0.05, 2.5, 500.0]}))
# (new_ledger sorts by height, so stamp by peak_id, not by position)
tl["admitted_by"] = tl["peak_id"].map({"W": "occurrence", "X": "", "Y": "occurrence", "Z": "height"})
tl["occurrence"] = tl["peak_id"].map({"W": 0.93, "X": np.nan, "Y": 0.90, "Z": 0.5})
for pid, f, ion in (("W", "C8H10O6", "C8H9O6-"), ("Y", "C9H12O6", "C9H11O6-"), ("Z", "C10H16O4", "C10H15O4-")):
    L.commit_assignment(tl, pid, neutral_formula=f, adduct="[M-H]-", ion_formula=ion,
                        ion_score=0.90, compound_score=0.90, eff_score=0.88, eff_margin=0.2,
                        tied=False, ppm_error=0.3, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="Pass 1", alternatives=[])
L.attach_isotopologue(tl, "X", "W", iso_label="13C", iso_match_score=0.9)   # W is corroborated
t = TR.compute_tiers(tl).set_index("peak_id")
check("tier cap: occurrence-admitted + uncorroborated -> Candidate 'persistent-weak'",
      t.at["Y", "tier"] == "Candidate" and str(t.at["Y", "tier_reason"]).startswith("persistent-weak"),
      t.loc["Y"].to_dict())
check("tier cap: occurrence-admitted but isotope-corroborated -> not capped by this rule",
      not str(t.at["W", "tier_reason"]).startswith("persistent-weak"), t.loc["W"].to_dict())
check("tier cap: height-admitted uncorroborated Good stays as the other rules decide (not persistent-weak)",
      not str(t.at["Z", "tier_reason"]).startswith("persistent-weak"), t.loc["Z"].to_dict())

# ...and the other two corroboration sources lift the cap exactly as the isotope
# child does: a SECOND CHANNEL carrying the same neutral, and a SERIES ANCHOR.
# (C = occurrence-admitted on [M-H]-, with C2 the same neutral on [M+Br]-;
# S = occurrence-admitted with a series unit; N = neither, the control.)
tl2 = L.new_ledger(pd.DataFrame({"peak_id": ["C", "C2", "S", "N", "K"],
                                 "mz": [264.0361, 343.9540, 278.0519, 292.0675, 306.0832],
                                 "height": [2.0, 2.1, 2.2, 2.3, 2.4]}))
tl2["admitted_by"] = "occurrence"
tl2["occurrence"] = 0.91
_rows = (("C", "C8H10O6", "[M-H]-", "cheminfo+grid", None),
         ("C2", "C8H10O6", "[M+Br]-", "cheminfo+grid", None),   # same neutral, 2nd channel
         ("S", "C9H12O6", "[M-H]-", "series-gka", "CH2"),       # series anchor
         ("N", "C10H14O6", "[M-H]-", "cheminfo+grid", None),    # control: neither
         ("K", "C2H4O3", "[M-H]-", "known:atmospheric-acid", None))   # pass-0 known species
for pid, f, add, meth, unit in _rows:
    L.commit_assignment(tl2, pid, neutral_formula=f, adduct=add,
                        ion_score=0.90, compound_score=0.90, eff_score=0.88, eff_margin=0.2,
                        tied=False, ppm_error=0.3, pass_no=1, method=meth,
                        confidence="Good", commentary="fixture", alternatives=[],
                        series_unit=unit)
t2 = TR.compute_tiers(tl2).set_index("peak_id")
check("tier cap control: occurrence-admitted with no corroboration at all -> persistent-weak",
      t2.at["N", "tier"] == "Candidate" and str(t2.at["N", "tier_reason"]).startswith("persistent-weak"),
      t2.loc["N"].to_dict())
check("tier cap EXEMPTION: a second channel on the same neutral lifts it",
      not str(t2.at["C", "tier_reason"]).startswith("persistent-weak")
      and not str(t2.at["C2", "tier_reason"]).startswith("persistent-weak"),
      t2.loc[["C", "C2"]].to_dict("index"))
check("tier cap EXEMPTION: a series anchor lifts it",
      not str(t2.at["S", "tier_reason"]).startswith("persistent-weak"), t2.loc["S"].to_dict())
# PRECEDENCE: the pass-0 known-species branch is tested BEFORE the persistence
# cap, so a curated identity stays Assigned even when it was admitted by
# persistence alone -- its evidence is the locked list, not this peak's height.
check("tier cap PRECEDENCE: a pass-0 known species stays Assigned despite persistence-only admission",
      t2.at["K", "tier"] == "Assigned" and str(t2.at["K", "tier_reason"]).startswith("known species"),
      t2.loc["K"].to_dict())

# empty / degenerate inputs
e = ADM.bin_occurrence(ts.iloc[:0])
check("bin_occurrence on an empty table -> empty frame with attrs", len(e) == 0 and e.attrs["n_samples"] == 0)
try:
    ADM.bin_occurrence(ts.drop(columns=["height"]))
    check("bin_occurrence without height raises", False)
except ValueError:
    check("bin_occurrence without height raises ValueError", True)

# ---------------------------------------------------------------------------
# ONE binning tolerance: the table is built at sampling.BATCH_TOL_PPM by default
# (the same rule the selection and the merge use), the lookup reads the table's
# own stamped tolerance, and a table without it is refused rather than guessed.
# ---------------------------------------------------------------------------
from peaky.batch import sampling as SS       # noqa: E402
from peaky.batch import assign_batch as AB  # noqa: E402
check("bin_occurrence default tol == sampling.BATCH_TOL_PPM (6.0)",
      tab.attrs["tol_ppm"] == SS.BATCH_TOL_PPM == 6.0, tab.attrs)
# ...and the OTHER two batch-level binnings are bound to the same constant, not
# to independent literals: the merge's default tolerance and the selector's.
check("the merge's DEFAULT_TOL_PPM tracks sampling.BATCH_TOL_PPM",
      AB.DEFAULT_TOL_PPM == SS.BATCH_TOL_PPM, AB.DEFAULT_TOL_PPM)
import inspect  # noqa: E402
check("select_cover_samples bins at BATCH_TOL_PPM by default",
      inspect.signature(SS.select_cover_samples).parameters["tol_ppm"].default
      == SS.BATCH_TOL_PPM,
      inspect.signature(SS.select_cover_samples).parameters["tol_ppm"].default)
# behavioural: two m/z 5.5 ppm apart are ONE bin at 6 ppm and two bins at 3 ppm,
# so the default really is 6 (the selector used to bin at timeseries' 5.0 while
# the admission table and the merge binned at 6, and a bin the selector covered
# was then not the bin the merge saw)
_a = 300.0
_b = 300.0 * (1 + 5.5e-6)
_tw = pd.DataFrame({"sample_item_id": ["s0", "s0", "s1", "s1"],
                    "mz": [_a, _b, _a, _b], "height": [10.0, 10.0, 10.0, 10.0]})
check("selector default binning merges a 5.5 ppm pair into one bin (6 ppm)",
      SS.select_cover_samples(_tw, k_min=1).attrs["selection"]["n_bins_total"] == 1,
      SS.select_cover_samples(_tw, k_min=1).attrs["selection"])
check("...and splits it at tol_ppm=3 (the argument is really honoured)",
      SS.select_cover_samples(_tw, k_min=1, tol_ppm=3.0).attrs["selection"]["n_bins_total"] == 2)
tab6 = ADM.bin_occurrence(ts, tol_ppm=6.0)
check("a table built at 6 ppm and the default table agree bin-for-bin",
      len(tab6) == len(tab) and np.allclose(tab6["mz"], tab["mz"])
      and np.allclose(tab6["occurrence"], tab["occurrence"]))
# the lookup is the table's own rule -- distinct spectra within the table's
# tolerance of the PROBE -- so a probe 4 ppm off the ion's centre sees most, not
# all, of its 2-ppm-sigma cloud at 6 ppm ([-2, +10] ppm of it), and the same
# probe read at 12 ppm ([-8, +16]) sees the whole cloud: the lookup really reads
# attrs['tol_ppm'], and no other number. (The window carries the stamp's 1.5 mDa
# floor -- traces.MZ_FLOOR_DA -- so at m/z 264 nothing NARROWER than ~5.7 ppm can
# be told apart, which is why the discriminating probe widens rather than narrows.)
probe = 264.0361 * (1 + 4e-6)
v6 = ADM.lookup_occurrence([probe], tab6)[0]
check("lookup by the table's own tol: 4 ppm off-centre reads most (not all) of the cloud at 6 ppm",
      np.isfinite(v6) and 0.6 < v6 < ADM.lookup_occurrence([264.0361], tab6)[0], v6)
tab12 = tab6.copy(); tab12.attrs.update(tab6.attrs); tab12.attrs["tol_ppm"] = 12.0
v12 = ADM.lookup_occurrence([probe], tab12)[0]
check("...and the whole cloud at 12 ppm (the lookup really reads attrs['tol_ppm'])",
      np.isfinite(v12) and v12 > v6 and v12 > 0.85, (v12, v6))
check("a per-peak table carries its sample codes (the lookup's rule needs them)",
      "sample" in tab6.columns and tab6.attrs.get("kind") == "per-peak")
bare = pd.DataFrame({"mz": [300.0], "occurrence": [0.9]})     # no attrs at all
try:
    ADM.lookup_occurrence([300.0], bare)
    check("lookup on a table without attrs['tol_ppm'] raises", False, "no error")
except ValueError as _e:
    check("lookup on a table without attrs['tol_ppm'] raises ValueError", "tol_ppm" in str(_e), _e)
# 1-bin table (a degenerate but reachable batch): the single bin is the only
# candidate, inside tolerance and NaN outside. NOTE what this does NOT pin: the
# `len(bs) == 1` branch in lookup_occurrence is behaviour-neutral -- with it
# deleted, np.clip's crossed bounds give left = -1 and at length 1 `bs[-1]` IS
# `bs[0]`, so the general path returns exactly these four values. The branch is
# kept as explicitness, and no test can discriminate it; this check pins the
# ANSWER for a one-bin table, which is what callers depend on.
one_bin = pd.DataFrame({"mz": [300.0], "occurrence": [0.9]}); one_bin.attrs["tol_ppm"] = 6.0
v1 = ADM.lookup_occurrence([300.0, 300.0 * (1 + 4e-6), 300.0 * (1 + 20e-6), 150.0], one_bin)
check("1-bin table: inside-tol probes hit it, outside-tol probes are NaN",
      v1[0] == 0.9 and v1[1] == 0.9 and np.isnan(v1[2]) and np.isnan(v1[3]), v1)
check("stamp_admission returns the resolved brightness gate as 'height_gate_cps'",
      "height_gate_cps" in counts and counts["height_gate_cps"] == 100.0 and "height_cutoff" not in counts,
      counts)

# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
from peaky import cli  # noqa: E402
Pp = cli.build_parser()
a = Pp.parse_args(["assign", "--sample-id", "X"])
check("assign: --occurrence-min default 'auto'", a.occurrence_min == "auto" == ADM.DEFAULT_OCCURRENCE_MIN)
a = Pp.parse_args(["batch", "--batch", "B", "--occurrence-min", "AUTO", "--height-cutoff", "10"])
check("batch: 'AUTO' accepted, absolute --height-cutoff parsed", a.occurrence_min == "auto" and a.height_cutoff == 10.0)
a = Pp.parse_args(["batch", "--batch", "B", "--occurrence-min", "0"])
check("batch: --occurrence-min 0 disables", a.occurrence_min == 0.0)
a = Pp.parse_args(["pool", "--batches", "x", "--occurrence-min", "0.9"])
check("pool: --occurrence-min parsed", a.occurrence_min == 0.9)
check("PassConfig.occurrence_min default", P.PassConfig().occurrence_min == ADM.DEFAULT_OCCURRENCE_MIN)
# the pipeline's knob -> config translation must pass 'auto' through (it once cast to float)
from peaky import pipeline as PL  # noqa: E402
c = PL.gate_config(occurrence_min="auto", height_cutoff_x_edge=5, height_cutoff_cps=None)
check("gate_config: 'auto' kept as a string, x_edge applied",
      c.occurrence_min == "auto" and c.height_cutoff_x_edge == 5.0 and c.height_cutoff_cps is None)
c = PL.gate_config(occurrence_min=0, height_cutoff_cps=10)
check("gate_config: 0 and an absolute cutoff", c.occurrence_min == 0.0 and c.height_cutoff_cps == 10.0)
check("gate_config: None keeps the default", PL.gate_config().occurrence_min == "auto")
try:
    PL.gate_config(occurrence_min="never")
    check("gate_config rejects an unknown string", False)
except ValueError:
    check("gate_config rejects an unknown string", True)

# ---------------------------------------------------------------------------
# WHICH SITES ARE GATED -- pinned, because it is the meaning of an empty
# `admitted_by` in every ledger this package writes, and it CANNOT be read off a
# grep for `admissible(`: one of the four call sites is `directors._target_peaks`,
# the shared target-peak helper of five pass functions. Counting call sites
# instead of callers under-reports the gated set by four and mislabels pass-2/3
# as brightness-only. If this list changes, the module note in admission.py,
# docs/ASSIGNMENT.md, docs/ARCHITECTURE.md, docs/OUTPUTS.md, SKILL.md and the
# CHANGELOG all describe the wrong set and must be re-derived with it.
# ---------------------------------------------------------------------------
import ast  # noqa: E402

_PKG = Path(ADM.__file__).resolve().parent
_dsrc = (_PKG / "passes" / "directors.py").read_text()
_tree = ast.parse(_dsrc)
_via_helper = {
    fn.name for fn in ast.walk(_tree) if isinstance(fn, ast.FunctionDef)
    and any(isinstance(n, ast.Name) and n.id == "_target_peaks"
            for n in ast.walk(fn)) and fn.name != "_target_peaks"
}
check("five pass functions share directors._target_peaks (pass-1/2/3 + both pass-3 cluster resolvers)",
      _via_helper == {"run_pass1", "run_pass2", "run_pass3",
                      "_resolve_hx_clusters", "_resolve_acid_i2_clusters"}, _via_helper)
_direct = sorted(f.relative_to(_PKG).as_posix() for f in _PKG.rglob("*.py")
                 if f.name != "admission.py" and "admissible(" in f.read_text())
check("the direct admissible() callers are exactly ladders / residual / siloxane / directors",
      _direct == ["ladders.py", "passes/directors.py", "residual.py", "siloxane.py"], _direct)
# ...and the helper really passes a persistence-only peak through, so pass-2's
# and pass-3's target lists contain sub-gate peaks (the claim that would be
# false if `_target_peaks` still filtered on height alone)
from peaky.assignment.passes import directors as _DIR  # noqa: E402
_gl = L.new_ledger(pd.DataFrame({"peak_id": ["weak", "bright"], "mz": [250.0, 260.0],
                                 "height": [2.0, 500.0]}))
_gtab = pd.DataFrame({"mz": [250.0, 260.0], "occurrence": [0.93, 0.93], "n_samples": [93, 93]})
_gtab.attrs.update(tol_ppm=6.0, n_samples=100)
_cfg_g = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.8)
ADM.stamp_admission(_gl, _cfg_g, _gtab)
_tp = set(_DIR._target_peaks(_gl, _cfg_g)["peak_id"])
check("_target_peaks admits the persistence-only peak (so pass-1/2/3 all see it)",
      _tp == {"weak", "bright"}, (_tp, dict(zip(_gl["peak_id"], _gl["admitted_by"]))))
_cfg_h = P.PassConfig(height_cutoff_cps=100.0, occurrence_min=0.0)
ADM.stamp_admission(_gl, _cfg_h, _gtab)
check("...and drops it when the persistence path is off (brightness alone)",
      set(_DIR._target_peaks(_gl, _cfg_h)["peak_id"]) == {"bright"})


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)

