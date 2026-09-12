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
check("no table: occurrence NaN, admitted_by height-only, n_bins 0",
      led_nb["occurrence"].isna().all() and (ADM.admissible(led_nb, cfg) == (led_nb["height"] >= 100)).all()
      and c["n_bins"] == 0)
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
from peaky.batch import sampling as SS  # noqa: E402
check("bin_occurrence default tol == sampling.BATCH_TOL_PPM (6.0)",
      tab.attrs["tol_ppm"] == SS.BATCH_TOL_PPM == 6.0, tab.attrs)
tab6 = ADM.bin_occurrence(ts, tol_ppm=6.0)
check("a table built at 6 ppm and the default table agree bin-for-bin",
      len(tab6) == len(tab) and np.allclose(tab6["mz"], tab["mz"])
      and np.allclose(tab6["occurrence"], tab["occurrence"]))
# a peak 4 ppm off a bin centre is inside a 6-ppm lookup; the lookup must use the
# table's tolerance (6), not any other number: at 3 ppm it would be NaN
probe = 264.0361 * (1 + 4e-6)
v6 = ADM.lookup_occurrence([probe], tab6)[0]
check("lookup matches by the table's own tol (4 ppm off, inside 6 ppm)", np.isfinite(v6) and v6 > 0.85, v6)
tab3 = tab6.copy(); tab3.attrs.update(tab6.attrs); tab3.attrs["tol_ppm"] = 3.0
check("...and would not at 3 ppm (the lookup really reads attrs['tol_ppm'])",
      np.isnan(ADM.lookup_occurrence([probe], tab3)[0]))
bare = pd.DataFrame({"mz": [300.0], "occurrence": [0.9]})     # no attrs at all
try:
    ADM.lookup_occurrence([300.0], bare)
    check("lookup on a table without attrs['tol_ppm'] raises", False, "no error")
except ValueError as _e:
    check("lookup on a table without attrs['tol_ppm'] raises ValueError", "tol_ppm" in str(_e), _e)
# 1-bin table: no bracketing pair exists -- the single bin is the only candidate
one_bin = pd.DataFrame({"mz": [300.0], "occurrence": [0.9]}); one_bin.attrs["tol_ppm"] = 6.0
v1 = ADM.lookup_occurrence([300.0, 300.0 * (1 + 4e-6), 300.0 * (1 + 20e-6), 150.0], one_bin)
check("1-bin table: inside-tol probes hit it, outside-tol probes are NaN (no index wrap)",
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


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
