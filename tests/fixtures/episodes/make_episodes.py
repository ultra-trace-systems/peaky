"""The synthetic EPISODE batch (tests/fixtures/episodes): 40 spectra of a uronium-type
Orbitrap batch, every id, timestamp and height synthetic.

    python tests/fixtures/episodes/make_episodes.py      # rewrites batch_ts.csv + components.json

`build(**switches)` returns the per-peak batch table (the shape `peaky batch` reads as its
time series: sample_item_id, sample_item_name, datetime_utc, peak_id, mz, height) and the
component record; the switches make the variants the tests need (`plume=False` is the
no-episode batch of the golden). `tests/test_episodes.py` pins the stored CSV to
`build()`, so the two cannot drift.

What is in it (`components.json` names every line, its m/z, its spectra and its role):

* 3 PERSISTENT ions in every spectrum, each with its 13C line at the first-order ratio
  (`isotopes.satellite_height`): C16H22O4 [M+H]+, C10H16O2 [M+(CH4N2O)H]+ (a urea adduct;
  11 carbons in the ion), C14H22O [M+H]+.
* a +3.85 mDa SATELLITE of C14H22O [M+H]+ in every spectrum: 0.4 x the parent with 3 %
  scatter, no 13C line of its own (the shoulder population a per-spectrum fit separates
  from an ion).
* a 4-spectrum PLUME (spectra 20-23, shape 0.40 / 1.00 / 0.75 / 0.35):
  - C8H10N4O2 [M+H]+ (caffeine-shaped), 3e5 at its peak, with its 13C line at the
    first-order ratio for 8 carbons;
  - the solvent ion C5H9NO [M+(CH4N2O)H]+ (an NMP-type urea cluster), 1e6 at the peak,
    with its 13C line;
  - the solvent CLUSTER of the persistent C16H22O4: C21H31NO5 [M+H]+ (= C16H22O4 +
    C5H9NO), whose height is the product of the parent's and the solvent ion's (it tracks
    both, r = 1 up to 2 % scatter) and which exists only inside the plume.
  The plume ion's EXACT-ALIAS RIVAL is C10H13FNSi [M+H]+, 0.25 mDa away, 10 carbons for 8:
  a formula alternative (no line of its own) only the 13C count can exclude.
* PAIR lines: 6 pairs of neighbouring spectra outside the plume share 6 minor lines each
  (prevalence 2, so the presence cover has bins to choose and takes 7 files without
  padding by TIC; never episodic: fewer than 3 spectra).
* NOISE: 25 picks per spectrum at heights 80-250 cps, at m/z no other spectrum has (the
  noise edge; singleton bins the cover's prevalence gate drops).

All masses carry a deterministic +-0.5 ppm jitter per spectrum.

The episode-evidence variants (all off by default, so `build()` is the stored table; none
draws from the random generator, so every default line keeps its jitter and heights):

* `envelope`: the plume ion's 15N (4 N), 13C2 and 18O lines at their first-order shares --
  its isotope envelope complete enough for the evidence level to reach 4b;
* `rival_lines`: the exact-alias rival's own 29Si / 30Si lines in the plume spectra (the
  rival's silicon is then no longer refuted by the spectra; only the 13C count could
  exclude it, and at 8 C against its 10 it cannot);
* `dim`: a dim plume ion, C7H8N4O2 [M+H]+ at 1.5e3 cps, no 13C line (its predicted 13C
  S/N ~1.4: untestable) -- it rises exactly with the plume (co-occurrence, r = 1);
* `pair_adduct`: the plume ion's urea adduct C8H10N4O2 [M+(CH4N2O)H]+ at 0.3x it (an
  adduct partner in the same spectra);
* `true_ion`: a line 3.85 mDa above the persistent C16H22O4 [M+H]+ that follows it at 0.1x
  and carries its own 13C line (11 C) -- an ion at the satellite's offset (its `reading`
  is recorded, not committed: a test commits it in the files it chooses);
* `cluster_track=False`: the solvent cluster's heights do not follow parent x solvent;
* `cluster_outside`: the solvent cluster is also in 10 spectra outside the plume (not
  confined to the episode);
* `cluster_pair`: the same NMP cluster of the persistent C16H22O4 under the urea ion form
  ([M+(C5H9NO)(CH4N2O)H]+, 0.3x the cluster) in the plume spectra -- the cluster's pair.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from peaky.chem import chemistry as C   # noqa: E402
from peaky.chem import isotopes as ISO  # noqa: E402

N_SPECTRA = 40
T0 = pd.Timestamp("2031-03-02 00:00:00", tz="UTC")
STEP_MIN = 10
PLUME_AT = (20, 21, 22, 23)
PLUME_SHAPE = (0.40, 1.00, 0.75, 0.35)
PAIRS = ((0, 1), (6, 7), (12, 13), (28, 29), (32, 33), (36, 37))
SAT_DMZ = 3.85e-3
SAT_SLOPE = 0.4
SEED = 20310302

PERSISTENT = (  # (key, neutral, adduct, carbons in the ion, base height, shape)
    ("P1", "C16H22O4", "[M+H]+", 16, 2.0e5, lambda i: 1 + 0.3 * np.sin(2 * np.pi * i / 20)),
    ("P2", "C10H16O2", "[M+(CH4N2O)H]+", 11, 5.0e4, lambda i: 1 + 0.2 * np.cos(2 * np.pi * i / 13)),
    ("P3", "C14H22O", "[M+H]+", 14, 1.0e5, lambda i: 1 + 0.4 * np.sin(2 * np.pi * i / 40 + 1)),
)
PLUME_ION = ("E1", "C8H10N4O2", "[M+H]+", 8, 3.0e5)
SOLVENT_ION = ("S1", "C5H9NO", "[M+(CH4N2O)H]+", 6, 1.0e6)
CLUSTER = ("K1", "C21H31NO5", "[M+H]+")          # P1's neutral + C5H9NO
CLUSTER_K = 0.02                                  # cluster = K x P1 x S1 / S1's peak height
RIVAL = ("C10H13FNSi", "[M+H]+")


def sid(i: int) -> str:
    return f"ep{i:02d}"


def _r13(n_c: int, h):
    return ISO.satellite_height("C", n_c, h)


DIM_ION = ("E2", "C7H8N4O2", "[M+H]+", 7, 1.5e3)
PAIR_ION = ("E3", "C8H10N4O2", "[M+(CH4N2O)H]+", 9, 0.3)        # x the plume ion
TRUE_ION = ("T1", "C11H20F2N4O2", "[M+H]+", 11, 0.1)            # x P1, 3.85 mDa above it
OUTSIDE_AT = (2, 5, 9, 14, 17, 26, 30, 34, 38, 39)
CLUSTER_NOTRACK = (1.0, 0.2, 0.9, 0.5)                           # the cluster's shape when it does not track


def build(*, plume: bool = True, c13_child: bool = True, cluster: bool = True,
          satellite: bool = True, pairs: bool = True, envelope: bool = False, rival_lines: bool = False,
          dim: bool = False, pair_adduct: bool = False, true_ion: bool = False, cluster_track: bool = True,
          cluster_outside: bool = False, cluster_pair: bool = False) -> tuple[pd.DataFrame, dict]:
    """(per-peak batch table, component record). Deterministic."""
    rng = np.random.default_rng(SEED)
    jit = rng.normal(0.0, 0.25, size=(N_SPECTRA, 64)).clip(-0.5, 0.5)   # ppm, per spectrum and line slot
    sat_scatter = rng.normal(0.0, 0.03, size=N_SPECTRA)
    clu_scatter = rng.normal(0.0, 0.02, size=N_SPECTRA)
    noise_h = rng.uniform(80.0, 250.0, size=(N_SPECTRA, 25))
    lines: list[dict] = []          # one record per line (the component list)
    rows: list[dict] = []

    def add(key, mz, spectra, heights, role, slot, **meta):
        lines.append(dict(key=key, mz=round(float(mz), 6), role=role,
                          spectra=[sid(i) for i in spectra], **meta))
        for i, h in zip(spectra, heights):
            rows.append(dict(i=i, key=key, mz=float(mz) * (1 + jit[i, slot] * 1e-6), height=float(h)))

    every = list(range(N_SPECTRA))
    slot = 0
    p_h = {}
    for key, nf, ad, nc, h0, shape in PERSISTENT:
        mz = C.ion_mz(nf, ad)
        hs = [h0 * shape(i) for i in every]
        p_h[key] = hs
        add(key, mz, every, hs, "persistent", slot, neutral_formula=nf, adduct=ad, n_c=nc); slot += 1
        add(key + "_13C", mz + ISO.D_13C, every, [_r13(nc, h) for h in hs], "13C", slot,
            parent=key); slot += 1
    if satellite:
        mz = C.ion_mz("C14H22O", "[M+H]+") + SAT_DMZ
        add("SAT", mz, every, [SAT_SLOPE * h * (1 + s) for h, s in zip(p_h["P3"], sat_scatter)],
            "satellite", slot, parent="P3", dmz=SAT_DMZ, slope=SAT_SLOPE)
    slot += 1
    if plume:
        key, nf, ad, nc, h0 = PLUME_ION
        mz = C.ion_mz(nf, ad)
        hs = [h0 * f for f in PLUME_SHAPE]
        add(key, mz, PLUME_AT, hs, "plume", slot, neutral_formula=nf, adduct=ad, n_c=nc,
            rival={"neutral_formula": RIVAL[0], "adduct": RIVAL[1],
                   "mz": round(C.ion_mz(*RIVAL), 6),
                   "dmz_mDa": round((C.ion_mz(*RIVAL) - mz) * 1e3, 4)}); slot += 1
        if c13_child:
            add(key + "_13C", mz + ISO.D_13C, PLUME_AT, [_r13(nc, h) for h in hs], "13C", slot,
                parent=key)
        slot += 1
        key, nf, ad, nc, h0 = SOLVENT_ION
        mz = C.ion_mz(nf, ad)
        s_h = [h0 * f for f in PLUME_SHAPE]
        add(key, mz, PLUME_AT, s_h, "solvent", slot, neutral_formula=nf, adduct=ad, n_c=nc); slot += 1
        add(key + "_13C", mz + ISO.D_13C, PLUME_AT, [_r13(nc, h) for h in s_h], "13C", slot,
            parent=key); slot += 1
        if cluster:
            key, nf, ad = CLUSTER
            hs = [CLUSTER_K * p_h["P1"][i] * s / h0 * (1 + clu_scatter[i])
                  for i, s in zip(PLUME_AT, s_h)]
            if not cluster_track:
                hs = [max(hs) * f for f in CLUSTER_NOTRACK]
            at = PLUME_AT
            if cluster_outside:
                at = tuple(sorted(PLUME_AT + OUTSIDE_AT))
                hv = dict(zip(PLUME_AT, hs))
                hs = [hv.get(i, 0.3 * max(hs)) for i in at]
            add(key, C.ion_mz(nf, ad), at, hs, "solvent_cluster", slot, neutral_formula=nf,
                adduct=ad, parent="P1", solvent="S1", solvent_neutral="C5H9NO")
            if cluster_pair:      # the same NMP cluster of P1's neutral under the urea ion form
                add("K2", C.ion_mz("C16H22O4", "[M+(C5H9NO)(CH4N2O)H]+"), PLUME_AT,
                    [0.3 * h for h in hs[:len(PLUME_AT)]], "solvent_cluster_pair", 49,
                    reading=["C16H22O4", "[M+(C5H9NO)(CH4N2O)H]+"], parent="P1", solvent="S1")
        slot += 1
        e_h = [PLUME_ION[4] * f for f in PLUME_SHAPE]
        e_mz = C.ion_mz(PLUME_ION[1], PLUME_ION[2])
        if envelope:          # the plume ion's 15N, 13C2 and 18O lines
            add("E1_15N", e_mz + ISO.D_15N, PLUME_AT, [h * ISO.R_15N_PER_N * 4 for h in e_h], "15N", 40,
                parent="E1")
            add("E1_13C2", e_mz + 2 * ISO.D_13C, PLUME_AT, [h * (ISO.R_13C_PER_C * 8) ** 2 / 2 * 7 / 8
                                                            for h in e_h], "13C2", 41, parent="E1")
            add("E1_18O", e_mz + ISO.D_18O, PLUME_AT, [h * ISO.R_18O_PER_O * 2 for h in e_h], "18O", 42,
                parent="E1")
        if rival_lines:       # the rival's silicon lines
            add("R1_29Si", e_mz + ISO.D_29SI, PLUME_AT, [h * ISO.R_29SI_PER_SI for h in e_h], "29Si", 43,
                parent="E1")
            add("R1_30Si", e_mz + ISO.D_30SI, PLUME_AT, [h * ISO.R_30SI_PER_SI for h in e_h], "30Si", 44,
                parent="E1")
        if dim:
            key, nf, ad, nc, h0d = DIM_ION
            add(key, C.ion_mz(nf, ad), PLUME_AT, [h0d * f for f in PLUME_SHAPE], "plume_dim", 45,
                neutral_formula=nf, adduct=ad, n_c=nc)
        if pair_adduct:
            key, nf, ad, nc, x = PAIR_ION
            add(key, C.ion_mz(nf, ad), PLUME_AT, [h * x for h in e_h], "plume_pair", 46, neutral_formula=nf,
                adduct=ad, n_c=nc, parent="E1")
    if true_ion:
        key, nf, ad, nc, x = TRUE_ION
        mz = C.ion_mz("C16H22O4", "[M+H]+") + SAT_DMZ
        hs = [x * h for h in p_h["P1"]]
        add(key, mz, every, hs, "true_ion", 47, reading=[nf, ad], n_c=nc, parent="P1", dmz=SAT_DMZ)
        add(key + "_13C", mz + ISO.D_13C, every, [_r13(nc, h) for h in hs], "13C", 48, parent=key)
    if pairs:
        for p, (a, b) in enumerate(PAIRS):
            for j in range(6):
                mz = 131.0417 + 17.3311 * j + 3.0071 * p
                add(f"Q{p}{j}", mz, (a, b), (2.0e3, 1.8e3), "pair", slot + j)
    real = np.sort([x["mz"] for x in lines])
    for i in every:
        for k in range(25):
            mz = 61.3713 + 13.9191 * k + 0.0213 * i
            j = np.searchsorted(real, mz)
            near = min(abs(real[max(j - 1, 0)] - mz), abs(real[min(j, len(real) - 1)] - mz))
            if near / mz * 1e6 < 50:
                continue          # never on (or within 50 ppm of) a real line
            rows.append(dict(i=i, key="noise", mz=mz, height=float(noise_h[i, k])))
    t = pd.DataFrame(rows).sort_values(["i", "mz"], kind="mergesort").reset_index(drop=True)
    t["sample_item_id"] = t["i"].map(sid)
    t["datetime_utc"] = [T0 + pd.Timedelta(minutes=STEP_MIN * int(i)) for i in t["i"]]
    t["sample_item_name"] = t["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    t["peak_id"] = [f"{s}-{n:03d}" for s, n in zip(t["sample_item_id"], t.groupby("i").cumcount())]
    t["mz"] = t["mz"].round(6)
    t["height"] = t["height"].round(1)
    comp = {"n_spectra": N_SPECTRA, "plume_spectra": [sid(i) for i in PLUME_AT] if plume else [],
            "plume_peak": sid(PLUME_AT[int(np.argmax(PLUME_SHAPE))]) if plume else None,
            "lines": lines}
    return (t[["sample_item_id", "sample_item_name", "datetime_utc", "peak_id", "mz", "height", "key"]],
            comp)


def write(out_dir: str = HERE) -> None:
    t, comp = build()
    t.to_csv(os.path.join(out_dir, "batch_ts.csv"), index=False)
    with open(os.path.join(out_dir, "components.json"), "w") as fh:
        json.dump(comp, fh, indent=1)
        fh.write("\n")


def load(path: str | None = None) -> pd.DataFrame:
    """The stored table, typed as `build()` returns it."""
    t = pd.read_csv(path or os.path.join(HERE, "batch_ts.csv"))
    t["datetime_utc"] = pd.to_datetime(t["datetime_utc"], utc=True)
    return t


if __name__ == "__main__":
    write()
