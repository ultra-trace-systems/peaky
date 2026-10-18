"""Offline tests for the MS2 breakdown stage: readings across energies and the
ladder gate (peaky/ms2/readings.py), fragment attribution (attribute.py),
survival / E50 / ion budget (breakdown.py), the orchestrator and figures
(analyze.py) and `peaky ms2 analyze`. No network, no Mascope: the run folders
are built from the committed fixtures (tests/fixtures/ms2) exactly as `ms2
extract` builds them.

Known answers, our own data:

  * monoterpene standard A (2026-09-11 10:52, limonene) and B (15:01,
    alpha-pinene): the raw-file breakdown of that evening (the RAW files read
    per scan, survival = precursor / (precursor + the tracked ions)) gave the
    [M+H]+ E50 at NCE 13.8 (6.32 V) for A and 16.3 (6.76 V) for B, and A's
    ammonium adduct C10H20^N+ at NCE 1.6 (4.19 V). Those tables' survival
    values are inlined below (rounded to 1e-5) and re-read here.
  * the 2026-09-10 exploratory 15N-ammonium file: seven parents and the
    co-isolates that day's script settled; its E50s and calls are pinned as
    regression values, each explained where it is pinned.

Synthetic: the hexose [M+NH4]+ dehydration ladder (and a 15N-ammonium one
read under several hash seeds), a two-precursor mixture with known branching,
three candidates, an absent named co-isolate, mixed activation kinds (a
majority, a tie), a precursor that does not fragment, the mass-balance guard
(an adduct's own in-source products count; every energy is checked; its
bound is the lowest energy's), the intact test against the lowest energy, a
spectrum scaled as a whole (nothing but S moves), a fragment whose rank-1
formula changes between energies (one fragment, its candidates read at its
signal-weighted m/z), its mirror -- a stray centroid only the target can read
takes nothing from the co-isolate that made the fragment (the intact test
reads its onset where every candidate reads it) -- and its neighbours, a
branching breach where the ion lost nothing, empty spectra,
partial volts, a peak two window ions read as their precursor, and the E50
edge cases. And: share weights chosen per candidate (a named ion the sample
lacks moves no other ion's weight, nor the intact test's onset), a printed
unnamed or radical loss piece (no ladder step), an ion beyond the MS1 window `ms2
extract` inventoried, an energy where an ion is unobserved (S a gap like F),
the branching note's steps and values, a product outside the observed m/z
range (seen by S only), a weak centroid between two resolved fragments (it
never merges them), the C2F3 = H2^NO4 alias of 15N data, the outputs
section's columns and the figure's colours."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.ms2 import analyze as AN
from peaky.ms2 import attribute as AT
from peaky.ms2 import breakdown as BD
from peaky.ms2 import readings as RD
from peaky.ms2 import sources as S
from peaky.ms2 import subformula as SF
from peaky.ms2 import assign_fragments, e50, e_cm

FIX = Path(__file__).resolve().parent / "fixtures" / "ms2"
NAN = float("nan")


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _ions(*formulas):
    return AN.collect_ions((f, NAN, AN.ion_mz(f)) for f in formulas)


def _tables(stem, iso=None):
    """What `ms2 extract` builds for a fixture sample (the window with the
    server's width, or `iso`; none when neither is known)."""
    cent = _load(f"{stem}_centroids.json")
    p = FIX / f"{stem}_summary.json"
    summ = _load(p.name) if p.exists() else None
    spectra = S.spectra_table("S", cent, summ)
    parents = S.parents_table("S", cent, summ)
    width = iso if iso is not None else (summ or {}).get("isolation_width")
    window = (S.window_table("S", _load(f"{stem}_ms1.json"), sorted(set(parents["parent_mz"])),
                             width) if width else S._frame({}, S.WINDOW_COLUMNS))
    return AN.RunTables(spectra=spectra, parents=parents, window=window,
                        manifest={"sample_id": "S", "parameters": {"isolation_width_used": width}})


def _mz(ion):
    return AN.ion_mz(ion)


#: the monoterpene windows' ions: [M+H]+, the 15N-ammonium adduct, and the
#: co-isolated C10H19O+ ([M+H+H2O]+-like) 13 mDa below it
MT_IONS = ("C10H17+", "C10H20^N+", "C10H19O+")
#: the 2026-09-10 file: each parent's ion and the co-isolates that day's
#: script settled (its COISO table)
NH4_IONS = ("C7H6NS+", "C4H10NS2+", "C8H10NO+",
            "C4H10^NO4+", "C8H9O2+", "C5H14^NO3+", "C6H18^NO2+", "C9H14^N+",
            "C11H13O+", "C11H15O2+", "C11H16^NO+", "C11H18^NO2+", "C12H21O2+",
            "C12H22^NO2+", "C8H28^NO4Si4+")

#: The raw-file breakdown of 2026-09-11 (nce, volts, survival), from the RAW
#: files read per scan: survival = precursor / (precursor + the tracked ions).
RAW_A_137 = ((1, 4.05, 0.74299), (3, 4.40, 0.70232), (5, 4.76, 0.66414),
             (9, 5.46, 0.58938), (20, 7.41, 0.38656))
#: A's adduct window, survival = C10H20^N+ / (C10H20^N+ + the co-isolated
#: C10H19O+ + 137 + 81 + 95 + 109 + 67 + 69): the co-isolate's intact peak is
#: in the denominator, and every shared fragment counts against the adduct.
RAW_A_155 = ((1, 4.07, 0.50892), (5, 4.87, 0.45021), (15, 6.87, 0.23149),
             (25, 8.88, 0.06535))
RAW_B_137 = ((1, 4.05, 0.75088), (5, 4.76, 0.67775), (10, 5.64, 0.59972),
             (15, 6.53, 0.52226), (25, 8.29, 0.35278), (35, 10.06, 0.19459))


def _raw_e50(rows):
    nce, v, s = zip(*rows)
    return e50(nce, s, v)


def _curve(res, parent, ion, col="f"):
    bd = res.breakdown
    g = bd[np.isclose(bd["parent_mz"], parent) & (bd["ion"] == ion)].sort_values("nce")
    return g["nce"].to_numpy(), g[col].to_numpy(), g


def _e50_row(res, parent, ion, curve="F"):
    e = res.e50
    r = e[np.isclose(e["parent_mz"], parent) & (e["ion"] == ion) & (e["curve"] == curve)]
    assert len(r) == 1, (parent, ion, curve)
    return r.iloc[0]


def _weights(text):
    """`ms2_fragments.csv`'s weights cell ('ion=w;...') -> {ion: w}."""
    return {k: float(v) for k, v in (kv.split("=") for kv in text.split(";"))}


@pytest.fixture(scope="module")
def mta():
    return AN.analyze_run(_tables("ms2_monoterpene_a"), _ions(*MT_IONS))


@pytest.fixture(scope="module")
def nh4():
    return AN.analyze_run(_tables("ms2_nh4_exploratory"), _ions(*NH4_IONS))


# ------------------------------------------------------- known answers ----
def test_the_raw_file_tables_give_the_e50s_reported_that_day():
    """e50 on the raw-file survival values reproduces the E50s of that day."""
    for rows, nce, volts in ((RAW_A_137, 13.8, 6.32), (RAW_A_155, 1.6, 4.19),
                             (RAW_B_137, 16.3, 6.76)):
        e = _raw_e50(rows)
        assert e.flag == BD.FLAG_INTERPOLATED and e.monotonic
        assert (round(e.nce, 1), round(e.volts, 2)) == (nce, volts)


def test_standard_a_protonated_e50_matches_the_raw_file_breakdown(mta):
    """[M+H]+ C10H17+ (137.1321 window, alone in it): F from the API's
    averaged S/N >= 3 spectra agrees with the raw-file survival within 0.0013
    at every step (the definitions agree here: one window ion, every reading
    is its fragment). The local slope at the crossing is 0.018 per NCE, so
    the E50s may differ by 0.0013 / 0.018 = 0.07 NCE: they differ by 0.04."""
    nce, f, g = _curve(mta, 137.1321, "C10H17+")
    raw = np.array([s for *_, s in RAW_A_137])
    assert list(nce) == [r[0] for r in RAW_A_137]
    assert np.abs(f - raw).max() < 0.0013 + 1e-4
    assert np.array_equal(g["f"], g["f_lo"])                   # nothing shared
    r = _e50_row(mta, 137.1321, "C10H17+")
    want = _raw_e50(RAW_A_137)
    assert r["flag"] == "interpolated" and abs(r["e50_nce"] - want.nce) <= 0.07
    assert abs(r["e50_v"] - want.volts) <= 0.07 * (7.41 - 5.46) / 11
    # the regression value, and its centre-of-mass energy on N2
    assert r["e50_nce"] == pytest.approx(13.891, abs=1e-3)
    assert r["e50_v"] == pytest.approx(6.327, abs=1e-3)
    assert r["e50_ecm"] == pytest.approx(r["e50_v"] * 28.0061 / (28.0061 + _mz("C10H17+")))
    assert r["role"] == "target" and r["ion_mz"] == pytest.approx(_mz("C10H17+"))
    # S compares spectra recorded at different times: reported, and said to
    # need a steady precursor supply (never an injection-time condition --
    # Orbitrap intensities are per unit injection time)
    s = _e50_row(mta, 137.1321, "C10H17+", "S")
    assert s["note"] == AN.S_STEADY and s["e50_nce"] > r["e50_nce"]
    assert "injection time" not in s["note"]


def test_standard_a_ammonium_adduct_differs_by_definition_only(mta):
    """The adduct C10H20^N+ (155.1559 window, with C10H19O+ co-isolated 13 mDa
    below). The raw-file survival (1.6 NCE) put the co-isolate's intact peak
    and every shared fragment against the adduct; rebuilt from the same API
    spectra that definition differs from the raw table by 0.0041 at NCE 1 and
    0.0065 at NCE 5, the crossing segment (0.010 elsewhere: averaged S/N >= 3
    spectra of one isolation centre, against per-scan raw centroids pooled over
    both the 155.1429 and 155.1559 centres). At the raw table's slope there,
    0.0147 per NCE, the E50s may differ by at most 0.0065 / 0.0147 = 0.44 NCE:
    the rebuilt definition reads 1.92 against the table's 1.61.

    F leaves the co-isolate's intact peak out and gives the adduct its MS2
    survivor share (0.72) of the shared fragments: E50 8.61. F_lo gives it
    every shared fragment: 6.14. The rebuilt raw-file definition <= F_lo <= F
    at every energy; the raw table itself sits 0.0027 ABOVE F_lo at NCE 25 --
    the same spectra-not-definitions difference as standard B's."""
    nce, f, g = _curve(mta, 155.1559, "C10H20^N+")
    _, _, o = _curve(mta, 155.1559, "C10H19O+")
    assert list(nce) == [1, 5, 15, 25]
    raw_def = g["intensity"].to_numpy() / (g["intensity"].to_numpy() + o["intensity"].to_numpy()
                                            + g["frag_all"].to_numpy())
    raw = np.array([s for *_, s in RAW_A_155])
    assert np.round(raw - raw_def, 4).tolist() == [-0.0041, -0.0065, 0.0103, 0.0036]
    e_raw_def = e50(nce, raw_def, g["hcd_v"])
    assert e_raw_def.nce == pytest.approx(1.924, abs=1e-3)
    assert abs(e_raw_def.nce - _raw_e50(RAW_A_155).nce) <= 0.0065 / 0.0147
    assert (raw_def <= g["f_lo"].to_numpy() + 1e-12).all()
    assert (g["f_lo"].to_numpy() <= f + 1e-12).all()
    assert raw[3] - g["f_lo"].to_numpy()[3] == pytest.approx(0.0027, abs=1e-4)
    r = _e50_row(mta, 155.1559, "C10H20^N+")
    assert (r["role"], r["flag"]) == ("target", "interpolated")
    assert r["e50_nce"] == pytest.approx(8.606, abs=1e-3)
    assert r["e50_v"] == pytest.approx(5.591, abs=1e-3)
    assert _e50_row(mta, 155.1559, "C10H20^N+", "F_lo")["e50_nce"] == pytest.approx(6.137, abs=1e-3)
    # the same ion seen from the other isolation centre (155.1429, where it is
    # the co-isolate) reads the same within 0.25 NCE
    other = _e50_row(mta, 155.1429, "C10H20^N+")
    assert other["role"] == "co-isolate" and abs(other["e50_nce"] - r["e50_nce"]) < 0.25
    # every fragment of that window is a sub-formula of both ions. C10H17+ and
    # C6H9+ are there at the lowest energy (in-source: no intact test); the
    # later ones appear where both ions are losing: survivor shares for all
    fr = mta.fragments[np.isclose(mta.fragments["parent_mz"], 155.1559)]
    assert set(fr["fragment"]) == {"C10H17+", "C6H9+", "C7H11+", "C8H13+", "C5H7+"}
    assert (fr["call"] == "ambiguous").all() and (fr["weight_source"] == "ms2 survivors").all()
    assert set(fr.loc[fr["in_source"], "fragment"]) == {"C10H17+", "C6H9+"}
    w = _weights(fr.set_index("fragment").loc["C10H17+", "weights"])
    assert w["C10H20^N+"] == pytest.approx(67108.999 / (67108.999 + 26181.926), abs=1e-5)
    assert fr.set_index("fragment").loc["C10H17+", "losses"] == "C10H20^N+: H3^N; C10H19O+: H2O"


def test_standard_a_branching_excess_is_unnamed_signal(mta):
    """docs/MS2.md §11.5: the [M+H]+ window's branching sum is flagged, 1.173 at
    NCE 5. The excess, 0.173, is in branching units; times the precursor's
    TIC-fraction loss (0.0664) it is 0.0115 of the TIC -- exactly what the
    unexplained share fell by (one window ion: mass balance). About half of
    that (0.0058) is the precursor's own side lobes, unexplained centroids
    within 4.6 mDa of it that no named ion can claim."""
    bd = mta.breakdown
    g = bd[np.isclose(bd["parent_mz"], 137.1321)].sort_values("nce").reset_index(drop=True)
    assert list(g["nce"]) == [1, 3, 5, 9, 20]
    s5 = g.loc[2, "branching_sum"]
    assert s5 == pytest.approx(1.1728, abs=1e-4) and g.loc[2, "branching_flag"]
    loss = g.loc[0, "frac"] - g.loc[2, "frac"]
    moved = g.loc[0, "budget_unexplained"] - g.loc[2, "budget_unexplained"]
    assert loss == pytest.approx(0.0664, abs=1e-4) and moved == pytest.approx(0.0115, abs=1e-4)
    assert (s5 - 1) * loss == pytest.approx(moved, abs=1e-9)
    w = [x for x in mta.windows if x.parent_mz == 137.1321][0]
    rd = mta.readings[np.isclose(mta.readings["parent_mz"], 137.1321)]

    def lobes(spec):
        pk = rd[rd["spectrum"] == spec].drop_duplicates("peak")
        near = pk[(pk["rank"] == 0) & (np.abs(pk["mz"] - w.precursor_mz) < 0.0046)]
        return near["intensity"].sum() / pk["intensity"].sum()
    drop = lobes("137.1321@hcd1.00") - lobes("137.1321@hcd5.00")
    assert drop == pytest.approx(0.0058, abs=1e-4) and 0.45 < drop / moved < 0.55
    assert any("side-lobe" in n for n in w.notes)


def test_standard_a_survivor_split_breaks_the_per_ion_balance(mta):
    """docs/MS2.md §11.5, 'What the survivor split assumes'. Every fragment of
    A's adduct window is shared and split 0.719 / 0.281 by the NCE 1
    survivors, so both ions read F = 0.713 at NCE 1 by construction. The
    window balances (its products gain 0.996, 1.010, 1.030 of what its two
    precursors lose), but split by survivors the adduct's branching sums to
    1.415 / 1.100 (flagged) and the co-isolate's to 0.566 / 0.835: the
    co-isolate, the more fragile (its TIC fraction -28 % by NCE 5, the
    adduct's -11 %), gets less than it made. A split by each ion's own loss
    -- balanced by construction, not used -- would move the adduct's F E50
    from 8.61 to 10.11 and the co-isolate's from 6.77 to 3.96: F_lo..F
    (6.14..8.61) is not a bound on both sides."""
    p, A, B = 155.1559, "C10H20^N+", "C10H19O+"
    _, fa, ga = _curve(mta, p, A)
    _, fb, gb = _curve(mta, p, B)
    assert fa[0] == pytest.approx(fb[0]) == pytest.approx(0.7131, abs=1e-4)
    assert np.round(ga["branching_sum"].to_numpy()[1:], 3).tolist() == [1.415, 1.1, 1.048]
    assert np.round(gb["branching_sum"].to_numpy()[1:], 3).tolist() == [0.566, 0.835, 0.987]
    assert list(ga["branching_flag"]) == [False, True, True, False]
    assert not gb["branching_flag"].any()
    br = mta.branching[np.isclose(mta.branching["parent_mz"], p)]
    by = br.groupby(["nce", "ion"]).agg(gain=("frag_gain", "sum"), loss=("ion_loss", "first"))
    win = by.groupby(level=0).sum()
    assert np.round((win["gain"] / win["loss"]).to_numpy(), 3).tolist() == [0.996, 1.01, 1.03]
    drop = lambda g: 1 - g["frac"].iloc[1] / g["frac"].iloc[0]  # noqa: E731
    assert (round(drop(gb), 2), round(drop(ga), 2)) == (0.28, 0.11)
    # the loss split, for the record (not what the code does)
    xa, xb = ga["frac"].to_numpy(), gb["frac"].to_numpy()
    ia, ib = ga["intensity"].to_numpy(), gb["intensity"].to_numpy()
    with np.errstate(invalid="ignore"):
        wa = (xa[0] - xa) / ((xa[0] - xa) + (xb[0] - xb))
    wa[0] = ia[0] / (ia[0] + ib[0])                # nothing lost yet: the survivors
    fa_loss = ia / (ia + wa * ga["frag_all"].to_numpy())
    fb_loss = ib / (ib + (1 - wa) * gb["frag_all"].to_numpy())
    nce = ga["nce"].to_numpy()
    assert e50(nce, fa_loss).nce == pytest.approx(10.115, abs=1e-3)
    assert e50(nce, fb_loss).nce == pytest.approx(3.956, abs=1e-3)
    assert _e50_row(mta, p, B)["e50_nce"] == pytest.approx(6.771, abs=1e-3)
    assert _e50_row(mta, p, A, "F_lo")["e50_nce"] < _e50_row(mta, p, A)["e50_nce"] < \
        e50(nce, fa_loss).nce


#: the nominal masses standard B's raw-file breakdown tracked (its table's columns)
RAW_B_TRACKED = (81, 95, 109, 67, 69, 93, 107, 121, 79, 57, 71, 55, 123)


def test_standard_b_has_no_volts_and_matches_the_raw_file_breakdown():
    """Standard B's summary was refused: the spectra come from the centroid
    keys, without volts or an isolation width. E50 on NCE only (volts and
    E_cm NaN, said); the window holds only the target, said.

    F reads E50 15.82 against the raw-file breakdown's 16.31. Not a difference
    of definitions: the raw-file survival -- precursor / (precursor + the
    tracked ions) -- rebuilt on the API spectra agrees with F within 0.0011 at
    every step, and exactly at the crossing's two steps (NCE 10 and 15, where
    every fragment read is a tracked one): E50 15.821. The difference is in the
    spectra: per precursor, the API spectra hold x1.021 and x1.037 the tracked
    fragments the raw-file table does at NCE 10 and 15 (x0.963 at NCE 1). A
    fragment / precursor ratio off by a factor 1 + d moves F = 1 / (1 + ratio)
    by about F (1 - F) d <= d / 4: with every ratio within 4 %, F lies within
    0.01 of the raw table at every step, and at the crossing's slope (0.0169
    per NCE) the E50 within 0.01 / 0.0169 = 0.59 NCE (it is 0.50 off)."""
    res = AN.analyze_run(_tables("ms2_monoterpene_b"), _ions("C10H17+"))
    (w,) = res.windows
    assert (w.parent_mz, w.target, w.co_isolates) == (137.13, "C10H17+", [])
    assert any("isolation width unknown" in n for n in w.notes)
    assert res.params["isolation_width_used"] is None
    nce, f, g = _curve(res, 137.13, "C10H17+")
    assert list(nce) == [r[0] for r in RAW_B_137]
    assert g["hcd_v"].isna().all()
    raw = np.array([s for *_, s in RAW_B_137])
    # the definition, on the API spectra: the tracked fragments only
    roles = AT.peak_roles(res.readings)
    frag = roles[roles["role"] == "fragment"]
    tracked = frag[np.rint(frag["mz"]).astype(int).isin(RAW_B_TRACKED)]
    trk = tracked.groupby("spectrum")["intensity"].sum().reindex(g["spectrum"]).fillna(0.0)
    i = g["intensity"].to_numpy()
    f_trk = i / (i + trk.to_numpy())
    assert np.abs(f_trk - f).max() < 0.0011
    assert np.array_equal(f_trk[2:4], f[2:4])                    # NCE 10, 15: identical
    assert e50(nce, f_trk).nce == pytest.approx(15.821, abs=1e-3)
    # the spectra: tracked fragments per precursor, API over the raw-file table
    ratio = (trk.to_numpy() / i) / ((1 - raw) / raw)
    assert np.round(ratio, 3).tolist() == [0.963, 0.993, 1.021, 1.037, 1.003, 1.015]
    assert np.abs(ratio - 1).max() < 0.04
    assert np.abs(f - raw).max() < 0.04 / 4
    r = _e50_row(res, 137.13, "C10H17+")
    want = _raw_e50(RAW_B_137).nce
    assert r["flag"] == "interpolated" and abs(r["e50_nce"] - want) < (0.04 / 4) / 0.0169
    assert r["e50_nce"] == pytest.approx(15.818, abs=1e-3)
    assert math.isnan(r["e50_v"]) and math.isnan(r["e50_ecm"]) and "volts unknown" in r["note"]
    # the precursor 2.3 mDa above the two-decimal key is the target
    assert w.precursor_mz == pytest.approx(137.13227, abs=1e-5)
    # with an isolation width the same window reads the same (no other ion named)
    res2 = AN.analyze_run(_tables("ms2_monoterpene_b", iso=0.4), _ions("C10H17+"))
    assert res2.e50.equals(res.e50) and res2.params["isolation_width_used"] == 0.4


#: (parent, ion) -> (flag, E50 NCE or None) of F, the 2026-09-10 file
NH4_E50 = {
    # C7H6NS+ (benzothiazole H+) barely fragments: F 0.73 at NCE 91
    (136.0213, "C7H6NS+"): ("above_range", None),
    # the dithiocarbamate co-isolate is spent by NCE 41 (-> C3H6NS+, -CH4S)
    (136.0213, "C4H10NS2+"): ("interpolated", 15.514),
    # 0.9 % of the window, seen at NCE 1 and next at 81 through a fragment:
    # no straight line across the hole
    (136.0213, "C8H10NO+"): ("gap", None),
    # two energies, F 0.67 -> 0.55
    (137.0572, "C4H10^NO4+"): ("above_range", None),
    (137.0572, "C6H18^NO2+"): ("interpolated", 2.735),
    (137.0572, "C5H14^NO3+"): ("below_range", None),
    # its precursor is in no MS2 spectrum; only its C9H11+ (-^NH3) at NCE 1
    (137.0572, "C9H14^N+"): ("single_energy", None),
    (161.096, "C11H13O+"): ("interpolated", 35.338),
    (179.1069, "C11H15O2+"): ("interpolated", 21.106),
    (179.1069, "C11H16^NO+"): ("interpolated", 6.144),
    # the ammonium adducts are half gone at the lowest step
    (197.13, "C11H18^NO2+"): ("below_range", None),
    (213.1612, "C12H22^NO2+"): ("below_range", None),
    (315.106, "C8H28^NO4Si4+"): ("interpolated", 1.777),
}


def test_exploratory_file_e50s(nh4):
    assert [w.status for w in nh4.windows] == ["analysed"] * 7
    assert [w.target for w in nh4.windows] == ["C7H6NS+", "C4H10^NO4+", "C11H13O+", "C11H15O2+",
                                               "C11H18^NO2+", "C12H22^NO2+", "C8H28^NO4Si4+"]
    for (parent, ion), (flag, val) in NH4_E50.items():
        r = _e50_row(nh4, parent, ion)
        assert r["flag"] == flag, (parent, ion, r.to_dict())
        if val is None:
            assert math.isnan(r["e50_nce"])
        else:
            assert r["e50_nce"] == pytest.approx(val, abs=1e-3), (parent, ion)
            assert math.isfinite(r["e50_v"]) and math.isfinite(r["e50_ecm"])
    # the gap curve of C8H10NO+ (seen at NCE 1 and 81 only) is monotonic: the
    # unobserved energies between are not a rise
    for curve in ("F", "F_lo"):
        r = _e50_row(nh4, 136.0213, "C8H10NO+", curve)
        assert (r["flag"], r["monotonic"], r["note"]) == ("gap", True, ""), curve
    # every ion of every window has its three curves
    assert len(nh4.e50) == 3 * sum(1 + len(w.co_isolates) for w in nh4.windows)
    # the 0.8 % co-isolate C12H21O2+ of the 197.13 window holds 1.8 % of the
    # adduct's in-source fragments by survivor share, and its own precursor
    # barely moves: its "products" grow 2-6x what it lost -- flagged, said
    bd = nh4.breakdown
    c12 = bd[(bd["parent_mz"] == 197.13) & (bd["ion"] == "C12H21O2+")]
    assert list(c12["branching_flag"]) == [False, True, True, True, True]
    assert c12["branching_sum"].iloc[1:].min() > 2.0
    w197 = [w for w in nh4.windows if w.parent_mz == 197.13][0]
    assert any(n.startswith("branching of C12H21O2+ sums to 2.037 at NCE 3") for n in w197.notes)
    assert not bd.loc[bd["ion"] == "C8H28^NO4Si4+", "branching_flag"].any()


def test_exploratory_file_calls(nh4):
    """Composition calls, and the in-source shared fragments left ambiguous.
    88.0215 C3H6NS+ is the co-isolated C4H10NS2+'s (-CH4S): no sub-formula of
    C7H6NS+ makes it. C9H11+ in the 179.107 window: the intact test would hand
    it to the 0.55 % co-isolate C11H16^NO+ (at its NCE 5 onset C11H15O2+ is
    still 98.7 % of its NCE 1 fraction), but by NCE 10 it holds more than that
    co-isolate ever held: the guard withdraws the call."""
    fr = nh4.fragments.set_index(["parent_mz", "fragment"])
    for (p, frag), (call, basis) in {
            (136.0213, "C3H6NS+"): ("C4H10NS2+", "composition"),
            (136.0213, "C6H5S+"): ("C7H6NS+", "composition"),
            (137.0572, "C4H7O4+"): ("C4H10^NO4+", "composition"),
            (137.0572, "C4H8^NO3+"): ("ambiguous", "ambiguous"),
            (161.096, "C2H3O+"): ("C11H13O+", "composition"),
            (179.1069, "C11H13O+"): ("ambiguous", "ambiguous"),
            (179.1069, "C9H11+"): ("ambiguous", "ambiguous"),
            (197.13, "C11H16^NO+"): ("C11H18^NO2+", "composition"),
            (213.1612, "C12H15+"): ("C12H22^NO2+", "composition"),
            (315.106, "C8H25O4Si4+"): ("C8H28^NO4Si4+", "composition")}.items():
        r = fr.loc[(p, frag)]
        assert (r["call"], r["basis"]) == (call, basis), (p, frag)
    c9 = fr.loc[(179.1069, "C9H11+")]
    # the first energy past the bound is NCE 10 (the co-isolate has no
    # fragment of its own at NCE 1: its bound is its precursor, 0.55 %)
    assert c9["test_note"] == (
        "intact test withdrawn: it left C11H16^NO+, but the fragment's TIC fraction grew by "
        "0.008291 at 179.1069@hcd10.00, more than the 0.005528 C11H16^NO+ held at the lowest "
        "energy (its precursor and the fragments only it can make)")
    assert (c9["onset_nce"], c9["in_source"], c9["excluded"]) == (5.0, False, "")
    assert _weights(c9["weights"])["C11H15O2+"] > 0.99
    c6 = fr.loc[(136.0213, "C6H5S+")]
    assert (c6["onset_nce"], c6["max_frac_nce"]) == (61.0, 91.0)
    assert c6["max_frac"] == pytest.approx(0.184269, abs=1e-6)
    # max_frac is a FRACTION of the spectrum's TIC (7433 counts of C3H6NS+ at
    # NCE 41 are a smaller share than 7298 of a smaller TIC at NCE 61)
    c3 = fr.loc[(136.0213, "C3H6NS+")]
    assert (c3["max_frac"], c3["max_frac_nce"]) == (pytest.approx(0.159106, abs=1e-6), 61.0)
    assert (fr["max_frac"] <= 1).all() and (fr["max_frac"] > 0).all()
    # a shared fragment's weights are the LOWEST-energy survivors: C7H6NS+
    # grows with energy, so its largest intensity would read 0.872204
    assert _weights(fr.loc[(136.0213, "C2H3NS+"), "weights"]) == \
        pytest.approx({"C7H6NS+": 0.87152, "C4H10NS2+": 0.12848}, abs=1e-6)
    assert fr.loc[(136.0213, "C2H3NS+"), "weight_source"] == "ms2 survivors"
    assert c6["onset_v"] == pytest.approx(14.58)
    assert c6["mz_theo"] == pytest.approx(_mz("C6H5S+"))
    # C12H15+ (-^NH3 -2 H2O, three pieces) appears from NCE 3 only
    assert fr.loc[(213.1612, "C12H15+"), "onset_nce"] == 3.0
    # the losses follow the candidates, in window order -- even where the
    # co-isolate reads the window's first peak (C4H10NS2+ in the 136.02 window,
    # C5H14^NO3+ in the 137.06 one)
    assert tuple(fr.loc[(136.0213, "C2H3NS+"), ["candidates", "losses"]]) == \
        ("C7H6NS+;C4H10NS2+", "C7H6NS+: C5H3; C4H10NS2+: H2S + C2H5")
    assert tuple(fr.loc[(137.0572, "C4H8^NO3+"), ["candidates", "losses"]]) == \
        ("C4H10^NO4+;C5H14^NO3+", "C4H10^NO4+: H2O; C5H14^NO3+: H2 + CH4")
    for r in nh4.fragments.itertuples():
        assert [x.split(": ")[0] for x in r.losses.split("; ")] == r.candidates.split(";")
    # no reading anywhere needed a fourth piece
    assert (nh4.readings["loss_pieces"] <= 3).all() and (nh4.readings["ladder_support"] == "").all()


def test_branching_is_nan_where_the_ion_lost_nothing(nh4):
    """No branching for an ion that lost nothing -- or GAINED: C7H6NS+ (136.02,
    intact to NCE 91 while its co-isolates fragment: 12 rows) and C11H13O+ at
    NCE 11 (7 rows). NaN, never a negative ratio; and their sums are NaN. A
    NaN sum is not a pass: where such an ion's products GREW, the invariant
    is broken worst of all, and the flag says so (`branching_breach`):
    C11H13O+ at NCE 11 (products +0.0205 of the TIC, its own fraction
    +0.0020) and C7H6NS+ at NCE 61 (+0.0123, +0.0067) -- each a larger excess
    than the flagged 1.059 of C7H6NS+ at NCE 81. An energy where the ion is
    unobserved -- no precursor and no fragment it can make, F's 0/0 -- has
    no branching, gain or loss at all (9 rows: the 0.9 % co-isolate C8H10NO+
    of 136.02 between NCE 1 and 81, and C9H14^N+ of 137.06 at one energy),
    where it once read a loss of everything it held."""
    b = nh4.branching
    assert (b["ion_loss"] < 0).sum() == 19 and (b["ion_loss"] == 0).sum() == 0
    hole = b[b["ion_loss"].isna()]
    assert hole[["frag_gain", "branching"]].isna().all().all()
    assert {(round(p, 4), i): int(n) for (p, i), n in
            hole.groupby(["parent_mz", "ion"]).size().items()} == \
        {(136.0213, "C8H10NO+"): 8, (137.0572, "C9H14^N+"): 1}
    assert b.loc[b["ion_loss"] <= 0, "branching"].isna().all()
    assert np.isfinite(b.loc[b["ion_loss"] > 0, "branching"]).all()
    gained = b[b["ion_loss"] < 0].groupby(["parent_mz", "ion"]).size()
    assert {(round(p, 4), i): int(n) for (p, i), n in gained.items()} == \
        {(136.0213, "C7H6NS+"): 12, (161.096, "C11H13O+"): 7}
    bd = nh4.breakdown
    r = bd[(bd["parent_mz"] == 161.096) & (bd["ion"] == "C11H13O+") & (bd["nce"] == 11)]
    assert len(r) == 1 and math.isnan(r["branching_sum"].iloc[0])
    # every (window, energy, ion) whose loss is <= 0 has a NaN sum
    lost = b.groupby(["parent_mz", "spectrum", "ion"])["ion_loss"].first()
    keyed = bd.set_index(["parent_mz", "spectrum", "ion"])["branching_sum"]
    assert keyed.reindex(lost[lost <= 0].index).isna().all()
    assert keyed.reindex(lost[lost > 0].index).notna().all()
    # ... and the flag reads the gain, not the NaN ratio: flagged exactly where
    # the products grew while the ion lost nothing
    gain = b.groupby(["parent_mz", "spectrum", "ion"])["frag_gain"].sum()
    flag = bd.set_index(["parent_mz", "spectrum", "ion"])["branching_flag"]
    breach = (lost <= 0) & (gain > 0)
    assert {(round(p, 4), s, i) for p, s, i in breach[breach].index} == {
        (161.096, "161.096@hcd11.00", "C11H13O+"), (136.0213, "136.0213@hcd61.00", "C7H6NS+")}
    assert flag.reindex(breach[breach].index).all()
    assert not flag.reindex(breach[(lost <= 0) & ~breach].index).any()
    assert r["branching_flag"].iloc[0]
    w161 = [w for w in nh4.windows if w.parent_mz == 161.096][0]
    assert any(n.startswith("branching of C11H13O+ sums to 1.302 at NCE 21, 1.074 at NCE 41 "
                            "(> 1.05); no ratio at NCE 11 (its products +0.0205 of the TIC, its "
                            "own fraction +0.0020): it lost nothing there") for n in w161.notes)


def test_a_breach_where_the_ion_lost_nothing_is_flagged():
    """A co-isolate whose TIC fraction never moves (30 counts of a constant
    1080) takes its survivor share (3 %) of the target's growing in-source
    fragment. It never fragments, yet its products grow: its loss is exactly
    0, its branching ratio NaN -- and that is flagged at every step, with a
    window note, where a NaN once let it through unflagged and unsaid. The
    target's sums (0.97) are not."""
    steps = [(e, {X: x, Y: 30.0, "C7H11+": 1050.0 - x})
             for e, x in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    r = res.fragments.set_index("fragment").loc["C7H11+"]
    assert (r["call"], r["weight_source"]) == ("ambiguous", "ms2 survivors")
    _, _, gy = _curve(res, w.parent_mz, Y)
    assert np.allclose(gy["frac"], 30 / 1080) and gy["branching_sum"].isna().all()
    assert list(gy["branching_flag"]) == [False, True, True, True]
    _, _, gx = _curve(res, w.parent_mz, X)
    assert np.allclose(gx["branching_sum"].iloc[1:], 1000 / 1030)
    assert not gx["branching_flag"].any()
    (note,) = w.notes
    assert note.startswith("branching of C10H19O+ has no ratio at NCE 10 (its products +0.0081 "
                           "of the TIC, its own fraction +0.0000), NCE 20")
    # the rule, alone: above the flag where the ion lost something; any
    # growth where it lost nothing or gained; nothing for a shrinking product
    for gain, loss, want in ((1.06, 1.0, True), (1.04, 1.0, False), (0.0205, -0.002, True),
                             (1e-9, 0.0, True), (0.0, -0.1, False), (-0.01, 0.0, False),
                             (-0.5, -0.1, False), (NAN, 0.0, False), (0.1, NAN, False)):
        assert AT.branching_breach(gain, loss) is want, (gain, loss)


def test_offsets_targets_and_the_ion_budget(nh4, mta):
    by = {w.parent_mz: w for w in nh4.windows + mta.windows}
    # one constant-ppm offset per window, from its target's located precursor
    assert by[136.0213].offset_ppm == pytest.approx(
        SF.mass_offset([(by[136.0213].precursor_mz, "C7H6NS+")]).offset_ppm)
    assert by[137.1321].offset_ppm == pytest.approx(-1.065, abs=1e-3)
    assert {round(w.offset_ppm, 2) for w in nh4.windows} <= {-0.13, -0.04, 0.24, 0.66, 0.11,
                                                            -0.1, 0.03}
    # the readings of a window are judged on that one axis, and the target's
    # precursor reads 0 ppm in the spectrum it was located in (one anchor)
    rd = nh4.readings[np.isclose(nh4.readings["parent_mz"], 136.0213)]
    assert np.allclose(rd["mz_cal"], SF.apply_offset(rd["mz"], by[136.0213].offset_ppm),
                       rtol=0, atol=1e-9)
    for w in nh4.windows + mta.windows:
        r = w.readings
        pre = r[(r["formula"] == w.target) & r["is_precursor"]
                & np.isclose(r["mz"], w.precursor_mz, rtol=0, atol=1e-9)]
        assert len(pre) and np.abs(pre["ppm"]).max() < 1e-6, w.parent_mz
    # the co-isolates: the named ions inside +-(0.2 + 0.005) Da of the key
    assert by[137.0572].co_isolates == ["C8H9O2+", "C5H14^NO3+", "C9H14^N+", "C6H18^NO2+"]
    assert by[136.0213].co_isolates == ["C4H10NS2+", "C8H10NO+"]
    assert by[161.096].co_isolates == []
    # ion budget: explained + unexplained = 1, per spectrum, from the readings
    bd = nh4.breakdown
    assert np.allclose(bd["budget_explained"] + bd["budget_unexplained"], 1.0)
    one = bd[bd["spectrum"] == "137.0572@hcd1.00"].iloc[0]
    rd = nh4.readings[nh4.readings["spectrum"] == "137.0572@hcd1.00"]
    peaks = rd.drop_duplicates("peak")
    explained = peaks.loc[peaks["rank"] > 0, "intensity"].sum()
    assert one["budget_explained"] == pytest.approx(explained / peaks["intensity"].sum())
    assert one["tic"] == pytest.approx(peaks["intensity"].sum())
    assert one["n_unexplained"] == int((peaks["rank"] == 0).sum()) == 3


def test_tables_have_their_contract(nh4):
    assert list(nh4.readings.columns) == list(RD.READINGS_COLUMNS)
    # the spectrum's four, assign_fragments', then ladder_support and harmonic
    assert list(nh4.readings.columns[4:-2]) == list(SF.OUTPUT_COLUMNS)
    assert list(nh4.readings.columns[-2:]) == ["ladder_support", "harmonic"]
    for df, spec in ((nh4.readings, RD.READINGS_COLUMNS), (nh4.breakdown, AN.BREAKDOWN_COLUMNS),
                     (nh4.e50, AN.E50_COLUMNS), (nh4.fragments, AN.FRAGMENT_COLUMNS),
                     (nh4.branching, AN.BRANCHING_COLUMNS)):
        assert {c: str(t) for c, t in df.dtypes.items()} == spec
    # every peak of every spectrum is in the readings
    n = nh4.readings.groupby("spectrum")["peak"].nunique()
    sp = _tables("ms2_nh4_exploratory").spectra.groupby("spectrum").size()
    assert (n == sp.reindex(n.index)).all() and len(n) == 29


def test_every_table_carries_the_spectrum_volts(nh4, mta):
    """`hcd_v` of the readings, branching and breakdown tables is the parents
    table's for that spectrum -- never the NCE -- and NaN for standard B,
    whose summary was refused."""
    b = AN.analyze_run(_tables("ms2_monoterpene_b"), _ions("C10H17+"))
    for res, stem in ((nh4, "ms2_nh4_exploratory"), (mta, "ms2_monoterpene_a"),
                      (b, "ms2_monoterpene_b")):
        volts = _tables(stem).parents.set_index("spectrum")["hcd_v"]
        for name in ("readings", "branching", "breakdown"):
            df = getattr(res, name)
            assert len(df), (stem, name)
            want = volts.reindex(df["spectrum"]).to_numpy()
            got = df["hcd_v"].to_numpy()
            assert np.array_equal(np.isnan(got), np.isnan(want)), (stem, name)
            assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=0, atol=0), \
                (stem, name)
    assert b.readings["hcd_v"].isna().all() and b.branching["hcd_v"].isna().all()
    assert nh4.readings["hcd_v"].notna().all()
    assert not np.allclose(nh4.branching["hcd_v"], nh4.branching["nce"])


def test_e_cm_of_every_ion_uses_its_own_mz(nh4, mta):
    """E_cm = V * m_gas / (m_gas + m/z) with each window ion's OWN m/z -- a
    co-isolate's too (12 of the 32 finite rows of these two files; C8H10NO+'s
    S, unobserved between NCE 1 and 81, is a gap like its F)."""
    n_co = 0
    for res in (nh4, mta):
        e = res.e50[np.isfinite(res.e50["e50_v"])]
        assert len(e)
        assert np.allclose(e["e50_ecm"], e["e50_v"] * 28.0061 / (28.0061 + e["ion_mz"]),
                           rtol=1e-12, atol=0)
        assert np.allclose(e["ion_mz"], [_mz(i) for i in e["ion"]], rtol=0, atol=1e-9)
        n_co += int((e["role"] == "co-isolate").sum())
    assert n_co == 12
    r = _e50_row(mta, 155.1429, "C10H20^N+")
    assert r["role"] == "co-isolate"
    assert r["e50_ecm"] == pytest.approx(r["e50_v"] * 28.0061 / (28.0061 + _mz("C10H20^N+")))


def _drop_group(activation, parent=137.1321):
    """Standard A's run with one summary group removed: that spectrum keeps
    its NCE (from the key) but has no volts."""
    summ = json.loads(json.dumps(_load("ms2_monoterpene_a_summary.json")))
    summ["groups"] = [g for g in summ["groups"]
                      if not (g["parent_peak_mz"] == parent and g["activation"] == activation)]
    return _extract(_load("ms2_monoterpene_a_centroids.json"), summ, "ms2_monoterpene_a_ms1.json")


def test_volts_note_follows_the_volts_the_e50_used():
    """'volts unknown' only when no spectrum has volts. A missing group at
    NCE 3 leaves the crossing's brackets (NCE 9 and 20) with volts: E50 6.327 V
    stands, and the note counts the spectra with volts. A missing group at
    NCE 9 (a bracket) leaves no volts E50, and the note says where."""
    res = AN.analyze_run(_drop_group("hcd3.00"), _ions(*MT_IONS))
    for curve in ("F", "F_lo"):
        r = _e50_row(res, 137.1321, "C10H17+", curve)
        assert r["e50_v"] == pytest.approx(6.327, abs=1e-3) and math.isfinite(r["e50_ecm"])
        assert r["note"] == "volts known for 4 of 5 spectra", curve
    assert _e50_row(res, 137.1321, "C10H17+", "S")["note"] == \
        f"{AN.S_STEADY}; volts known for 4 of 5 spectra"
    w = [x for x in res.windows if x.parent_mz == 137.1321][0]
    assert w.manifest()["volts_known"] is False                  # not every spectrum's
    # every row of a fully-known window has no volts note
    assert not res.e50.loc[res.e50["parent_mz"] != 137.1321, "note"].str.contains("volts").any()
    res = AN.analyze_run(_drop_group("hcd9.00"), _ions(*MT_IONS))
    r = _e50_row(res, 137.1321, "C10H17+")
    assert r["e50_nce"] == pytest.approx(13.891, abs=1e-3) and math.isnan(r["e50_v"])
    assert r["note"] == "volts unknown at the crossing (known for 4 of 5 spectra)"
    # no volts at all (standard B, refused summary): 'volts unknown'
    b = AN.analyze_run(_tables("ms2_monoterpene_b"), _ions("C10H17+"))
    assert _e50_row(b, 137.13, "C10H17+")["note"] == "volts unknown"
    assert b.windows[0].manifest()["volts_known"] is False


def test_empty_spectra_are_left_out_and_an_empty_window_skipped():
    """`ms2 extract` writes a parent row with n_peaks 0 for a spectrum the
    server returned without peaks. One such energy in a window is left out of
    the breakdown (said) and the rest is analysed as before; a window of
    nothing but empty spectra is skipped with its own reason."""
    t = _synthetic(round(_mz(X), 4), [(1, {}), (10, {})])
    assert list(t.parents["n_peaks"]) == [0, 0] and t.parents["precursor_mz"].isna().all()
    (w,) = AN.analyze_run(t, _ions(X)).windows
    assert (w.status, w.reason, w.notes) == ("skipped", AN.SKIP_NO_PEAKS,
                                            ["skipped: no spectrum with peaks"])
    cent = json.loads(json.dumps(_load("ms2_monoterpene_a_centroids.json")))
    for k, v in cent["137.1321@hcd5.00"].items():
        if isinstance(v, list):
            cent["137.1321@hcd5.00"][k] = []
    t = _extract(cent, _load("ms2_monoterpene_a_summary.json"), "ms2_monoterpene_a_ms1.json")
    res = AN.analyze_run(t, _ions(*MT_IONS))
    w = [x for x in res.windows if x.parent_mz == 137.1321][0]
    assert w.status == "analysed" and w.notes[0] == "1 spectrum(s) with no peaks left out"
    assert sorted(set(w.breakdown["nce"])) == [1.0, 3.0, 9.0, 20.0]
    assert _e50_row(res, 137.1321, "C10H17+")["e50_nce"] == pytest.approx(13.891, abs=1e-3)


def test_a_peak_two_window_ions_read_as_their_precursor_is_said():
    """Two co-isolates 10.8 mDa apart (C10H20^N+ and C10H21N+: ^N against
    NH) are not aliases at a 6 mDa tolerance, but a peak midway between them
    reads as the precursor of both. It goes to the first in window order, and
    the window says so -- and the other, left with I = 0 while it takes an
    equal share of a fragment, is flagged."""
    Z = "C10H21N+"
    mid = (_mz(X) + _mz(Z)) / 2
    steps = [(1, {Y: 1000.0, mid: 500.0}), (10, {Y: 600.0, mid: 300.0, "C10H17+": 400.0})]
    res = AN.analyze_run(_synthetic(round(_mz(Y), 4), steps), _ions(Y, X, Z), tol_mda=6.0)
    (w,) = res.windows
    assert (w.target, w.co_isolates, w.left_out) == (Y, [X, Z], [])
    assert w.notes[0] == ("2 peak(s) read as the precursor of several window ions (C10H20^N+;"
                          "C10H21N+): each is given to the first of them in window order")
    _, _, gx = _curve(res, w.parent_mz, X)
    _, _, gz = _curve(res, w.parent_mz, Z)
    assert list(gx["intensity"]) == [500.0, 300.0] and list(gz["intensity"]) == [0.0, 0.0]
    assert list(gz["branching_flag"]) == [False, True]


def test_deterministic(nh4):
    again = AN.analyze_run(_tables("ms2_nh4_exploratory"), _ions(*NH4_IONS))
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        assert getattr(again, name).equals(getattr(nh4, name)), name


# ------------------------------------------------------- the ladder gate ----
def test_no_peak_of_our_files_needs_a_fourth_piece():
    """On the 2026-09-10 and 2026-09-11 files the ladder gate, the strict
    k_max 3 and even the ungated k_max 4 read exactly the same table: no peak
    needs a fourth piece at all."""
    for stem, ions in (("ms2_nh4_exploratory", NH4_IONS), ("ms2_monoterpene_a", MT_IONS),
                       ("ms2_monoterpene_b", ("C10H17+",))):
        t, ii = _tables(stem), _ions(*ions)
        gated = AN.analyze_run(t, ii).readings
        strict = AN.analyze_run(t, ii, ladder=False).readings
        k4 = AN.analyze_run(t, ii, k_max=4).readings
        assert gated.equals(strict) and k4.equals(strict), stem


HEX = "C6H16NO6+"            # hexose + NH4+ (14N)
#: the chain: -NH3, then water step by step; -NH3 -3 H2O (C6H7O3+, loss H9NO3,
#: DBE -3) is the first step that needs four molecules
CHAIN = ("C6H16NO6+", "C6H13O6+", "C6H11O5+", "C6H9O4+", "C6H7O3+")


def _spec(nce, ions, key="198.0972"):
    meta = {"spectrum": f"{key}@hcd{nce:.2f}", "parent_mz": float(key), "nce": float(nce),
            "hcd_v": 4.0 + 0.2 * nce}
    peaks = pd.DataFrame({"mz": [_mz(i) if isinstance(i, str) else i for i in ions],
                          "intensity": np.arange(len(ions), 0, -1, dtype=float) * 100})
    return meta, peaks


def _reading(df, spectrum, formula, parent=None):
    r = df[(df["spectrum"] == spectrum) & (df["formula"] == formula)
           & ((df["parent"] == parent) if parent else True)]
    return r.iloc[0] if len(r) else None


def test_ladder_admits_the_fourth_step_only_when_the_third_is_read():
    """The hexose [M+NH4]+ chain. C6H7O3+ (-NH3 -3 H2O) needs four pieces;
    it is read when the same window ion's previous step is read at some
    energy: C6H9O4+ (-NH3 -2 H2O, loss H7NO2 = this loss minus H2O) -- or
    C6H10NO3+ (-3 H2O, this loss minus NH3)."""
    lone = [_spec(1, CHAIN[:3]), _spec(10, [CHAIN[0], CHAIN[4]])]
    df = RD.read_window(lone, {"hex": HEX})
    r = df[df["spectrum"] == "198.0972@hcd10.00"]
    p127 = r[np.isclose(r["mz"], _mz("C6H7O3+"))]
    assert len(p127) == 1 and p127.iloc[0]["rank"] == 0          # unexplained
    assert (p127.iloc[0]["formula"], p127.iloc[0]["n_parents"]) == ("", 0)
    # the step before it is the ladder, at a lower, the same or a higher energy
    for steps in ([_spec(1, CHAIN[:4]), _spec(10, [CHAIN[0], CHAIN[4]])],
                  [_spec(1, CHAIN[:3]), _spec(10, [CHAIN[0], CHAIN[3], CHAIN[4]])],
                  [_spec(1, CHAIN[:3]), _spec(10, [CHAIN[0], CHAIN[4]]),
                   _spec(20, [CHAIN[0], CHAIN[3]])]):
        df = RD.read_window(steps, {"hex": HEX})
        got = _reading(df, "198.0972@hcd10.00", "C6H7O3+")
        assert got is not None
        assert (got["loss"], got["loss_name"], got["loss_pieces"]) == \
            ("H9NO3", "H2O + H2O + H2O + H3N", 4)
        assert (got["ladder_support"], got["rank"], got["n_candidates"]) == ("H7NO2", 1, 1)
    # -3 H2O keeping the ammonium (C6H10NO3+) is the other ladder
    df = RD.read_window([_spec(1, [CHAIN[0], "C6H10NO3+"]), _spec(10, [CHAIN[0], CHAIN[4]])],
                        {"hex": HEX})
    assert _reading(df, "198.0972@hcd10.00", "C6H7O3+")["ladder_support"] == "H6O3"
    # both previous steps read: the support is the FIRST piece, in the
    # decomposition's order (H2O + H2O + H2O + H3N), that finds one -- H2O,
    # whose step is H7NO2 -- not the last (H3N -> H6O3)
    df = RD.read_window([_spec(1, [CHAIN[0], CHAIN[3], "C6H10NO3+"]),
                         _spec(10, [CHAIN[0], CHAIN[4]])], {"hex": HEX})
    assert {r["loss"] for _, r in df[df["spectrum"] == "198.0972@hcd1.00"].iterrows()
            if r["rank"] == 1 and not r["is_precursor"]} == {"H7NO2", "H6O3"}
    got = _reading(df, "198.0972@hcd10.00", "C6H7O3+")
    assert (got["loss_name"], got["ladder_support"]) == ("H2O + H2O + H2O + H3N", "H7NO2")
    # strict k_max 3 never reads it; an explicit k_max 4 always does, ungated
    steps = [_spec(1, CHAIN[:4]), _spec(10, [CHAIN[0], CHAIN[4]])]
    assert _reading(RD.read_window(steps, {"hex": HEX}, ladder=False), "198.0972@hcd10.00",
                    "C6H7O3+") is None
    k4 = _reading(RD.read_window(lone, {"hex": HEX}, k_max=4), "198.0972@hcd10.00", "C6H7O3+")
    assert k4 is not None and k4["ladder_support"] == ""
    # the support is each window ion's own. C7H11O4+ = C7H18NO6+ - H7NO2 is
    # read by the other ion only (the hexose adduct holds six C): it does not
    # admit the adduct's four-piece C6H7O3+ ... but it does admit the other
    # ion's own, whose loss CH11NO3 = H2O + H2O + H3N + CH4O is H7NO2 + CH4O
    other = RD.read_window([_spec(1, [CHAIN[0], "C7H11O4+"]), _spec(10, [CHAIN[0], CHAIN[4]])],
                           {"other": "C7H18NO6+", "hex": HEX})
    o = _reading(other, "198.0972@hcd1.00", "C7H11O4+")
    assert (o["parent"], o["loss"], o["n_parents"]) == ("other", "H7NO2", 1)
    assert _reading(other, "198.0972@hcd10.00", "C6H7O3+", "hex") is None
    mine = _reading(other, "198.0972@hcd10.00", "C6H7O3+", "other")
    assert (mine["loss_name"], mine["ladder_support"], mine["n_parents"]) == \
        ("H2O + H2O + H3N + CH4O", "H7NO2", 1)


def test_the_ladder_step_need_not_be_a_printed_piece():
    """A four-piece loss prints ONE decomposition -- fewest pieces, then fewest
    unnamed, then library order -- and it need not hold the piece the ladder
    took. C9H22NO6+ -> C8H11O+ loses CH11NO5, printed H3N + CH4O + H2O2 +
    H2O2 (no H2O: the decompositions with water need an unnamed piece);
    C8H13O2+ (loss CH9NO4, H2O + H2O + CH5NO2) is read, one H2O back. That
    step admits the reading: this loss minus ONE closed-shell library
    molecule is a read step. Testing only the printed pieces left C8H11O+
    unexplained whatever was read."""
    P = "C9H22NO6+"
    dec = SF.decompose_loss("CH11NO5", k_max=4)
    assert dec.name == "H3N + CH4O + H2O2 + H2O2" and "H2O" not in dec.pieces
    assert SF.decompose_loss("CH9NO4", k_max=3).name == "H2O + H2O + CH5NO2"
    key = f"{_mz(P):.4f}"
    for steps in ([_spec(1, [P, "C8H13O2+"], key), _spec(10, [P, "C8H11O+"], key)],
                  [_spec(10, [P, "C8H13O2+", "C8H11O+"], key)]):
        df = RD.read_window(steps, {"p": P})
        got = _reading(df, f"{key}@hcd10.00", "C8H11O+")
        assert got is not None
        assert (got["loss"], got["loss_pieces"], got["rank"], got["ladder_support"]) == \
            ("CH11NO5", 4, 1, "CH9NO4")
    # no step one molecule back: unexplained, as before
    df = RD.read_window([_spec(1, [P], key), _spec(10, [P, "C8H11O+"], key)], {"p": P})
    lone = df[(df["spectrum"] == f"{key}@hcd10.00") & np.isclose(df["mz"], _mz("C8H11O+"))]
    assert len(lone) == 1 and lone.iloc[0]["rank"] == 0
    # one molecule means one named closed-shell piece: not an unnamed
    # remainder, not a radical, not two
    v = RD._vec
    assert RD._one_molecule(v("H2O")) and RD._one_molecule(v("H3N"))
    assert not RD._one_molecule(v("CH5NO2"))                 # unnamed
    assert not RD._one_molecule(v("HO"))                     # a radical
    assert not RD._one_molecule(v("H4O2"))                   # two waters
    assert RD._ladder_step("CH11NO5", "H3N + CH4O + H2O2 + H2O2",
                           {tuple(v("CH9NO4").tolist())}) is not None
    assert RD._ladder_step("CH11NO5", "H3N + CH4O + H2O2 + H2O2",
                           {tuple(v("CH7NO3").tolist())}) is None       # two waters back


def test_a_printed_unnamed_or_radical_piece_is_no_ladder_step():
    """A printed piece is a step only when it is one named closed-shell
    molecule -- the rule for an unprinted one. The hexose [M+NH4]+ after its
    -3 H2O step C6H10NO3+ (loss H6O3) reads C5H5+ as H2O + H2O + H2O +
    CH5NO3 (an unnamed remainder), and after C6H9O4+ (loss H7NO2) reads
    C6H8O3+ as H2O + H2O + H3N + HO (OH*): taking the printed piece as the
    step kept both, by what the printer happened to list -- H8NO3 prints
    its OH*, while H8NO4 prints HO2* and the same step one OH* back was
    refused. Both are unexplained now, as when alone; ungated k_max 4 still
    reads them."""
    v = RD._vec
    assert RD._ladder_step("CH11NO6", "H2O + H2O + H2O + CH5NO3",
                           {tuple(v("H6O3").tolist())}) is None
    assert RD._ladder_step("H8NO3", "H2O + H2O + H3N + HO", {tuple(v("H7NO2").tolist())}) is None
    assert RD._ladder_step("H8NO4", "H2O + H2O + H3N + HO2",
                           {tuple(v("H7NO3").tolist())}) is None
    assert SF.decompose_loss("H8NO3", k_max=4).name == "H2O + H2O + H3N + HO"
    assert SF.decompose_loss("H8NO4", k_max=4).name == "H2O + H2O + H3N + HO2"
    # a printed named closed-shell piece still is one (H2O: the hexose ladder)
    assert SF._formula_of(RD._ladder_step("H9NO3", "H2O + H2O + H2O + H3N",
                                          {tuple(v("H7NO2").tolist())})) == "H7NO2"
    for step, frag, loss, name in (
            ("C6H10NO3+", "C5H5+", "CH11NO6", "H2O + H2O + H2O + CH5NO3"),
            ("C6H9O4+", "C6H8O3+", "H8NO3", "H2O + H2O + H3N + HO")):
        steps = [_spec(1, [HEX, step]), _spec(10, [HEX, frag])]
        df = RD.read_window(steps, {"hex": HEX})
        r = df[(df["spectrum"] == "198.0972@hcd10.00") & np.isclose(df["mz"], _mz(frag))]
        assert len(r) == 1 and r.iloc[0]["rank"] == 0 and r.iloc[0]["formula"] == "", frag
        k4 = _reading(RD.read_window(steps, {"hex": HEX}, k_max=4), "198.0972@hcd10.00", frag)
        assert (k4["loss"], k4["loss_name"], k4["loss_pieces"]) == (loss, name, 4)


def test_gate_modes():
    assert RD.gate_mode() == (RD.GATE_LADDER, 4)
    assert RD.gate_mode(ladder=False) == (RD.GATE_STRICT, 3)
    assert RD.gate_mode(3) == (RD.GATE_STRICT, 3)
    assert RD.gate_mode(4) == (RD.GATE_UNGATED, 4)
    assert RD.gate_mode(5, ladder=False) == (RD.GATE_UNGATED, 5)
    for bad in (0, 7, 2.5, True):
        with pytest.raises(ValueError):
            RD.gate_mode(bad)
    # the manifest's words: below 3 is stricter, not "more chance compositions"
    for k in (1, 2):
        note = AN.gate_note(*RD.gate_mode(k))
        assert "stricter than the default 3" in note and "more chance" not in note
    assert "1 piece is read" in AN.gate_note(RD.GATE_UNGATED, 1)
    assert "2 pieces is read" in AN.gate_note(RD.GATE_UNGATED, 2)
    assert "more chance compositions" in AN.gate_note(*RD.gate_mode(5))
    assert AN.gate_note(*RD.gate_mode(3)).startswith("strict k_max 3")


def _table(rows):
    """A readings table from (spectrum, peak, parent, formula, loss, loss_name,
    pieces, radical, rank) rows, the rest filled consistently."""
    out = []
    for spec, peak, parent, formula, loss, name, k, radical, rank in rows:
        out.append({"spectrum": spec, "parent_mz": 100.0, "nce": 1.0, "hcd_v": 4.0,
                    "peak": peak, "mz": 50.0 + peak, "mz_cal": 50.0 + peak, "intensity": 10.0,
                    "parent": parent, "parent_ion": parent, "formula": formula,
                    "parity": "EE" if formula else "", "dbe": 1.5 if formula else NAN,
                    "ppm": 0.0 if formula else NAN, "loss": loss, "loss_pieces": k,
                    "loss_name": name, "unnamed_pieces": 0, "radical_loss": radical,
                    "is_precursor": False, "rank": rank, "n_candidates": 0, "n_parents": 0,
                    "ladder_support": ""})
    df = pd.DataFrame(out)
    read = df["rank"] > 0
    df.loc[read, "n_candidates"] = df[read].groupby(["spectrum", "peak", "parent"])[
        "rank"].transform("size")
    df.loc[read, "n_parents"] = df[read].groupby(["spectrum", "peak"])["parent"].transform(
        "nunique")
    return RD._frame(df)


def test_an_unsupported_fourth_piece_is_dropped_and_the_peak_re_ranked():
    four = ("H2O + H2O + H2O + H3N", "H9NO3")
    df = _table([
        # peak 0: a closed-shell 4-piece reading outranks a radical 1-piece one
        ("s1", 0, "A", "C5H7O3+", four[1], four[0], 4, False, 1),
        ("s1", 0, "A", "C5H6O3+", "H10NO3", "H10NO3", 1, True, 2),
        # peak 1: read by A only as a 4-piece, by B as one piece
        ("s1", 1, "A", "C4H5O3+", four[1], four[0], 4, False, 1),
        ("s1", 1, "B", "C4H5O3+", "H2O", "H2O", 1, False, 1),
        # peak 2: its only reading is an unsupported 4-piece
        ("s1", 2, "A", "C3H3O3+", four[1], four[0], 4, False, 1),
        # peak 3: never read
        ("s1", 3, "", "", "", "", 0, False, 0),
    ])
    out = RD.ladder_gate(df)
    assert list(out.columns) == list(RD.READINGS_COLUMNS)
    p0 = out[out["peak"] == 0]
    assert len(p0) == 1 and (p0.iloc[0]["formula"], p0.iloc[0]["rank"],
                             p0.iloc[0]["n_candidates"], p0.iloc[0]["n_parents"]) == \
        ("C5H6O3+", 1, 1, 1)
    p1 = out[out["peak"] == 1]
    assert list(p1["parent"]) == ["B"] and p1.iloc[0]["n_parents"] == 1
    p2 = out[out["peak"] == 2]
    assert len(p2) == 1 and p2.iloc[0]["rank"] == 0 and p2.iloc[0]["formula"] == ""
    assert p2.iloc[0]["n_parents"] == 0 and math.isnan(p2.iloc[0]["ppm"])
    assert p2.iloc[0]["intensity"] == 10.0 and p2.iloc[0]["mz"] == 52.0
    assert list(out["peak"]) == [0, 1, 2, 3]                       # order kept
    # supported at some energy -> kept as it is, with its support
    sup = pd.concat([df, _table([("s2", 0, "A", "C5H9O4+", "H7NO2", "H2O + H2O + H3N", 3, False,
                                  1)])], ignore_index=True)
    out = RD.ladder_gate(sup)
    kept = out[(out["spectrum"] == "s1") & (out["peak"] == 0)]
    assert list(kept["formula"]) == ["C5H7O3+", "C5H6O3+"] and list(kept["rank"]) == [1, 2]
    assert kept.iloc[0]["ladder_support"] == "H7NO2"
    # the support is what k_max 3 would rank first: a <= 3-piece reading behind a
    # 4-piece one (here a closed 4-piece outranks a radical 3-piece) counts
    behind = pd.concat([df, _table([
        ("s2", 0, "A", "C4H3O2+", "H11NO4", "H2O + H2O + H2O + H5NO", 4, False, 1),
        ("s2", 0, "A", "C5H9O4+", "H7NO2", "H2O + H2O + H3N", 3, True, 2)])],
        ignore_index=True)
    out = RD.ladder_gate(behind)
    assert out[(out["spectrum"] == "s1") & (out["peak"] == 0)].iloc[0]["ladder_support"] == \
        "H7NO2"
    # ... and a 4-piece reading supports nothing (s2's H11NO4 is dropped)
    assert "C4H3O2+" not in set(out["formula"])
    # a support that is not rank 1 among the <= 3-piece readings does not count
    weak = pd.concat([df, _table([("s2", 0, "A", "C9H9+", "H2", "H2", 1, False, 1),
                                  ("s2", 0, "A", "C5H9O4+", "H7NO2", "H2O + H2O + H3N", 3, False,
                                   2)])], ignore_index=True)
    assert "C5H7O3+" not in set(RD.ladder_gate(weak)["formula"])


def test_the_gate_re_ranks_a_real_window():
    """At 20 ppm the ^NH4+ hexose's 113.0233 reads CH6^NO5+ (one piece,
    19.8 ppm) and C5H5O3+ (four pieces, 0 ppm); the gate drops the second
    and the first is alone."""
    steps = [_spec(1, [_mz("C6H16^NO6+"), 113.02332], key="199.0942")]
    k4 = RD.read_window(steps, {"h": "C6H16^NO6+"}, tol_ppm=20, k_max=4)
    gated = RD.read_window(steps, {"h": "C6H16^NO6+"}, tol_ppm=20)
    r4 = k4[k4["peak"] == 1]
    assert list(r4["formula"]) == ["CH6^NO5+", "C5H5O3+"] and list(r4["n_candidates"]) == [2, 2]
    rg = gated[gated["peak"] == 1]
    assert list(rg["formula"]) == ["CH6^NO5+"] and list(rg["n_candidates"]) == [1]
    assert gated.equals(RD.read_window(steps, {"h": "C6H16^NO6+"}, tol_ppm=20, ladder=False))


def test_a_drop_keeps_the_spectrum_and_peak_order():
    """`rerank` keeps the order of everything it does not drop: spectra in
    the order given (NCE 1, 2, 10 -- not the keys' text order 1, 10, 2), each
    spectrum's peaks in m/z order, the dropped peak in its place as one
    unexplained row; every other row is the ungated k_max 4 table's."""
    steps = [_spec(1, CHAIN[:3]), _spec(2, [CHAIN[0], CHAIN[1]]), _spec(10, [CHAIN[0], CHAIN[4]])]
    keys = ["198.0972@hcd1.00", "198.0972@hcd2.00", "198.0972@hcd10.00"]
    gated = RD.read_window(steps, {"hex": HEX})
    k4 = RD.read_window(steps, {"hex": HEX}, k_max=4)
    assert list(dict.fromkeys(gated["spectrum"])) == keys
    assert list(zip(gated["spectrum"], gated["peak"])) == list(zip(k4["spectrum"], k4["peak"]))
    dropped = (k4["loss_pieces"] == 4).to_numpy()
    assert dropped.sum() == 1 and gated.loc[dropped, "rank"].iloc[0] == 0
    assert gated[~dropped].reset_index(drop=True).equals(k4[~dropped].reset_index(drop=True))


def test_read_window_is_deterministic_and_typed():
    steps = [_spec(1, CHAIN[:4]), _spec(10, [CHAIN[0], CHAIN[4]])]
    a = RD.read_window(steps, {"hex": HEX})
    b = RD.read_window(list(steps), {"hex": HEX})
    assert a.equals(b)
    assert {c: str(t) for c, t in a.dtypes.items()} == RD.READINGS_COLUMNS
    assert list(a["spectrum"].unique()) == ["198.0972@hcd1.00", "198.0972@hcd10.00"]
    # one spectrum's rows are exactly assign_fragments' at k_max 4 (nothing to gate there)
    one = assign_fragments(steps[0][1], {"hex": HEX}, k_max=4)
    first = a[a["spectrum"] == "198.0972@hcd1.00"].iloc[:, 4:-2].reset_index(drop=True)
    assert first.equals(one)
    empty = RD.read_window([], {"hex": HEX})
    assert len(empty) == 0 and list(empty.columns) == list(RD.READINGS_COLUMNS)


#: glucosamine + 15NH4+: its losses hold N AND ^N. NCE 1 reads C6H9O4+ (loss
#: H8N^NO = H2O + H3N + H3^N, three pieces); NCE 10 the fourth step C6H7O3+
#: (H10N^NO2, four pieces), on that ladder.
_SEED_SCRIPT = r"""
import json
import pandas as pd
from peaky.ms2 import analyze as AN, readings as RD
g = "C6H17N^NO5+"
def spec(nce, ions):
    meta = {"spectrum": f"198.1102@hcd{nce:.2f}", "parent_mz": 198.1102, "nce": float(nce),
            "hcd_v": 4.0 + 0.2 * nce}
    return meta, pd.DataFrame({"mz": [AN.ion_mz(i) for i in ions], "intensity": [200.0, 100.0]})
df = RD.read_window([spec(1, [g, "C6H9O4+"]), spec(10, [g, "C6H7O3+"])], {"g": g})
r = df[df["formula"] == "C6H7O3+"]
print(json.dumps({"rows": r[["loss", "loss_pieces", "rank", "ladder_support"]].astype(str)
                  .values.tolist(), "minus": RD._minus("CH8N^NO3", "H2O"),
                  "n_unexplained": int((df["rank"] == 0).sum())}))
"""


def test_the_ladder_does_not_depend_on_the_hash_seed():
    """A loss holding both N and ^N: its previous step must be found whatever
    the interpreter's string-hash seed. Each seed is its own process (one
    process has one seed); 8 and 16 are seeds under which a set-ordered
    formula wrote ^N before N and the supported reading was dropped."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    outs = []
    for seed in ("0", "8", "16"):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
        p = subprocess.run([sys.executable, "-c", _SEED_SCRIPT], cwd=root, env=env,
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr
        outs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    assert all(o == outs[0] for o in outs), outs
    assert outs[0] == {"rows": [["H10N^NO2", "4", "1", "H8N^NO"]], "minus": "CH6N^NO2",
                       "n_unexplained": 0}
    # the ladder compares count vectors: a step written in another element order is the same step
    assert RD._minus("H10N^NO2", "H2O") == "H8N^NO" == SF._formula_of(RD._vec("H8^NNO"))


def test_re_ranking_is_within_each_window_ion():
    """After a drop, ranks are recomputed per (spectrum, peak, parent): a
    peak two window ions read keeps a rank 1 for each, and both stay its
    candidates."""
    four = ("H2O + H2O + H2O + H3N", "H9NO3")
    df = _table([
        # A: an unsupported 4-piece rank 1, then a 1-piece radical; B: one piece
        ("s1", 0, "A", "C5H7O3+", four[1], four[0], 4, False, 1),
        ("s1", 0, "A", "C5H6O3+", "H10NO3", "H10NO3", 1, True, 2),
        ("s1", 0, "B", "C5H7O3+", "H2O", "H2O", 1, False, 1)])
    out = RD.ladder_gate(df)
    assert list(out["parent"]) == ["A", "B"] and list(out["rank"]) == [1, 1]
    assert list(out["n_candidates"]) == [1, 1] and list(out["n_parents"]) == [2, 2]
    r = AT.peak_roles(out, ["A", "B"]).iloc[0]
    assert (r["role"], r["candidates"], r["key"]) == ("fragment", "A;B", "C5H6O3+ | C5H7O3+")
    # through read_window: the hexose's lone four-piece C6H7O3+ is dropped (the
    # table re-ranks), and C6H11O5+, which both window ions read, keeps both
    steps = [_spec(1, [HEX, CHAIN[2]]), _spec(10, [HEX, CHAIN[2], CHAIN[4]])]
    df = RD.read_window(steps, {"hex": HEX, "w": CHAIN[1]})
    assert "C6H7O3+" not in set(df.loc[df["parent"] == "hex", "formula"])
    for spec in ("198.0972@hcd1.00", "198.0972@hcd10.00"):
        r = df[(df["spectrum"] == spec) & (df["formula"] == CHAIN[2])]
        assert list(r["parent"]) == ["hex", "w"] and list(r["rank"]) == [1, 1]
        assert list(r["n_parents"]) == [2, 2]
        roles = AT.peak_roles(df[df["spectrum"] == spec], ["hex", "w"])
        assert roles.loc[np.isclose(roles["mz"], _mz(CHAIN[2])), "candidates"].iloc[0] == "hex;w"


#: the hexose run: NCE 1 holds the chain to its third step, NCE 10 the
#: precursor and the fourth step C6H7O3+ only
HEX_RUN = [(1, dict(zip(CHAIN[:4], (1000.0, 300.0, 200.0, 100.0)))),
           (10, {HEX: 500.0, CHAIN[4]: 300.0})]


def test_the_gate_mode_reaches_the_readings():
    """The three modes, end to end: gated and ungated read C6H7O3+ at NCE 10
    (with and without its support) and F there is 500 / 800; strict k_max 3
    does not read it, and F is 1."""
    t = _synthetic(round(_mz(HEX), 4), HEX_RUN)
    for kw, gate, frag, support, f10 in (({}, "ladder-k4", True, "H7NO2", 0.625),
                                         ({"ladder": False}, "strict-k3", False, None, 1.0),
                                         ({"k_max": 4}, "ungated", True, "", 0.625)):
        res = AN.analyze_run(t, _ions(HEX), **kw)
        assert res.params["gate"] == gate, kw
        rd = res.readings
        four = rd[rd["loss_pieces"] == 4]
        assert (CHAIN[4] in set(res.fragments["fragment"])) is frag, kw
        assert list(four["ladder_support"]) == ([support] if frag else []), kw
        _, f, _ = _curve(res, res.windows[0].parent_mz, HEX)
        assert f[1] == pytest.approx(f10), kw


# --------------------------------------------- synthetic mixture, branching ----
X, Y = "C10H20^N+", "C10H19O+"      # the fragile target and a robust co-isolate
NCE = (1, 10, 20, 30)
X_I = (1000.0, 700.0, 400.0, 100.0)
Y_I = (1000.0, 1000.0, 800.0, 400.0)


def _payload(parent, steps, *, width=0.4, volts=True, resolution=None):
    """A centroid answer + summary in the server's shapes, from (nce, {ion
    formula or m/z: intensity}) steps; volts 4 + 0.2 NCE; `resolution`: the
    resolving power every centroid reports (none by default)."""
    cent, groups = {}, []
    for nce, peaks in steps:
        act = f"hcd{nce:.2f}"
        cent[S.spectrum_key(parent, act)] = {
            "parent_peak_mz": parent, "activation": act,
            "mz": [_mz(k) if isinstance(k, str) else float(k) for k in peaks],
            "intensity": [float(v) for v in peaks.values()]}
        if resolution is not None:
            cent[S.spectrum_key(parent, act)]["resolution"] = [float(resolution)] * len(peaks)
        groups.append({"parent_peak_mz": parent, "activation": act,
                       "hcd_energy": [4.0 + 0.2 * nce] if volts else [], "scan_count": 5,
                       "t_min": 0.0, "t_max": 1.0})
    summ = {"parent_peaks": [parent], "groups": groups, "isolation_width": width,
            "ms1_scan_count": 1, "ms2_scan_count": 5 * len(steps)}
    return cent, summ


def _synthetic(parent, steps, **kw):
    cent, summ = _payload(parent, steps, **kw)
    return AN.RunTables(spectra=S.spectra_table("S", cent, summ),
                        parents=S.parents_table("S", cent, summ),
                        window=S._frame({}, S.WINDOW_COLUMNS), manifest={})


def _mixture():
    """X loses dX and makes C10H17+ (0.5 dX; Y could make it too, -H2O),
    C5H10^N+ (0.3 dX, X only) and C6H9+ (0.2 dX; Y could make it too). Y loses
    dY and makes C6H9O+ (Y only). C7H11+ (either) is in-source and constant;
    m/z 120.5 is read by nothing. The TIC is constant (2080)."""
    steps = []
    for e, x, y in zip(NCE, X_I, Y_I):
        dx, dy = 1000 - x, 1000 - y
        peaks = {X: x, Y: y, "C7H11+": 50.0, 120.5: 30.0}
        for f, v in (("C10H17+", 0.5 * dx), ("C5H10^N+", 0.3 * dx), ("C6H9+", 0.2 * dx),
                     ("C6H9O+", dy)):
            if v > 0:
                peaks[f] = v
        steps.append((e, peaks))
    return steps


def test_two_precursor_mixture_with_known_branching():
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), _mixture()), _ions(X, Y))
    (w,) = res.windows
    assert (w.target, w.co_isolates, w.status) == (X, [Y], "analysed")
    fr = res.fragments.set_index("fragment")
    # composition: one candidate each
    assert (fr.loc["C5H10^N+", "call"], fr.loc["C5H10^N+", "basis"]) == (X, "composition")
    assert (fr.loc["C6H9O+", "call"], fr.loc["C6H9O+", "basis"]) == (Y, "composition")
    assert fr.loc["C6H9O+", "onset_nce"] == 20.0
    # the largest TIC fraction and where: C10H17+ holds 0.5 dX = 450 of 2080 at NCE 30
    assert (fr.loc["C10H17+", "max_frac"], fr.loc["C10H17+", "max_frac_nce"]) == \
        (pytest.approx(450 / 2080), 30.0)
    assert fr.loc["C7H11+", "max_frac"] == pytest.approx(50 / 2080)
    # the intact test: Y is still 100 % of itself where C10H17+ / C6H9+ appear
    for f in ("C10H17+", "C6H9+"):
        assert (fr.loc[f, "call"], fr.loc[f, "basis"], fr.loc[f, "excluded"]) == \
            (X, "intact test", Y), f
        assert _weights(fr.loc[f, "weights"]) == {X: 1.0, Y: 0.0}
        assert fr.loc[f, "onset_nce"] == 10.0 and not fr.loc[f, "in_source"]
    # in-source: no test, shares of the lowest-energy survivors (1000 : 1000)
    z = fr.loc["C7H11+"]
    assert (z["call"], z["weight_source"], z["in_source"]) == ("ambiguous", "ms2 survivors", True)
    assert _weights(z["weights"]) == {X: 0.5, Y: 0.5}
    # survival: F = I / (I + its fragments + half the in-source one)
    _, fx, gx = _curve(res, w.parent_mz, X)
    _, fy, gy = _curve(res, w.parent_mz, Y)
    assert np.allclose(fx, np.array(X_I) / 1025.0)
    assert np.allclose(fy, np.array(Y_I) / 1025.0)
    # F_lo: every fragment the ion can make by composition, whole
    dx = 1000 - np.array(X_I)
    dy = 1000 - np.array(Y_I)
    assert np.allclose(gx["f_lo"], np.array(X_I) / 1050.0)
    assert np.allclose(gy["f_lo"], np.array(Y_I) / (np.array(Y_I) + dy + 0.7 * dx + 50))
    assert np.allclose(gx["s"], np.array(X_I) / 1000)
    # branching: X gives 0.5 + 0.3 + 0.2 of what it loses, Y all of it
    bx, by = gx["branching_sum"].to_numpy(), gy["branching_sum"].to_numpy()
    assert math.isnan(bx[0]) and np.allclose(bx[1:], 1.0)
    assert math.isnan(by[0]) and math.isnan(by[1]) and np.allclose(by[2:], 1.0)
    assert (res.breakdown["branching_sum"].dropna() <= 1 + 1e-9).all()
    assert not res.breakdown["branching_flag"].any()
    br = res.branching[(res.branching["ion"] == X) & (res.branching["nce"] == 20)]
    assert dict(zip(br["fragment"], br["branching"].round(9))) == \
        {"C10H17+": 0.5, "C5H10^N+": 0.3, "C6H9+": 0.2, "C7H11+": 0.0}
    # E50s, exact: X crosses 512.5 between NCE 10 (700) and 20 (400)
    ex, ey = _e50_row(res, w.parent_mz, X), _e50_row(res, w.parent_mz, Y)
    assert ex["e50_nce"] == pytest.approx(16.25) and ey["e50_nce"] == pytest.approx(27.1875)
    assert ex["e50_v"] == pytest.approx(4 + 0.2 * 16.25)
    assert ex["e50_ecm"] == pytest.approx((4 + 0.2 * 16.25) * 28.0061 / (28.0061 + _mz(X)))
    assert _e50_row(res, w.parent_mz, X, "F_lo")["e50_nce"] == pytest.approx(10 + 175 / 300 * 10)
    # the ion budget: m/z 120.5 is the only unexplained signal
    assert np.allclose(gx["budget_explained"], 2050 / 2080)
    assert (gx["n_unexplained"] == 1).all() and np.allclose(gx["tic"], 2080)


def test_the_fragile_ion_breaks_first_whichever_is_the_target():
    """The same mixture isolated at Y's m/z: Y is the target now, and every
    ion's E50 is what it was -- the ordering is the ions', not the roles'."""
    a = AN.analyze_run(_synthetic(round(_mz(X), 4), _mixture()), _ions(X, Y))
    b = AN.analyze_run(_synthetic(round(_mz(Y), 4), _mixture()), _ions(X, Y))
    assert (a.windows[0].target, b.windows[0].target) == (X, Y)
    for ion in (X, Y):
        for curve in ("F", "F_lo", "S"):
            ra = _e50_row(a, a.windows[0].parent_mz, ion, curve)
            rb = _e50_row(b, b.windows[0].parent_mz, ion, curve)
            assert ra["e50_nce"] == pytest.approx(rb["e50_nce"]), (ion, curve)
    for res in (a, b):
        p = res.windows[0].parent_mz
        for curve in ("F", "F_lo", "S"):
            assert _e50_row(res, p, X, curve)["e50_nce"] < _e50_row(res, p, Y, curve)["e50_nce"]


def test_the_guard_withdraws_a_call_the_ion_cannot_carry():
    """The 179.107 case, synthetic: a 1 % co-isolate loses what the fragment
    holds at its onset while the main ion is intact; then the fragment grows
    far past anything the co-isolate held."""
    main, co = "C11H15O2+", "C11H16^NO+"
    steps = [(1, {main: 10000.0, co: 100.0}),
             (5, {main: 10000.0, co: 50.0, "C9H11+": 50.0}),
             (10, {main: 8000.0, "C9H11+": 2150.0})]
    res = AN.analyze_run(_synthetic(round(_mz(main), 4), steps), _ions(main, co))
    r = res.fragments.set_index("fragment").loc["C9H11+"]
    assert (r["call"], r["basis"], r["excluded"]) == ("ambiguous", "ambiguous", "")
    assert r["test_note"].startswith(f"intact test withdrawn: it left {co}")
    assert _weights(r["weights"]) == pytest.approx({main: 10000 / 10100, co: 100 / 10100})
    # without the late growth the intact test's call stands
    res = AN.analyze_run(_synthetic(round(_mz(main), 4), steps[:2]), _ions(main, co))
    r = res.fragments.set_index("fragment").loc["C9H11+"]
    assert (r["call"], r["basis"], r["excluded"]) == (co, "intact test", main)
    # the edge: the co-isolate held 100 / 10100 of the TIC; a fragment that
    # reaches 1.04x that share keeps the call, 1.07x withdraws it
    for frag, kept in ((104.0, True), (107.0, False)):
        st = [(1, {main: 10000.0, co: 100.0}), (5, {main: 10000.0, "C9H11+": frag})]
        r = AN.analyze_run(_synthetic(round(_mz(main), 4), st), _ions(main, co)) \
            .fragments.set_index("fragment").loc["C9H11+"]
        assert (r["basis"] == "intact test") is kept, frag
    # the guard reads EVERY energy, not the last: the fragment peaks at NCE 10
    # (21 % of the TIC, 21x what the co-isolate held) and has decayed to 60
    # counts by NCE 20, below the bound -- still withdrawn, at NCE 10
    decay = steps + [(20, {main: 10000.0, "C9H11+": 60.0})]
    r = AN.analyze_run(_synthetic(round(_mz(main), 4), decay), _ions(main, co)) \
        .fragments.set_index("fragment").loc["C9H11+"]
    assert (r["call"], r["basis"]) == ("ambiguous", "ambiguous")
    assert "grew by 0.2118 at 179.1067@hcd10.00, more than the 0.009901" in r["test_note"]


def test_an_absent_named_ion_does_not_bypass_the_guard():
    """The guard case with a third named ion P (C12H19O+) that this sample
    lacks -- in no MS2 spectrum, no MS1 centroid inside the inventoried
    window: a measured share of 0 -- and that could make C9H11+ by
    composition (- C3H8O). P weighs 0, so it is set aside before the
    intact test: the test leaves the 1 % co-isolate alone and the guard
    withdraws that call, exactly as without P. Kept in the test, P (never
    there, so never excluded) made two candidates of one: the guard, which
    reads a single call, was skipped, and the co-isolate took 100 % of a
    fragment that grows to 21x all it held -- the main ion's F flat at 1,
    the co-isolate's E50 5.0 instead of 7.48 and its branching 21."""
    main, co, P = "C11H15O2+", "C11H16^NO+", "C12H19O+"
    steps = [(1, {main: 10000.0, co: 100.0}),
             (5, {main: 10000.0, co: 50.0, "C9H11+": 50.0}),
             (10, {main: 8000.0, "C9H11+": 2150.0})]
    t = _synthetic(round(_mz(main), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(main) + 0.0001, 10000 / 10100), (_mz(co), 100 / 10100)])
    two = AN.analyze_run(t, _ions(main, co))
    three = AN.analyze_run(t, _ions(main, co, P))
    assert three.windows[0].co_isolates == [co, P]
    a = two.fragments.set_index("fragment").loc["C9H11+"]
    b = three.fragments.set_index("fragment").loc["C9H11+"]
    assert b["candidates"] == f"{main};{co};{P}"
    for col in ("call", "basis", "excluded", "weight_source"):
        assert a[col] == b[col], col
    assert (b["call"], b["basis"]) == ("ambiguous", "ambiguous")
    assert _weights(b["weights"]) == pytest.approx({main: 10000 / 10100, co: 100 / 10100, P: 0.0})
    assert a["test_note"].startswith(f"intact test withdrawn: it left {co}")
    assert b["test_note"] == (f"{P}: in no MS2 spectrum of the window nor in the MS1 window "
                              "(absent from this sample): weight 0, set aside before the intact "
                              "test; " + a["test_note"])
    same = ["ion", "nce", "f", "f_lo", "branching_sum", "branching_flag"]
    pd.testing.assert_frame_equal(two.breakdown[same], three.breakdown[
        three.breakdown["ion"] != P][same].reset_index(drop=True))
    cols = ["ion", "curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(two.e50[cols], three.e50[three.e50["ion"] != P][cols]
                                  .reset_index(drop=True))
    assert _e50_row(three, p, co)["e50_nce"] == pytest.approx(7.475, abs=1e-3)
    assert not three.breakdown.loc[three.breakdown["ion"] == co, "branching_flag"].any()
    # an absent ion is set aside only on a MEASURED 0: with no window table
    # P's share is unknown, and P stays a candidate (an equal split, said)
    t.window = S._frame({}, S.WINDOW_COLUMNS)
    r = AN.analyze_run(t, _ions(main, co, P)).fragments.set_index("fragment").loc["C9H11+"]
    assert r["weight_source"] == "equal" and "set aside" not in r["test_note"]
    # the helper: no survivor AND a measured MS1 share of 0
    assert AT.absent_ions(["a", "b", "c", "d"], {"a": 5.0, "b": 0.0},
                          {"a": 0.0, "b": 0.0, "c": NAN}) == ["b"]


def test_the_guard_counts_what_the_called_ion_made_in_source():
    """The bound of the guard is everything the called ion held at the
    lowest energy: its precursor AND the fragments only it can make. X, a
    fragile ammonium adduct, is mostly its in-source product C5H10^N+ (X's
    only: no other window ion holds ^N) at NCE 1; that product goes on to
    C6H9+, which the intact co-isolate Y could make by composition. The
    intact test leaves X; C6H9+ grows by 0.125 of the TIC at NCE 10, twice
    X's own precursor fraction (0.0625) but well within X + C5H10^N+
    (0.625). The call stands: Y, which never fragments, keeps F = 1 and no
    E50, and X's branching sums to exactly 1 (mass balance holds). A bound
    of the precursor alone withdrew the call, gave Y 86 % of C6H9+ and a
    false E50 of 23.8, and X branching sums of -2.4 to -7.1 nothing flagged."""
    G = "C5H10^N+"
    steps = [(1, {X: 100.0, G: 900.0, Y: 600.0}),
             (10, {X: 50.0, G: 750.0, "C6H9+": 200.0, Y: 600.0}),
             (20, {X: 20.0, G: 380.0, "C6H9+": 600.0, Y: 600.0}),
             (30, {X: 5.0, G: 95.0, "C6H9+": 900.0, Y: 600.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    assert (w.target, w.co_isolates, w.notes) == (X, [Y], [])
    fr = res.fragments.set_index("fragment")
    assert (fr.loc[G, "call"], fr.loc[G, "basis"], fr.loc[G, "in_source"]) == \
        (X, "composition", True)
    r = fr.loc["C6H9+"]
    assert (r["candidates"], r["call"], r["basis"], r["excluded"], r["test_note"]) == \
        (f"{X};{Y}", X, "intact test", Y, "")
    assert _weights(r["weights"]) == {X: 1.0, Y: 0.0}
    # the bound, and why the precursor alone was not one
    frac = pd.DataFrame({X: [100 / 1600], Y: [600 / 1600]})
    phi = pd.DataFrame({G: [900 / 1600], "C6H9+": [0.0]})
    cands = {G: [X], "C6H9+": [X, Y]}
    assert AT.held_at_lowest(X, frac, phi, cands) == pytest.approx(0.625)
    assert AT.held_at_lowest(Y, frac, phi, cands) == pytest.approx(0.375)    # shares no G
    assert 200 / 1600 > AT.BRANCHING_FLAG * 100 / 1600
    _, fy, gy = _curve(res, w.parent_mz, Y)
    assert np.allclose(fy, 1.0) and _e50_row(res, w.parent_mz, Y)["flag"] == "above_range"
    assert np.allclose(gy["s"], 1.0)
    _, _, gx = _curve(res, w.parent_mz, X)
    assert math.isnan(gx["branching_sum"].iloc[0])
    assert np.allclose(gx["branching_sum"].iloc[1:], 1.0)
    assert not res.breakdown["branching_flag"].any()
    # a named ion the sample lacks that could make C5H10^N+ too (C9H16^NO+:
    # no MS2 survivor, no MS1 centroid) does not take it out of what X held:
    # the bound and the call are the same as without it
    t = _synthetic(round(_mz(X), 4), steps)
    t.window = _window_rows(w.parent_mz, [(_mz(X) + 0.0001, 0.2), (_mz(Y) - 0.0001, 0.8)])
    Z = "C9H16^NO+"
    res = AN.analyze_run(t, _ions(X, Y, Z))
    assert res.windows[0].co_isolates == [Z, Y]
    fr = res.fragments.set_index("fragment")
    assert (fr.loc[G, "candidates"], fr.loc[G, "call"]) == (f"{X};{Z}", X)
    r = fr.loc["C6H9+"]
    assert (r["call"], r["basis"], r["excluded"]) == (X, "intact test", Y)
    assert "withdrawn" not in r["test_note"]


def test_the_guard_bound_is_what_the_ion_held_at_the_lowest_energy():
    """The bound is the called ion's share at the LOWEST energy, not its
    largest: the co-isolate holds 0.99 % of the TIC at NCE 1, and at NCE 10,
    where the main ion has fallen and the TIC with it, 5.4 %. The fragment
    grows by 3.2 % of the TIC -- more than the co-isolate held at the lowest
    energy, so the call is withdrawn. A bound at its largest share (5.4 %)
    let the call stand and gave C9H11+ wholly to the co-isolate."""
    main, co = "C11H15O2+", "C11H16^NO+"
    steps = [(1, {main: 10000.0, co: 100.0}),
             (5, {main: 10000.0, co: 50.0, "C9H11+": 50.0}),
             (10, {main: 1700.0, co: 100.0, "C9H11+": 60.0})]
    res = AN.analyze_run(_synthetic(round(_mz(main), 4), steps), _ions(main, co))
    r = res.fragments.set_index("fragment").loc["C9H11+"]
    assert (r["call"], r["basis"], r["excluded"]) == ("ambiguous", "ambiguous", "")
    assert (f"grew by {60 / 1860:.4g} at 179.1067@hcd10.00, more than the {100 / 10100:.4g} "
            f"{co} held at the lowest energy") in r["test_note"]
    assert _weights(r["weights"]) == pytest.approx({main: 10000 / 10100, co: 100 / 10100})
    # the helper itself: row 0, whatever the later rows hold
    frac = pd.DataFrame({co: [0.0099, 0.05], main: [0.99, 0.9]})
    phi = pd.DataFrame({"G+": [0.01, 0.2], "H+": [0.3, 0.4]})
    assert AT.held_at_lowest(co, frac, phi, {"G+": [co], "H+": [main, co]}) == \
        pytest.approx(0.0199)


def test_the_intact_test_compares_with_the_lowest_energy():
    """The intact test's reference is a candidate's fraction at the LOWEST
    energy, not at the step before the fragment's onset. Y falls by 45 %
    into its own C6H9O+ between NCE 1 and 10, then holds; the shared C6H9+
    appears at NCE 20, the third energy, as X falls. Against the lowest
    energy Y has lost 45 % and is not excluded -- an ion that fell before
    the onset may still be the one making the fragment -- so the call is
    'ambiguous', split by the lowest-energy survivors (1000 : 1000). Against
    the step before (NCE 10, where Y had already fallen) Y read intact, was
    excluded, and X's E50 moved from 26 to 22.5 and Y's from 15.45 to none.
    The cost of the conservative split is said, not hidden: Y's products
    then grow by more than it lost, and its branching is flagged."""
    steps = [(1, {X: 1000.0, Y: 1000.0}),
             (10, {X: 1000.0, Y: 550.0, "C6H9O+": 450.0}),
             (20, {X: 600.0, Y: 550.0, "C6H9O+": 450.0, "C6H9+": 400.0}),
             (30, {X: 200.0, Y: 550.0, "C6H9O+": 450.0, "C6H9+": 800.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    r = res.fragments.set_index("fragment").loc["C6H9+"]
    assert (r["candidates"], r["call"], r["basis"], r["excluded"], r["test_note"]) == \
        (f"{X};{Y}", "ambiguous", "ambiguous", "", "")
    assert (r["onset_nce"], r["in_source"], r["weight_source"]) == (20.0, False, "ms2 survivors")
    assert _weights(r["weights"]) == {X: 0.5, Y: 0.5}
    assert _e50_row(res, w.parent_mz, X)["e50_nce"] == pytest.approx(26.0)
    ey = _e50_row(res, w.parent_mz, Y)
    # Y's F: 0.55 at NCE 10, 550 / 1200 at 20 (its fragment + half of C6H9+)
    assert ey["flag"] == "interpolated"
    assert ey["e50_nce"] == pytest.approx(10 + 0.05 / (0.55 - 550 / 1200) * 10)
    _, _, gy = _curve(res, w.parent_mz, Y)
    assert list(gy["branching_flag"]) == [False, False, True, True]
    assert gy["branching_sum"].iloc[2] == pytest.approx((0.225 + 0.5 * 0.2) / 0.225)


def test_attribution_and_survival_do_not_depend_on_the_spectrum_scale():
    """One spectrum scaled as a whole -- a drift of the precursor supply
    between energies, space charge, an AGC misprediction (Orbitrap
    intensities are per unit injection time, so the injection time itself is
    not one) -- must leave every call, weight, F, F_lo, their E50s and the
    branching as they were: the intact test, the guard and the branching
    read TIC fractions, F and F_lo are ratios within one spectrum. Only S (I / I at the lowest
    energy) and the raw intensities move. The intact test on raw counts
    called C10H17+ and C6H9+ 'ambiguous' once NCE 10 was scaled, and moved
    the E50s from 16.25 / 27.19 to 19.66 / 24.23."""
    p = round(_mz(X), 4)
    base = AN.analyze_run(_synthetic(p, _mixture()), _ions(X, Y))
    same = ["f", "f_lo", "frac", "branching_sum", "branching_flag", "budget_explained",
            "budget_unexplained", "n_peaks", "n_unexplained"]
    for at, k in ((1, 0.5), (10, 0.5), (10, 2.0), (20, 0.5)):
        steps = [(e, {q: v * (k if e == at else 1.0) for q, v in pk.items()})
                 for e, pk in _mixture()]
        res = AN.analyze_run(_synthetic(p, steps), _ions(X, Y))
        pd.testing.assert_frame_equal(res.fragments, base.fragments, obj=f"fragments {at} x{k}")
        pd.testing.assert_frame_equal(res.breakdown[same], base.breakdown[same])
        pd.testing.assert_frame_equal(res.branching, base.branching)
        fe = res.e50[res.e50["curve"] != "S"].reset_index(drop=True)
        pd.testing.assert_frame_equal(fe, base.e50[base.e50["curve"] != "S"].reset_index(
            drop=True))
        # the scale is in the data: S and the intensities moved
        assert not np.allclose(res.breakdown["s"], base.breakdown["s"])
        assert not np.allclose(res.breakdown["intensity"], base.breakdown["intensity"])


def test_peak_roles():
    """One row per (spectrum, peak): a precursor reading wins; a fragment's
    key is its formula, or every window ion's formula joined when they differ."""
    df = _table([("s1", 0, "A", "F1+", "H2O", "H2O", 1, False, 1),
                 ("s1", 0, "B", "F2+", "CO", "CO", 1, False, 1),
                 ("s1", 0, "B", "F3+", "C2H4", "C2H4", 1, False, 2),
                 ("s1", 1, "A", "F1+", "H2O", "H2O", 1, False, 1),
                 ("s1", 1, "B", "F1+", "CH4", "CH4", 1, False, 1),
                 ("s1", 2, "", "", "", "", 0, False, 0)])
    pre = _table([("s1", 3, "B", "B+", "", "", 0, False, 1)])
    pre["is_precursor"] = True
    roles = AT.peak_roles(pd.concat([df, pre], ignore_index=True))
    assert list(roles["role"]) == ["fragment", "fragment", "unexplained", "precursor"]
    assert list(roles["key"]) == ["F1+ | F2+", "F1+", "", ""]
    assert list(roles["candidates"]) == ["A;B", "A;B", "", "B"]
    assert roles["losses"].iloc[0] == "A: H2O; B: CO"
    assert roles["ion"].iloc[3] == "B"
    assert AT.fragment_key(["C6H9+", "C6H9+"]) == "C6H9+"
    assert AT.fragment_key(["b+", "a+", "b+"]) == "a+ | b+"
    # window order, not first appearance: B reads the first peak alone, A is
    # the target -- the candidates and their losses follow A, B
    late = _table([("s1", 0, "B", "F5+", "CO2", "CO2", 1, False, 1),
                   ("s1", 1, "A", "F6+", "H2O", "H2O", 1, False, 1),
                   ("s1", 1, "B", "F6+", "C2H4", "C2H4", 1, False, 1)])
    assert AT.peak_roles(late)["candidates"].iloc[1] == "B;A"            # first appearance
    r = AT.peak_roles(late, ["A", "B"]).iloc[1]
    assert (r["candidates"], r["losses"]) == ("A;B", "A: H2O; B: C2H4")
    # a peak two window ions both read as their precursor: marked, the first keeps it
    both = _table([("s1", 0, "B", "X+", "", "", 0, False, 1),
                   ("s1", 0, "A", "Y+", "", "", 0, False, 1)])
    both["is_precursor"] = True
    r = AT.peak_roles(both, ["A", "B"]).iloc[0]
    assert (r["role"], r["ion"], r["candidates"]) == ("precursor", "A", "A;B")


def test_fragment_losses_cover_every_candidate():
    """A fragment's losses are each candidate's, in the candidates' order,
    from the lowest energy at which it reads the fragment -- not the first
    energy's row alone (B reads it at s1, A only from s2). A reading of a
    formula the fragment cannot be where it sits (not one of its
    compositions) gives way to a later one that is: B's s1 reading as
    C5H5O+ is passed over for its s2 reading as C6H9+."""
    roles = pd.DataFrame({"spectrum": ["s1", "s2"], "role": ["fragment"] * 2,
                          "key": ["C6H9+"] * 2, "candidates": ["B", "A;B"],
                          "formulas": ["B: C6H9+", "A: C6H9+; B: C6H9+"],
                          "losses": ["B: CO", "A: H2O; B: C2H4"]})
    att = pd.DataFrame([{"key": "C6H9+", "candidates": "A;B", "n_candidates": 2,
                         "call": "ambiguous", "basis": "ambiguous", "weights": {"A": 0.5, "B": 0.5},
                         "weight_source": "equal", "excluded": "", "test_note": ""}])
    frag = pd.DataFrame({"C6H9+": [10.0, 20.0]}, index=["s1", "s2"])
    meta = pd.DataFrame({"nce": [1.0, 10.0], "hcd_v": [4.0, 6.0]}, index=["s1", "s2"])
    tic = pd.Series([100.0, 100.0], index=frag.index)
    out = AN._fragment_rows(150.0, att, roles, frag, tic, meta)
    assert (out["candidates"].iloc[0], out["losses"].iloc[0]) == ("A;B", "A: H2O; B: CO")
    out = AN._fragment_rows(150.0, att, roles, frag, tic, meta, compositions={"C6H9+": ["C6H9+"]})
    assert out["losses"].iloc[0] == "A: H2O; B: CO"
    other = roles.assign(formulas=["B: C5H5O+", "A: C6H9+; B: C6H9+"])
    out = AN._fragment_rows(150.0, att, other, frag, tic, meta, compositions={"C6H9+": ["C6H9+"]})
    assert out["losses"].iloc[0] == "A: H2O; B: C2H4"
    # no reading of a composition at all: the lowest-energy one, as before
    out = AN._fragment_rows(150.0, att, other, frag, tic, meta, compositions={"C6H9+": []})
    assert out["losses"].iloc[0] == "A: H2O; B: CO"


#: an organonitrate under the 15N reagent (14N + ^N) and a CHO co-isolate
#: 15 mDa above it. C4H8N^N+ (the target's only) and C5H9O+ (either's) lie
#: 0.443 mDa apart, 5.2 ppm at m/z 85: at the exact C4H8N^N+ m/z only it is
#: read (5 ppm); 0.3 ppm lower, C5H9O+ comes inside the tolerance and ranks
#: first. In 15N data that is a sub-ppm jitter between energies.
ON, ON_CO = "C10H21N^NO5+", "C15H22O3+"


def _jitter_steps():
    m = _mz("C4H8N^N+")
    return [(1, {ON: 1000.0, ON_CO: 1000.0, m: 20.0}),
            (10, {ON: 700.0, ON_CO: 800.0, m * (1 - 0.3e-6): 300.0, 120.5: 200.0}),
            (20, {ON: 400.0, ON_CO: 600.0, m: 600.0, 120.5: 400.0}),
            (30, {ON: 100.0, ON_CO: 400.0, m: 900.0, 120.5: 600.0})]


def test_a_fragment_whose_rank_1_formula_changes_with_the_energy_is_one_fragment():
    """One physical fragment, one row: the peak reads C4H8N^N+ at NCE 1, 20 and
    30 and C5H9O+ at NCE 10. Keyed per spectrum it was two rows -- C4H8N^N+
    (3 spectra, NCE 10 missing) and C5H9O+ (onset 10, not in-source), the
    latter split 0.5 / 0.5 with the co-isolate, which put the target's F at
    NCE 10 at 0.82 instead of 0.70 and its E50 at 17.6 instead of 16.7, and
    gave the intact co-isolate an F of 0.84. Merged, the fragment is the
    target's by composition: its signal-weighted m/z sits 0.05 ppm below
    C4H8N^N+ and 5.16 ppm above C5H9O+, beyond the 5 ppm tolerance, so the
    co-isolate -- which reads it only as C5H9O+, in the one spectrum where the
    peak sat 0.3 ppm low -- is not a candidate. That position is weighted by
    the share of its spectrum's TIC each peak holds, not by raw counts: the
    NCE 10 spectrum scaled as a whole by 20 (a supply drift) changes
    nothing, where raw counts would have pulled the position 0.24 ppm low
    and let the co-isolate in."""
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), _jitter_steps()), _ions(ON, ON_CO))
    (w,) = res.windows
    assert (w.target, w.co_isolates, w.status) == (ON, [ON_CO], "analysed")
    # the precondition: the target's rank-1 reading of the peak changes at NCE 10
    rd = res.readings
    top = rd[(rd["parent"] == ON) & (rd["rank"] == 1) & ~rd["is_precursor"]]
    assert dict(zip(top["nce"], top["formula"])) == {1.0: "C4H8N^N+", 10.0: "C5H9O+",
                                                     20.0: "C4H8N^N+", 30.0: "C4H8N^N+"}
    assert set(rd.loc[(rd["parent"] == ON_CO) & (rd["rank"] > 0) & ~rd["is_precursor"],
                      "nce"]) == {10.0}
    (r,) = res.fragments.itertuples()
    assert (r.fragment, r.candidates, r.call, r.basis) == \
        ("C4H8N^N+ | C5H9O+", ON, ON, "composition")
    assert (r.onset_nce, r.in_source, r.n_spectra) == (1.0, True, 4)
    assert (r.max_frac, r.max_frac_nce) == (pytest.approx(900 / 2000), 30.0)
    assert math.isnan(r.mz_theo) and _weights(r.weights) == {ON: 1.0}
    assert r.losses.startswith(f"{ON}: ") and ON_CO not in r.losses
    assert r.test_note == (f"not a candidate: {ON_CO} reads it only as C5H9O+ (+5.16 ppm) -- "
                           "beyond the reading tolerance of its signal-weighted m/z 85.06523")
    _, f_on, g_on = _curve(res, w.parent_mz, ON)
    assert np.allclose(f_on, [1000 / 1020, 0.7, 0.4, 0.1])
    assert np.allclose(g_on["f_lo"], f_on)
    assert _e50_row(res, w.parent_mz, ON)["e50_nce"] == pytest.approx(10 + 0.2 / 0.3 * 10)
    _, f_co, _ = _curve(res, w.parent_mz, ON_CO)
    assert np.allclose(f_co, 1.0)                  # its loss went to the unnamed m/z 120.5
    assert not res.breakdown["branching_flag"].any()
    # the position is a TIC-fraction mean: a spectrum's scale moves nothing
    m = AT.fragment_mz(w.roles)["C4H8N^N+ | C5H9O+"]
    assert (m / _mz("C4H8N^N+") - 1) * 1e6 == pytest.approx(-0.3 * 0.15 / (20 / 2020 + 0.9),
                                                             abs=1e-3)
    scaled = [(e, {k: v * (20.0 if e == 10 else 1.0) for k, v in pk.items()})
              for e, pk in _jitter_steps()]
    again = AN.analyze_run(_synthetic(round(_mz(ON), 4), scaled), _ions(ON, ON_CO))
    pd.testing.assert_frame_equal(again.fragments, res.fragments)
    pd.testing.assert_frame_equal(again.e50[again.e50["curve"] != "S"].reset_index(drop=True),
                                  res.e50[res.e50["curve"] != "S"].reset_index(drop=True))


def _mirror_steps(stray=5.0, co=(1000.0, 700.0, 400.0, 100.0), other=None):
    """The organonitrate case mirrored: the CHO co-isolate falls into C5H9O+,
    centroided 1.2 ppm high (4.0 ppm below C4H8N^N+: the target reads it as
    either, the co-isolate as C5H9O+), while the target never fragments. At
    NCE 1 a stray centroid of `stray` counts sits at the exact C4H8N^N+ m/z,
    5.2 ppm above C5H9O+: only the target can read it, and it is within 5 ppm
    of the others, so it merges with them. `other`: an unnamed m/z 120.5
    per step (signal no window ion claims)."""
    n, hi = _mz("C4H8N^N+"), _mz("C5H9O+") * (1 + 1.2e-6)
    steps = []
    for i, (e, frag) in enumerate(zip((1, 10, 20, 30), (stray, 300.0, 600.0, 900.0))):
        pk = {ON: 1000.0, ON_CO: co[i]}
        if frag > 0:
            pk[n if e == 1 else hi] = frag
        if other is not None:
            pk[120.5] = other[i]
        steps.append((e, pk))
    return steps


def test_a_stray_centroid_does_not_take_a_fragment_from_the_ion_that_made_it():
    """The mirror of the organonitrate case. The co-isolate makes the
    fragment and the target never loses signal, so the fragment is the
    co-isolate's: its F crosses 0.5 at NCE 16.67 and the target's F stays 1.
    The fragment sits 1.2 ppm above C5H9O+ (signal-weighted; the stray
    centroid holds 0.25 % of the NCE 1 TIC and moves it by 0.01 ppm), where
    both ions can make it: both are candidates. Every candidate reads it
    from NCE 10, the intact test reads its onset there -- the target is
    intact (100 % of its NCE 1 fraction), excluded -- and calls the
    co-isolate; the guard lets it stand (the fragment grows by 0.4475 of the
    TIC, the co-isolate held 0.4988). Asked of every spectrum, the stray
    centroid vetoed the co-isolate -- which reads the fragment in the three
    spectra that hold 99.7 % of it -- and gave the fragment to the target by
    composition: the target's F fell to 0.53 with its branching flagged at
    every energy above NCE 1, and the co-isolate read no E50."""
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), _mirror_steps()), _ions(ON, ON_CO))
    (w,) = res.windows
    assert (w.target, w.co_isolates, w.status) == (ON, [ON_CO], "analysed")
    rd = res.readings
    co_reads = rd[(rd["parent"] == ON_CO) & (rd["rank"] == 1) & ~rd["is_precursor"]]
    assert set(co_reads["nce"]) == {10.0, 20.0, 30.0} and set(co_reads["formula"]) == {"C5H9O+"}
    (r,) = res.fragments.itertuples()
    assert (r.fragment, r.candidates, r.call, r.basis, r.excluded) == \
        ("C4H8N^N+ | C5H9O+", f"{ON};{ON_CO}", ON_CO, "intact test", ON)
    assert _weights(r.weights) == {ON: 0.0, ON_CO: 1.0}
    assert (r.onset_nce, r.in_source, r.n_spectra) == (1.0, True, 4)
    assert r.test_note == ("the intact test reads its onset at 250.1415@hcd10.00, the first "
                           "spectrum in which every candidate reads it (present from "
                           "250.1415@hcd1.00, read there by some of them only, holding at most "
                           "0.0166x its TIC fraction at 250.1415@hcd10.00: a stray)")
    m = AT.fragment_mz(w.roles)[r.fragment]
    assert (m / _mz("C5H9O+") - 1) * 1e6 == pytest.approx(1.2 + 5 / 2005 / (5 / 2005 + 0.9)
                                                           * (5.2105 - 1.2), abs=2e-3)
    _, f_on, g_on = _curve(res, w.parent_mz, ON)
    assert np.allclose(f_on, 1.0) and _e50_row(res, w.parent_mz, ON)["flag"] == "above_range"
    _, f_co, g_co = _curve(res, w.parent_mz, ON_CO)
    assert np.allclose(f_co, [1000 / 1005, 0.7, 0.4, 0.1])
    assert _e50_row(res, w.parent_mz, ON_CO)["e50_nce"] == pytest.approx(10 + 0.2 / 0.3 * 10)
    assert np.allclose(g_co["branching_sum"].iloc[1:],
                       [(0.15 - 5 / 2005) / (1000 / 2005 - 0.35),
                        (0.30 - 5 / 2005) / (1000 / 2005 - 0.20),
                        (0.45 - 5 / 2005) / (1000 / 2005 - 0.05)])
    assert not res.breakdown["branching_flag"].any()
    assert r.losses == f"{ON}: C6H13O5; {ON_CO}: C10H13O2"
    # without the stray centroid: the same call, the same E50s (only the
    # co-isolate's F at NCE 1 moves, 1000 / 1005 -> 1)
    clean = AN.analyze_run(_synthetic(round(_mz(ON), 4), _mirror_steps(stray=0.0)),
                           _ions(ON, ON_CO))
    c = clean.fragments.iloc[0]
    assert (c["fragment"], c["call"], c["basis"], c["test_note"]) == \
        ("C5H9O+", ON_CO, "intact test", "")
    cols = ["ion", "curve", "e50_nce", "e50_v", "flag"]
    pd.testing.assert_frame_equal(clean.e50[cols], res.e50[cols])


def test_a_fragment_where_both_compositions_read_it_has_both_candidates():
    """The organonitrate case with the 0.3 ppm jitter at NCE 10, 20 and 30 --
    where the signal is -- instead of at one energy: the fragment sits 0.30
    ppm below C4H8N^N+ and 4.92 ppm above C5H9O+, inside the tolerance of
    both, so the co-isolate is a candidate (a vote of the spectra, three of
    four, said otherwise). Every candidate reads it from NCE 10, but the
    NCE 1 peak only the target reads holds 0.066x of its TIC fraction there
    -- more than a stray: the fragment is in-source, no test runs, and it is
    split by the NCE 1 survivors (both ions fall from NCE 10: the moved
    onset gave the same split)."""
    m = _mz("C4H8N^N+")
    lo = m * (1 - 0.3e-6)
    steps = [(1, {ON: 1000.0, ON_CO: 1000.0, m: 20.0}),
             (10, {ON: 700.0, ON_CO: 800.0, lo: 300.0, 120.5: 200.0}),
             (20, {ON: 400.0, ON_CO: 600.0, lo: 600.0, 120.5: 400.0}),
             (30, {ON: 100.0, ON_CO: 400.0, lo: 900.0, 120.5: 600.0})]
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps), _ions(ON, ON_CO))
    (r,) = res.fragments.itertuples()
    assert (r.candidates, r.call, r.basis, r.excluded, r.weight_source) == \
        (f"{ON};{ON_CO}", "ambiguous", "ambiguous", "", "ms2 survivors")
    assert _weights(r.weights) == {ON: 0.5, ON_CO: 0.5}
    assert r.in_source and r.test_note.startswith(
        "present from 250.1415@hcd1.00, read there by some of its candidates only, but holding "
        "0.066x its TIC fraction at 250.1415@hcd10.00")
    w = res.windows[0]
    pos = (AT.fragment_mz(w.roles)[r.fragment] / m - 1) * 1e6
    assert pos == pytest.approx(-0.3 * 0.9 / (20 / 2020 + 0.9), abs=1e-3)


def test_an_in_source_fragment_one_candidate_reads_first_stays_in_source():
    """The reverse of the stray centroid. The target holds an in-source
    C4H8N^N+ of 300 counts at every energy (it never fragments further);
    the co-isolate falls into its own C11H13O+. At NCE 1 the peak sits at
    the exact C4H8N^N+ m/z, read by the target only; from NCE 10 on 0.5 ppm
    low, where the co-isolate reads it as C5H9O+ too. Its signal-weighted
    m/z is within the tolerance of both: two candidates. Every candidate
    reads it from NCE 10 -- but the NCE 1 peak holds as much of the TIC as
    the fragment ever does (1x): not a stray, so the fragment is in-source,
    no test runs and it is split by the NCE 1 survivors. Read at NCE 10 the
    target was intact and excluded, the co-isolate was called by 'intact
    test' -- with the 300 counts only the target can have made at NCE 1 --
    its E50 11.67 and the target's F flat at 1, nothing flagged."""
    n = _mz("C4H8N^N+")
    def steps(jit):
        out = []
        for e, b in zip(NCE, (1000.0, 700.0, 400.0, 100.0)):
            pk = {ON: 1000.0, ON_CO: b, (n if e == 1 else n * (1 - jit)): 300.0}
            if b < 1000:
                pk["C11H13O+"] = 1000.0 - b
            out.append((e, pk))
        return out
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps(0.5e-6)), _ions(ON, ON_CO))
    (w,) = res.windows
    r = res.fragments.set_index("fragment").loc["C4H8N^N+ | C5H9O+"]
    assert r["candidates"] == f"{ON};{ON_CO}"
    assert (r["call"], r["basis"], r["excluded"], r["weight_source"]) == \
        ("ambiguous", "ambiguous", "", "ms2 survivors")
    assert r["in_source"] and _weights(r["weights"]) == {ON: 0.5, ON_CO: 0.5}
    assert r["test_note"].startswith(
        "present from 250.1415@hcd1.00, read there by some of its candidates only, but holding "
        "1x its TIC fraction at 250.1415@hcd10.00")
    # the co-isolate: F = I / (I + its C11H13O+ + half of the in-source 300)
    _, f_co, _ = _curve(res, w.parent_mz, ON_CO)
    assert np.allclose(f_co, np.array([1000.0, 700.0, 400.0, 100.0]) / 1150.0)
    assert _e50_row(res, w.parent_mz, ON_CO)["e50_nce"] == pytest.approx(10 + 125 / 300 * 10)
    _, f_on, _ = _curve(res, w.parent_mz, ON)
    assert np.allclose(f_on, 1000.0 / 1150.0)
    assert not res.breakdown["branching_flag"].any()
    # no jitter: only the target reads it -- its own, by composition, and the
    # co-isolate's E50 is its own C11H13O+'s (16.67)
    clean = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps(0.0)), _ions(ON, ON_CO))
    c = clean.fragments.set_index("fragment").loc["C4H8N^N+"]
    assert (c["candidates"], c["call"], c["basis"]) == (ON, ON, "composition")
    assert _e50_row(clean, w.parent_mz, ON_CO)["e50_nce"] == pytest.approx(10 + 2 / 3 * 10)


def test_the_guard_still_reads_a_call_made_at_the_shared_onset():
    """The mirror case with a co-isolate of 1/40 the size: the target is
    intact at NCE 10 (an unnamed ion's loss keeps the TIC at 2000) and the
    test leaves the co-isolate -- but the fragment grows by 0.1475 of the
    TIC there, past the 0.025 that co-isolate held: withdrawn, ambiguous,
    and both notes kept. The cost is said: the target, which lost nothing,
    gets most of the fragment by survivor share and its branching is
    flagged."""
    steps = _mirror_steps(co=(50.0, 35.0, 20.0, 5.0), other=(945.0, 665.0, 380.0, 95.0))
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps), _ions(ON, ON_CO))
    (r,) = res.fragments.itertuples()
    assert (r.call, r.basis, r.excluded) == ("ambiguous", "ambiguous", "")
    assert r.test_note.startswith("the intact test reads its onset at 250.1415@hcd10.00")
    assert "; intact test withdrawn: it left C15H22O3+, but the fragment's TIC fraction grew " \
        "by 0.1475 at 250.1415@hcd10.00, more than the 0.025 C15H22O3+ held" in r.test_note
    assert _weights(r.weights) == pytest.approx({ON: 1000 / 1050, ON_CO: 50 / 1050})
    _, _, g_on = _curve(res, res.windows[0].parent_mz, ON)
    assert list(g_on["branching_flag"]) == [False, True, True, True]


def test_merge_fragment_peaks_links_by_mz_and_by_formula():
    """Fragment peaks are linked across the window's spectra when their
    calibrated m/z lie within the reading tolerance of each other, or when
    they share a rank-1 formula (two peaks of one formula may sit up to twice
    the tolerance apart); the name is every formula read in the group. Other
    roles are never touched, and an unexplained peak never bridges two.
    At m/z 85 the tolerance is 5 ppm = 0.425 mDa: s1 -> s2 is 0.21 mDa (an
    m/z link, two formulas), s2 -> s3 0.45 mDa (a formula link only)."""
    rows = [("s1", "fragment", "C5H9O+", 85.06479, "A;B"),
            ("s2", "fragment", "C4H8N^N+", 85.06500, "A"),
            ("s3", "fragment", "C4H8N^N+", 85.06545, "A"),
            ("s1", "fragment", "C6H9+", 81.06988, "A"),
            ("s2", "fragment", "C6H9+", 81.07050, "A"),         # 0.62 mDa: by formula
            ("s2", "fragment", "C5H7O+", 83.04914, "A"),
            ("s3", "fragment", "C6H10+", 82.07770, "A"),
            ("s3", "fragment", "C6H10+ | C5H6O+", 82.07800, "A;B"),
            ("s3", "precursor", "", 155.1430, "B")]
    roles = pd.DataFrame(rows, columns=["spectrum", "role", "key", "mz_cal", "candidates"])
    out = AT.merge_fragment_peaks(roles, tol_ppm=5.0, tol_mda=0.3)
    both = "C4H8N^N+ | C5H9O+"
    assert list(out["key"]) == [both, both, both, "C6H9+", "C6H9+", "C5H7O+",
                                "C5H6O+ | C6H10+", "C5H6O+ | C6H10+", ""]
    assert out.drop(columns="key").equals(roles.drop(columns="key"))
    # 0.43 mDa apart at m/z 85, just beyond 5 ppm: two fragments, and an
    # unexplained peak 0.22 mDa from each does not join them
    apart = pd.DataFrame([("s1", "fragment", "C4H8N^N+", 85.06523, "A"),
                          ("s2", "fragment", "C5H9O+", 85.06480, "A;B"),
                          ("s3", "unexplained", "", 85.06501, "")], columns=roles.columns)
    assert list(AT.merge_fragment_peaks(apart, tol_ppm=5.0, tol_mda=0.3)["key"]) == \
        ["C4H8N^N+", "C5H9O+", ""]
    # the order of the rows does not change the names
    back = AT.merge_fragment_peaks(roles.iloc[::-1], tol_ppm=5.0, tol_mda=0.3)
    assert back.sort_index().equals(out)


def _alias_roles(peaks):
    """Hand-built roles of one fragment group: (spectrum, m/z, intensity,
    {window ion: formula}) per peak, each spectrum's TIC 2000 (a precursor
    peak holds the rest)."""
    rows = []
    for spec, mz, inten, reads in peaks:
        rows.append((spec, "fragment", "G", ";".join(reads),
                     "; ".join(f"{i}: {f}" for i, f in reads.items()), float(mz), float(inten)))
    for spec in dict.fromkeys(p[0] for p in peaks):
        used = sum(p[2] for p in peaks if p[0] == spec)
        rows.append((spec, "precursor", "", "A", "", 250.0, 2000.0 - used))
    return pd.DataFrame(rows, columns=["spectrum", "role", "key", "candidates", "formulas",
                                       "mz_cal", "intensity"])


def test_candidates_across_energies():
    """A fragment's candidates are the window ions that read it as a formula
    within the reading tolerance of its signal-weighted m/z -- not a vote of
    the spectra. N = C4H8N^N+, O = C5H9O+ (0.443 mDa below; 5 ppm at m/z 85
    is 0.425 mDa). A can make both, B only O."""
    n, o = _mz("C4H8N^N+"), _mz("C5H9O+")
    lo = n * (1 - 0.3e-6)                          # 0.3 ppm low: both read it there
    both, a_n = {"A": "C5H9O+", "B": "C5H9O+"}, {"A": "C4H8N^N+"}
    # the jitter in a weak spectrum: the fragment sits at N, B is out, said
    weak = _alias_roles([("s1", n, 900, a_n), ("s2", lo, 100, both)])
    cand, notes = AT.candidates_across(weak, ["A", "B"])
    assert cand == {"G": ["A"]}
    assert notes["G"].startswith("not a candidate: B reads it only as C5H9O+ (+5.18 ppm) -- "
                                 "beyond the reading tolerance of its signal-weighted m/z")
    assert AT.fragment_compositions(weak)["G"][1] == ["C4H8N^N+"]
    # the same jitter where the signal is: the fragment sits where both
    # formulas read it, and B is a candidate, unsaid
    strong = _alias_roles([("s1", n, 100, a_n), ("s2", lo, 900, both)])
    assert AT.candidates_across(strong, ["A", "B"]) == ({"G": ["A", "B"]}, {})
    assert AT.fragment_compositions(strong)["G"][1] == ["C4H8N^N+", "C5H9O+"]
    # a stray centroid of 5 counts at N, which only A reads, vetoes nobody
    stray = _alias_roles([("s1", n, 5, a_n), ("s2", o * (1 + 1.2e-6), 900, both),
                          ("s3", o * (1 + 1.2e-6), 600, both)])
    assert AT.candidates_across(stray, ["A", "B"]) == ({"G": ["A", "B"]}, {})
    # the weights are TIC fractions: a spectrum's scale moves nothing
    tic_s2 = weak.assign(intensity=np.where(weak["spectrum"] == "s2", 50.0, 1.0)
                         * weak["intensity"])
    assert AT.fragment_mz(tic_s2) == pytest.approx(AT.fragment_mz(weak))
    assert AT.candidates_across(tic_s2, ["A", "B"])[0] == {"G": ["A"]}
    # the tolerance is the caller's; window order, not first appearance
    assert AT.candidates_across(weak, ["B", "A"], tol_ppm=5.3)[0] == {"G": ["B", "A"]}
    # no formula within the tolerance of the position (1 ppm, 0.1 mDa: the
    # peaks sit 0.08 mDa outside N and O, on opposite sides): every reader, said
    apart = _alias_roles([("s1", n + 0.8e-4, 500, a_n),
                          ("s2", o - 0.8e-4, 500, {"B": "C5H9O+"})])
    cand, notes = AT.candidates_across(apart, ["A", "B"], tol_ppm=1.0, tol_mda=0.1)
    assert cand == {"G": ["A", "B"]}
    assert notes["G"].startswith("no formula read in it lies within the reading tolerance of "
                                 "its signal-weighted m/z")


def test_shared_presence():
    """Where every candidate of a fragment reads it: two peaks of one spectrum
    pool their readers; a spectrum without the fragment is False; a fragment
    with one candidate is shared wherever it is read."""
    roles = pd.DataFrame(
        [("s1", "fragment", "K", "A"), ("s2", "fragment", "K", "A;B"),
         ("s2", "fragment", "L", "A"), ("s2", "fragment", "L", "B"),
         ("s3", "fragment", "M", "B"), ("s1", "precursor", "", "A")],
        columns=["spectrum", "role", "key", "candidates"])
    sh = AT.shared_presence(roles, {"K": ["A", "B"], "L": ["A", "B"], "M": ["B"]},
                            ["s1", "s2", "s3"])
    assert list(sh.columns) == ["K", "L", "M"] and list(sh.index) == ["s1", "s2", "s3"]
    assert sh.to_dict("list") == {"K": [False, True, False], "L": [False, True, False],
                                  "M": [False, False, True]}
    # an ion the sample lacks is not waited for: K is shared where A, the one
    # candidate left, reads it; M, whose every candidate is absent, keeps them
    cand = {"K": ["A", "B"], "L": ["A", "B"], "M": ["B"]}
    assert AT.live_candidates(cand, ["B"]) == {"K": ["A"], "L": ["A"], "M": ["B"]}
    sh = AT.shared_presence(roles, cand, ["s1", "s2", "s3"], absent=["B"])
    assert sh.to_dict("list") == {"K": [True, True, False], "L": [False, True, False],
                                  "M": [False, False, True]}


def test_the_intact_test_reads_the_onset_every_candidate_shares():
    """`attribute_fragments` with `shared`: the fragment is present from e1,
    where only some candidates read it -- a stray of 5 counts, 0.0166x of its
    TIC fraction at e2 -- and every candidate reads it from e2. A is intact
    at e2, B has fallen: the test runs at e2 and calls B, said. Without
    `shared` the fragment is in-source and the call ambiguous; with no
    spectrum every candidate reads, it has no onset: ambiguous, said. A
    presence before the shared onset larger than a stray is the fragment's
    own onset."""
    idx = ["e1", "e2", "e3"]
    ion = pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 700.0, 400.0]}, index=idx)
    frag = pd.DataFrame({"F": [5.0, 300.0, 600.0]}, index=idx)
    tic = ion.sum(axis=1) + frag["F"]
    shared = pd.DataFrame({"F": [False, True, True]}, index=idx)
    r = AT.attribute_fragments(ion, frag, tic, {"F": ["A", "B"]}, shared=shared).iloc[0]
    assert (r["call"], r["basis"], r["excluded"], r["weights"]) == \
        ("B", "intact test", "A", {"A": 0.0, "B": 1.0})
    assert r["test_note"] == ("the intact test reads its onset at e2, the first spectrum in "
                              "which every candidate reads it (present from e1, read there by "
                              "some of them only, holding at most 0.0166x its TIC fraction at "
                              "e2: a stray)")
    r = AT.attribute_fragments(ion, frag, tic, {"F": ["A", "B"]}).iloc[0]
    assert (r["call"], r["excluded"], r["weight_source"]) == ("ambiguous", "", "ms2 survivors")
    assert r["test_note"].startswith("present at the lowest energy")
    # shared from the lowest energy on: in-source as before, no onset note
    r = AT.attribute_fragments(ion, frag, tic, {"F": ["A", "B"]},
                               shared=shared.assign(F=True)).iloc[0]
    assert r["call"] == "ambiguous" and r["test_note"].startswith("present at the lowest energy")
    r = AT.attribute_fragments(ion, frag, tic, {"F": ["A", "B"]},
                               shared=shared.assign(F=False)).iloc[0]
    assert (r["call"], r["excluded"]) == ("ambiguous", "")
    assert r["test_note"].startswith("no spectrum in which every candidate reads it (present "
                                     "from e1")
    # one candidate: composition, whatever `shared` says
    r = AT.attribute_fragments(ion, frag, tic, {"F": ["B"]}, shared=shared.assign(F=False))
    assert (r.iloc[0]["call"], r.iloc[0]["basis"]) == ("B", "composition")
    # a moved onset at which every candidate is intact (a constant TIC: the
    # fragment grows from signal no window ion claims): both notes, the
    # moved onset's and why no test could decide
    flat = pd.Series(3000.0, index=idx)
    still = ion.assign(B=[1000.0, 1000.0, 700.0])
    r = AT.attribute_fragments(still, frag, flat, {"F": ["A", "B"]}, shared=shared).iloc[0]
    assert (r["call"], r["excluded"]) == ("ambiguous", "")
    assert r["test_note"] == ("the intact test reads its onset at e2, the first spectrum in "
                              "which every candidate reads it (present from e1, read there by "
                              "some of them only, holding at most 0.0167x its TIC fraction at "
                              "e2: a stray); every candidate intact at the fragment's onset")
    # absent at e1, shared at e2 only: the onset the test reads IS the
    # fragment's own -- no onset note (the later spectrum some candidates
    # cannot read changes nothing), and the call says nothing else
    late = pd.DataFrame({"F": [0.0, 300.0, 600.0]}, index=idx)
    r = AT.attribute_fragments(ion, late, flat, {"F": ["A", "B"]},
                               shared=pd.DataFrame({"F": [False, True, False]}, index=idx)
                               ).iloc[0]
    assert (r["call"], r["basis"], r["excluded"], r["test_note"]) == \
        ("B", "intact test", "A", "")
    # what some candidates read before the shared onset moves it only as a
    # stray (<= ONSET_STRAY_SHARE of the fragment's TIC fraction there): an
    # in-source fragment read at the lowest energy by one candidate only (a
    # sub-ppm jitter) is still in-source -- no test, an ambiguous split --
    # never called for the other ion because the one that read it is intact
    big = pd.DataFrame({"F": [300.0, 300.0, 600.0]}, index=idx)
    r = AT.attribute_fragments(ion, big, flat, {"F": ["A", "B"]}, shared=shared).iloc[0]
    assert (r["call"], r["excluded"], r["weight_source"]) == ("ambiguous", "", "ms2 survivors")
    assert r["test_note"] == ("present from e1, read there by some of its candidates only, but "
                              "holding 1x its TIC fraction at e2 (the first spectrum in which "
                              "every candidate reads it) -- more than a stray (<= 0.05x): the "
                              "intact test reads the fragment's own onset; present at the lowest "
                              "energy (in-source, or formed below the lowest step)")
    assert AT.ONSET_STRAY_SHARE == 0.05
    for e1, moved in ((14.0, True), (16.0, False)):          # 0.047x and 0.053x
        f = pd.DataFrame({"F": [e1, 300.0, 600.0]}, index=idx)
        r = AT.attribute_fragments(ion, f, flat, {"F": ["A", "B"]}, shared=shared).iloc[0]
        assert (r["basis"] == "intact test") is moved, e1
        assert ("a stray)" in r["test_note"]) is moved, e1
    # not at the lowest energy: the test runs at the fragment's own onset
    r = AT.attribute_fragments(ion, late, flat, {"F": ["A", "B"]},
                               shared=pd.DataFrame({"F": [False, False, True]}, index=idx)
                               ).iloc[0]
    assert (r["call"], r["basis"], r["excluded"]) == ("B", "intact test", "A")
    assert r["test_note"].startswith("present from e2, read there by some of its candidates "
                                     "only, but holding 0.5x its TIC fraction at e3")


def test_intact_test_and_share_weights():
    idx = ["e1", "e2", "e3"]
    frac = pd.DataFrame({"a": [0.5, 0.5, 0.2], "b": [0.3, 0.2, 0.1], "c": [0.0, 0.0, 0.0]},
                        index=idx)
    absent = pd.Series([False, True, True], index=idx)
    assert AT.intact_test(["a", "b"], frac, absent) == (["a"], "")
    # 98 %: 0.49 / 0.5 stays excluded, 0.489 does not
    f2 = frac.copy()
    f2.loc["e2", "a"] = 0.49
    assert AT.intact_test(["a", "b"], f2, absent)[0] == ["a"]
    f2.loc["e2", "a"] = 0.489
    assert AT.intact_test(["a", "b"], f2, absent)[0] == []
    # a candidate absent at the lowest energy cannot be tested: it stays
    assert AT.intact_test(["a", "c"], frac, absent) == (["a"], "")
    assert AT.intact_test(["b", "c"], frac, absent) == ([], "")
    # present at the lowest energy, never present, or every candidate intact
    assert AT.intact_test(["a", "b"], frac, pd.Series([True, True, True], index=idx))[1] \
        .startswith("present at the lowest energy")
    assert AT.intact_test(["a", "b"], frac, pd.Series([False] * 3, index=idx))[1] == \
        "never present"
    f3 = frac.assign(b=[0.3, 0.3, 0.3])
    assert AT.intact_test(["a", "b"], f3, absent) == (
        [], "every candidate intact at the fragment's onset")
    # weights, per candidate: its survivor; else its MS1 share on the
    # survivors' scale; MS1 shares alone when no candidate has a survivor;
    # else equal (and said)
    assert AT.share_weights(["a", "b"], {"a": 3.0, "b": 1.0}) == ({"a": 0.75, "b": 0.25},
                                                                  "ms2 survivors", "")
    w, src, note = AT.share_weights(["a", "b"], {"a": 3.0, "b": 0.0}, {"a": 0.2, "b": 0.6})
    assert (w, src) == (pytest.approx({"a": 0.25, "b": 0.75}), "ms2 survivors + ms1 window")
    assert note.startswith("no MS2 survivor at the lowest energy for b (share 0.6): weighed by "
                           "its MS1 window share on the survivors' scale (15 per unit share, "
                           "from a)")
    w, src, note = AT.share_weights(["a", "b"], {"a": 0.0, "b": 0.0}, {"a": 0.2, "b": 0.6})
    assert (w, src, note) == (pytest.approx({"a": 0.25, "b": 0.75}), "ms1 window", "")
    w, src, note = AT.share_weights(["a", "b"], {"a": 3.0}, {"a": 0.2})
    assert (w, src) == ({"a": 0.5, "b": 0.5}, "equal")
    assert note == "b: no MS2 survivor at the lowest energy and no MS1 share known: an equal split"
    # the equal split is over the candidates NOT absent from both: P (no
    # survivor, a measured MS1 share of 0) takes nothing beside the unknown Z
    w, src, note = AT.share_weights(["X", "Z", "P"], {"X": 1000.0}, {"X": 0.5, "Z": NAN, "P": 0.0})
    assert (w, src) == ({"X": 0.5, "Z": 0.5, "P": 0.0}, "equal")
    assert note.startswith("Z: no MS2 survivor") and "P" not in note
    # ... and on the other equal branch (no candidate with both to set the scale)
    w, src, note = AT.share_weights(["X", "Z", "P"], {"X": 1000.0}, {"X": NAN, "Z": 0.4, "P": 0.0})
    assert (w, src) == ({"X": 0.5, "Z": 0.5, "P": 0.0}, "equal")
    assert note.startswith("Z: no MS2 survivor at the lowest energy, and no candidate with a "
                           "survivor has an MS1 share")
    # every candidate absent: an equal split over them all, said
    w, src, note = AT.share_weights(["Z", "P"], {}, {"Z": 0.0, "P": 0.0})
    assert (w, src) == ({"Z": 0.5, "P": 0.5}, "equal") and note.startswith("every candidate")


def test_share_weights_come_from_the_lowest_energy():
    """Y is absent at the lowest energy and appears at NCE 10: its survivor
    is the LOWEST energy's (0), not its largest, so Y's share of the
    in-source fragment comes from the MS1 window, on X's survivor scale (X
    1000 at 0.25 of the window: Y's 0.75 is 3000 -- X 0.25, Y 0.75).
    Survivors taken at another energy would read 'ms2 survivors' 0.77 / 0.23
    instead."""
    steps = [(1, {X: 1000.0, "C7H11+": 50.0}), (10, {X: 600.0, Y: 300.0, "C7H11+": 450.0}),
             (20, {X: 300.0, Y: 300.0, "C7H11+": 750.0})]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(X) + 0.0001, 0.25), (_mz(Y) - 0.0001, 0.75)])
    r = AN.analyze_run(t, _ions(X, Y)).fragments.set_index("fragment").loc["C7H11+"]
    assert (r["call"], r["in_source"], r["weight_source"]) == \
        ("ambiguous", True, "ms2 survivors + ms1 window")
    assert _weights(r["weights"]) == pytest.approx({X: 0.25, Y: 0.75})
    assert f"no MS2 survivor at the lowest energy for {Y} (share 0.75)" in r["test_note"]


def test_a_missing_survivor_falls_back_to_the_ms1_window():
    """Y's precursor is absent from the lowest-energy MS2 spectrum: its share
    of the in-source fragment comes from the MS1 window, put on X's survivor
    scale (X 0.25, Y 0.75); X keeps its survivor."""
    steps = [(1, {X: 1000.0, "C7H11+": 50.0}), (10, {X: 600.0, "C7H11+": 450.0})]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = S._frame({"sample_id": ["S", "S"], "parent_mz": [p, p],
                         "mz": [_mz(Y) - 0.0001, _mz(X) + 0.0001], "offset_da": [0.0, 0.0],
                         "intensity": [3.0, 1.0], "share": [0.75, 0.25],
                         "resolution": [NAN, NAN], "signal_to_noise": [NAN, NAN],
                         "is_parent": [False, True]}, S.WINDOW_COLUMNS)
    res = AN.analyze_run(t, _ions(X, Y))
    r = res.fragments.set_index("fragment").loc["C7H11+"]
    assert r["weight_source"] == "ms2 survivors + ms1 window"
    assert _weights(r["weights"]) == pytest.approx({X: 0.25, Y: 0.75})
    # no window table at all: an equal split, said
    t.window = S._frame({}, S.WINDOW_COLUMNS)
    r = AN.analyze_run(t, _ions(X, Y)).fragments.set_index("fragment").loc["C7H11+"]
    assert r["weight_source"] == "equal"


def _window_rows(p, rows):
    """An `ms2_window.csv` for parent `p` from (m/z, share) rows."""
    n = len(rows)
    return S._frame({"sample_id": ["S"] * n, "parent_mz": [p] * n,
                     "mz": [m for m, _ in rows], "offset_da": [m - p for m, _ in rows],
                     "intensity": [10.0 * s for _, s in rows], "share": [s for _, s in rows],
                     "resolution": [NAN] * n, "signal_to_noise": [NAN] * n,
                     "is_parent": [i == 0 for i in range(n)]}, S.WINDOW_COLUMNS)


def test_a_named_co_isolate_absent_from_the_sample_weighs_0():
    """An ion list shared across windows names Y, which this sample does not
    hold: Y is in the window by m/z, in no MS2 spectrum and not in the MS1
    window. Its MS1 share is a measured 0, so it weighs 0: C10H17+ (which Y
    could make by composition) stays X's, by X's survivor, and X's F and E50
    are what they are without Y."""
    steps = [(e, {X: x, **({"C10H17+": 1000 - x} if x < 1000 else {})})
             for e, x in zip(NCE, X_I)]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(X) + 0.0001, 1.0)])
    alone = AN.analyze_run(t, _ions(X))
    both = AN.analyze_run(t, _ions(X, Y))
    assert both.windows[0].co_isolates == [Y]
    r = both.fragments.set_index("fragment").loc["C10H17+"]
    # the one candidate the sample holds: as if Y were not named, said
    assert (r["candidates"], r["call"], r["basis"], r["weight_source"]) == \
        (f"{X};{Y}", X, "composition", "composition")
    assert _weights(r["weights"]) == {X: 1.0, Y: 0.0}
    assert r["test_note"].startswith(f"{Y}: in no MS2 spectrum of the window nor in the "
                                     "MS1 window (absent from this sample): weight 0, set aside")
    for res in (alone, both):
        _, f, _ = _curve(res, p, X)
        assert np.allclose(f, np.array(X_I) / 1000.0)
        assert _e50_row(res, p, X)["e50_nce"] == pytest.approx(10 + 200 / 300 * 10)
    assert any(n.startswith(f"{Y} is named inside the window, but no MS2 spectrum holds") and
               "weighs 0" in n for n in both.windows[0].notes)
    # the weights: a measured MS1 share of 0 is a share; a missing one is not
    assert AT.share_weights([X, Y], {X: 5.0, Y: 0.0}, {X: 0.3, Y: 0.0}) == \
        ({X: 1.0, Y: 0.0}, "ms2 survivors", "")
    # X is in the MS2 spectrum: it keeps its survivor whatever MS1 holds
    assert AT.share_weights([X, Y], {X: 5.0}, {X: 0.0, Y: 0.0})[:2] == \
        ({X: 1.0, Y: 0.0}, "ms2 survivors")
    assert AT.share_weights([X, Y], {X: 5.0}, {X: 0.3, Y: NAN})[1] == "equal"


def test_a_fragment_only_absent_ions_can_make_is_no_ions():
    """Y is named but absent (in no MS2 spectrum, not in the MS1 window), and
    C5H9O+ is a fragment only Y can make (X holds no O). It goes to no window
    ion -- call '', basis 'absent only', every weight 0, said -- so Y shows no
    breakdown curve (F, F_lo and S NaN, E50 'no_data') instead of an F of 0
    read off a peak the sample's own ions cannot make; X reads as without Y."""
    steps = [(e, {X: x, "C5H9O+": 50.0}) for e, x in zip(NCE, X_I)]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(X) + 0.0001, 1.0)])
    alone = AN.analyze_run(t, _ions(X))
    both = AN.analyze_run(t, _ions(X, Y))
    r = both.fragments.set_index("fragment").loc["C5H9O+"]
    assert (r["candidates"], r["call"], r["basis"], r["weight_source"]) == \
        (Y, "", AT.BASIS_ABSENT, "none")
    assert _weights(r["weights"]) == {Y: 0.0}
    assert r["test_note"] == (f"only {Y} reads it, and it is in no MS2 spectrum of the window nor "
                              "in the MS1 window (absent from this sample): no window ion takes "
                              "it")
    yb = both.breakdown[both.breakdown["ion"] == Y]
    assert yb[["f", "f_lo", "s"]].isna().all().all()
    ye = both.e50[both.e50["ion"] == Y]
    assert len(ye) and set(ye["flag"]) == {"no_data"}
    # X reads as without Y -- its ion budget too (Y's peak is unexplained)
    cols = ["ion", "f", "f_lo", "s", "budget_explained", "budget_unexplained", "n_unexplained"]
    pd.testing.assert_frame_equal(alone.breakdown[cols], both.breakdown[
        both.breakdown["ion"] == X][cols].reset_index(drop=True))
    ecols = ["ion", "curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(alone.e50[ecols], both.e50[both.e50["ion"] == X][ecols]
                                  .reset_index(drop=True))
    assert both.branching[both.branching["ion"] == Y].empty
    # the constants a call turns on are in the manifest
    for k, v in (("onset_stray_share", AT.ONSET_STRAY_SHARE),
                 ("satellite_ratio", AT.SATELLITE_RATIO), ("satellite_fwhm", AT.SATELLITE_FWHM),
                 ("polyoxy_max_c", AT.POLYOXY_MAX_C), ("polyoxy_min_o", AT.POLYOXY_MIN_O),
                 ("intact_fraction", AT.INTACT_FRACTION), ("branching_flag", AT.BRANCHING_FLAG)):
        assert both.params[k] == v, k


def test_an_ion_located_only_at_a_later_energy_is_not_absent():
    """The target X has no peak at NCE 1 (only its own fragment C10H17+, in
    source) and is located from NCE 10; the MS1 window holds one unrelated
    centroid. It is not absent -- absence is read over every energy -- so its
    fragment stays its own and F falls 0, .7, .4, .1, not 'intact' at 1."""
    steps = [(1, {"C10H17+": 100.0}), (10, {X: 700.0, "C10H17+": 300.0}),
             (20, {X: 400.0, "C10H17+": 600.0}), (30, {X: 100.0, "C10H17+": 900.0})]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(X) + 0.1, 1.0)])
    res = AN.analyze_run(t, _ions(X))
    r = res.fragments.set_index("fragment").loc["C10H17+"]
    assert (r["call"], r["basis"]) == (X, "composition")
    _, f, g = _curve(res, p, X)
    assert np.allclose(f, [0.0, 0.7, 0.4, 0.1]) and np.allclose(g["f_lo"].to_numpy(), f)
    # F is already 0 at NCE 1 (only the in-source fragment there): below range
    assert _e50_row(res, p, X)["flag"] == "below_range"


def test_an_absent_ion_is_in_no_curve_through_a_shared_fragment():
    """Absent Y (in no MS2 spectrum, not in the MS1 window) could make the
    shared C10H17+ and owns C5H9O+: it shows no curve at all -- F, F_lo and S
    NaN, E50 'no_data' -- not an F_lo of 0 from the fragment it shares."""
    steps = [(e, {X: x, "C10H17+": 1000.0 - x, "C5H9O+": 30.0}) for e, x in zip(NCE, X_I)]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(X) + 0.0001, 1.0)])
    res = AN.analyze_run(t, _ions(X, Y))
    yb = res.breakdown[res.breakdown["ion"] == Y]
    assert yb[["f", "f_lo", "s"]].isna().all().all()
    assert set(res.e50.loc[res.e50["ion"] == Y, "flag"]) == {"no_data"}
    assert res.fragments.set_index("fragment").loc["C10H17+", "call"] == X


def test_the_ms1_shares_are_one_to_one():
    """Two named ions near one MS1 centroid: the nearer takes its share, the
    other 0 (a peak's share is never counted twice); a named ion no centroid
    matches has 0; no window row for the parent -> {} (unknown)."""
    p = 150.0
    win = _window_rows(p, [(150.0002, 0.7), (150.1, 0.3)])
    ions = {"a": 150.0, "b": 150.0012, "c": 149.95}
    assert AN._ms1_shares(win, p, ions, 0.3) == {"a": 0.7, "b": 0.0, "c": 0.0}
    assert AN._ms1_shares(win, p, {"b": 150.0012}, 0.3) == {"b": 0.7}
    assert AN._ms1_shares(win, 151.0, ions, 0.3) == {}


def test_named_aliases_are_not_both_window_ions(nh4):
    """C10H14NO^N+ lies 0.44 mDa above C11H15O2+ (the CHO / N^N alias): inside
    the reading tolerance, so the precursor peak reads as both. Named
    together, the alias is left out of the 179.107 window -- said -- and the
    window reads as it does without it."""
    res = AN.analyze_run(_tables("ms2_nh4_exploratory"), _ions(*NH4_IONS, "C10H14NO^N+"))
    w = [x for x in res.windows if x.parent_mz == 179.1069][0]
    assert (w.target, w.co_isolates, w.left_out) == ("C11H15O2+", ["C11H16^NO+"],
                                                     ["C10H14NO^N+"])
    assert any(n.startswith("C10H14NO^N+ left out: +0.44 mDa from C11H15O2+") for n in w.notes)
    for name in ("breakdown", "e50", "fragments", "branching"):
        assert getattr(res, name).equals(getattr(nh4, name)), name
    assert _e50_row(res, 179.1069, "C11H15O2+")["e50_nce"] == pytest.approx(21.106, abs=1e-3)
    assert AN.analyze_run(_tables("ms2_nh4_exploratory"), _ions(*NH4_IONS)).windows[3] \
        .manifest()["left_out"] == []
    # between two co-isolates the one named first is kept; the target always is
    ions = {"T": 150.0, "b": 150.1, "a": 150.1004, "far": 150.15}
    assert AN.drop_aliases("T", ["b", "a", "far"], ions, tol_ppm=5, tol_mda=0.3)[:2] == \
        (["b", "far"], ["a"])
    ions = {"T": 150.0, "a": 150.1004, "b": 150.1, "t2": 150.0005}
    co, out, notes = AN.drop_aliases("T", ["t2", "b", "a"], ions, tol_ppm=5, tol_mda=0.3)
    assert (co, out) == (["a"], ["b", "t2"])                  # in the order named
    assert "kept a (named first)" in notes[0]
    assert "kept T (the target, nearest the located precursor)" in notes[1]


def _kinds(steps):
    """RunTables for X's window from (activation, {ion: intensity}) steps;
    volts 4 + 0.2 NCE for HCD, none for CID."""
    parent = round(_mz(X), 4)
    cent, groups = {}, []
    for act, peaks in steps:
        cent[S.spectrum_key(parent, act)] = {
            "parent_peak_mz": parent, "activation": act, "mz": [_mz(k) for k in peaks],
            "intensity": list(peaks.values())}
        e = float(act[3:])
        groups.append({"parent_peak_mz": parent, "activation": act, "scan_count": 5,
                       "hcd_energy": [4.0 + 0.2 * e] if act.startswith("hcd") else [],
                       "t_min": 0.0, "t_max": 1.0})
    summ = {"parent_peaks": [parent], "groups": groups, "isolation_width": 0.4,
            "ms1_scan_count": 1, "ms2_scan_count": 5 * len(steps)}
    return AN.RunTables(spectra=S.spectra_table("S", cent, summ),
                        parents=S.parents_table("S", cent, summ),
                        window=S._frame({}, S.WINDOW_COLUMNS), manifest={})


def test_one_activation_kind_per_breakdown():
    """An NCE of CID is not an NCE of HCD: a parent with both is broken down on
    the kind with the most spectra (HCD here), the other read but left out --
    said -- and the E50 is the HCD curve's (16.67, on volts 7.33)."""
    parent = round(_mz(X), 4)
    steps = [("hcd1.00", {X: 1000.0}), ("hcd10.00", {X: 700.0, "C10H17+": 300.0}),
             ("cid20.00", {X: 50.0, "C10H17+": 950.0}),
             ("hcd30.00", {X: 100.0, "C10H17+": 900.0})]
    t = _kinds(steps)
    assert list(t.parents["kind"]) == ["hcd", "hcd", "cid", "hcd"]
    res = AN.analyze_run(t, _ions(X))
    (w,) = res.windows
    assert (w.status, w.activation_kind) == ("analysed", "hcd")
    assert any(n.startswith("spectra of 2 activation kinds") and "(1 cid)" in n for n in w.notes)
    assert list(res.breakdown["nce"]) == [1.0, 10.0, 30.0]
    assert f"{parent!r}@cid20.00" in set(res.readings["spectrum"])
    r = _e50_row(res, w.parent_mz, X)
    assert (r["flag"], r["monotonic"]) == ("interpolated", True)
    assert r["e50_nce"] == pytest.approx(10 + 200 / 600 * 20)
    assert r["e50_v"] == pytest.approx(4 + 0.2 * r["e50_nce"])
    assert w.manifest()["activation_kind"] == "hcd"
    # a tie (2 HCD, 2 CID) goes to HCD: its NCEs, its volts
    tie = [("hcd1.00", {X: 1000.0}), ("cid5.00", {X: 900.0, "C10H17+": 100.0}),
           ("cid20.00", {X: 300.0, "C10H17+": 700.0}), ("hcd30.00", {X: 100.0, "C10H17+": 900.0})]
    res = AN.analyze_run(_kinds(tie), _ions(X))
    (w,) = res.windows
    assert (w.activation_kind, list(res.breakdown["nce"])) == ("hcd", [1.0, 30.0])
    assert any("uses the 2 hcd spectra" in n and "(2 cid)" in n for n in w.notes)
    r = _e50_row(res, w.parent_mz, X)
    assert r["e50_nce"] == pytest.approx(1 + 500 / 900 * 29)
    assert r["e50_v"] == pytest.approx(4 + 0.2 * r["e50_nce"]) and r["note"] == ""
    # a CID majority is broken down on CID: its NCEs, no volts (said), and
    # the manifest says 'cid'
    cid = [("hcd1.00", {X: 1000.0}), ("cid5.00", {X: 900.0, "C10H17+": 100.0}),
           ("cid10.00", {X: 600.0, "C10H17+": 400.0}), ("cid20.00", {X: 300.0, "C10H17+": 700.0})]
    res = AN.analyze_run(_kinds(cid), _ions(X))
    (w,) = res.windows
    assert (w.activation_kind, list(res.breakdown["nce"])) == ("cid", [5.0, 10.0, 20.0])
    assert any("uses the 3 cid spectra" in n and "(1 hcd)" in n for n in w.notes)
    assert (w.manifest()["activation_kind"], w.manifest()["volts_known"]) == ("cid", False)
    r = _e50_row(res, w.parent_mz, X)
    assert r["e50_nce"] == pytest.approx(10 + 100 / 300 * 10)
    assert math.isnan(r["e50_v"]) and r["note"] == "volts unknown"


def test_an_excluded_candidate_weighs_0_among_three():
    """Three window ions can make C6H9+. Y is intact where it appears (the
    intact test excludes it); X and Z are both losing: ambiguous between them,
    by their survivors (1000 : 500), and Y weighs 0 -- its F stays 1."""
    Z = "C9H15O2+"
    steps = [(1, {X: 1000.0, Y: 1000.0, Z: 500.0}),
             (10, {X: 600.0, Y: 1000.0, Z: 300.0, "C6H9+": 600.0}),
             (20, {X: 200.0, Y: 1000.0, Z: 100.0, "C6H9+": 1200.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y, Z))
    (w,) = res.windows
    assert w.co_isolates == [Z, Y]
    r = res.fragments.set_index("fragment").loc["C6H9+"]
    assert (r["candidates"], r["call"], r["excluded"], r["weight_source"]) == \
        (f"{X};{Z};{Y}", "ambiguous", Y, "ms2 survivors")
    assert _weights(r["weights"]) == pytest.approx({X: 2 / 3, Z: 1 / 3, Y: 0.0})
    _, fy, _ = _curve(res, w.parent_mz, Y)
    assert np.allclose(fy, 1.0) and _e50_row(res, w.parent_mz, Y)["flag"] == "above_range"
    _, fx, _ = _curve(res, w.parent_mz, X)
    _, fz, _ = _curve(res, w.parent_mz, Z)
    assert np.allclose(fx, [1.0, 0.6, 0.2]) and np.allclose(fz, [1.0, 0.6, 0.2])


def test_a_precursor_that_does_not_fragment():
    """A window with no fragment at all (and one with an unexplained peak):
    F = 1 at every energy, E50 above range -- not a crash."""
    for extra in ({}, {120.5: 30.0}):
        steps = [(e, {"C10H17+": v, **extra}) for e, v in ((1, 1000.0), (10, 900.0),
                                                           (20, 800.0))]
        res = AN.analyze_run(_synthetic(round(_mz("C10H17+"), 4), steps), _ions("C10H17+"))
        (w,) = res.windows
        assert w.status == "analysed" and len(res.fragments) == 0 and len(res.branching) == 0
        _, f, g = _curve(res, w.parent_mz, "C10H17+")
        assert np.allclose(f, 1.0) and np.allclose(g["f_lo"], 1.0)
        assert set(res.e50.loc[res.e50["curve"] != "S", "flag"]) == {"above_range"}
    # the matrices of a window with no fragment (precursor + unexplained), no
    # precursor either, or no peak at all: no NaN fragment key, ever
    roles = AT.peak_roles(res.windows[0].readings)
    specs = list(dict.fromkeys(roles["spectrum"]))
    assert set(roles["role"]) == {"precursor", "unexplained"}
    for sel in (roles, roles[roles["role"] == "unexplained"], roles.iloc[:0]):
        ion_i, frag_i, ion_n = AT.intensity_matrices(sel, specs, ["C10H17+"])
        assert list(frag_i.columns) == [] and list(frag_i.index) == specs
        assert list(ion_i.columns) == ["C10H17+"] and list(ion_n.columns) == ["C10H17+"]
    ion_i, _, ion_n = AT.intensity_matrices(roles, specs, ["C10H17+"])
    assert list(ion_i["C10H17+"]) == [1000.0, 900.0, 800.0] and (ion_n["C10H17+"] == 1).all()
    # ... and a window with fragments but no precursor reading
    fr = roles.assign(role="fragment", key="F+", candidates="C10H17+")
    _, frag_i, _ = AT.intensity_matrices(fr[fr["mz"] < 130], specs, ["C10H17+"])
    assert list(frag_i.columns) == ["F+"]


def test_tight_tolerances_still_analyse_every_window():
    """At 0.5 ppm with no mDa floor standard A's 155 windows lose every
    fragment reading at some energy: each window is still analysed."""
    res = AN.analyze_run(_tables("ms2_monoterpene_a"), _ions(*MT_IONS), tol_ppm=0.5, tol_mda=0)
    assert [w.status for w in res.windows] == ["analysed"] * 3
    assert (res.params["tol_ppm"], res.params["tol_mda"]) == (0.5, 0.0)
    ppm = res.readings.loc[res.readings["rank"] > 0, "ppm"]
    assert np.abs(ppm).max() <= 0.5 + 1e-9


# ------------------------------------------------------------------ E50 ----
def test_e50_edge_cases():
    e = e50([1, 2, 3], [0.9, 0.7, 0.3], [4.0, 5.0, 6.0])
    assert (e.flag, e.nce, e.volts, e.monotonic) == ("interpolated", 2.5, 5.5, True)
    e = e50([1, 2, 3], [0.9, 0.8, 0.6])                       # never crosses
    assert (e.flag, math.isnan(e.nce), e.nce_max) == ("above_range", True, 3.0)
    e = e50([1, 2, 3], [0.45, 0.3, 0.1])                      # starts below
    assert (e.flag, math.isnan(e.nce)) == ("below_range", True)
    e = e50([5], [0.8])
    assert (e.flag, e.n_energies) == ("single_energy", 1)
    assert e50([1, 2], [NAN, NAN]).flag == "no_data"
    # non-monotonic: the FIRST downward crossing, flagged
    e = e50([1, 2, 3, 4, 5], [0.9, 0.4, 0.7, 0.6, 0.2])
    assert (e.flag, e.nce, e.monotonic) == ("interpolated", 1.8, False)
    # a rise within MONOTONIC_TOL is noise, not a flag
    assert e50([1, 2, 3], [0.9, 0.91, 0.2]).monotonic
    assert not e50([1, 2, 3], [0.9, 0.93, 0.2]).monotonic
    # exactly 0.5 at a step is not yet below it
    e = e50([1, 2, 3], [0.8, 0.5, 0.2])
    assert e.nce == pytest.approx(2.0) and e50([1, 2], [0.8, 0.5]).flag == "above_range"
    # volts unknown: NCE only
    e = e50([1, 2, 3], [0.9, 0.7, 0.3], [NAN, NAN, NAN])
    assert e.nce == 2.5 and math.isnan(e.volts)
    e = e50([1, 2, 3], [0.9, 0.7, 0.3], [4.0, NAN, 6.0])
    assert e.nce == 2.5 and math.isnan(e.volts)
    # input order does not matter; a NaN energy is dropped
    assert e50([3, 1, 2], [0.3, 0.9, 0.7]).nce == 2.5
    assert e50([1, NAN, 2, 3], [0.9, 0.1, 0.7, 0.3]).nce == 2.5
    # an unobserved energy between the bracketing points: a gap, not a line
    # -- and a hole is not a rise: monotonic is judged between OBSERVED energies
    e = e50([1, 11, 21, 81], [1.0, NAN, NAN, 0.0])
    assert (e.flag, math.isnan(e.nce), e.n_energies, e.monotonic) == ("gap", True, 2, True)
    # ... but one AFTER the crossing does not matter
    e = e50([1, 2, 3, 4], [0.9, 0.3, NAN, 0.1])
    assert (e.flag, e.monotonic) == ("interpolated", True)
    assert not e50([1, 2, 3, 4], [0.9, 0.3, NAN, 0.4]).monotonic     # a rise across a hole
    # a hole of ONE energy is a gap too
    e = e50([1, 2, 3], [0.9, NAN, 0.2])
    assert (e.flag, math.isnan(e.nce), e.n_energies, e.monotonic) == ("gap", True, 2, True)
    # unobserved at the lowest energy: the first OBSERVED point decides
    e = e50([1, 2, 3], [NAN, 0.4, 0.2])
    assert (e.flag, e.nce_min) == ("below_range", 2.0)
    assert e50([1, 2, 3], [NAN, 0.9, 0.2]).nce == pytest.approx(2 + 0.4 / 0.7)
    # starting exactly at 0.5 is not below range: the crossing is AT the first energy
    e = e50([1, 2], [0.5, 0.2])
    assert (e.flag, e.nce) == ("interpolated", 1.0)
    # a NaN energy is dropped, value and all: not an energy, not a point of the span
    e = e50([1, 2, NAN], [0.9, 0.8, 0.1])
    assert (e.flag, e.n_energies, e.nce_max) == ("above_range", 2, 2.0)
    e = e50([1, NAN, 2, 3], [0.9, 0.1, 0.7, 0.3])
    assert (e.n_energies, e.nce_min, e.nce_max) == (3, 1.0, 3.0)


def test_e_cm():
    assert e_cm(10.0, 137.13248) == pytest.approx(10 * 28.0061 / (28.0061 + 137.13248))
    assert e_cm(10.0, 100.0, gas_mass=39.9624) == pytest.approx(10 * 39.9624 / 139.9624)
    assert np.allclose(e_cm(np.array([4.0, 8.0]), 100.0), np.array([4.0, 8.0]) * 28.0061 / 128.0061)
    assert BD.N2_MASS == 28.0061
    for bad in (0, -1, NAN):
        with pytest.raises(ValueError):
            e_cm(1.0, 100.0, gas_mass=bad)
    with pytest.raises(ValueError):
        e_cm(1.0, 0.0)


def test_survival_onset_and_budget_helpers():
    ion = pd.DataFrame({"x": [100.0, 50.0, 0.0]}, index=["a", "b", "c"])
    frag = pd.DataFrame({"f": [0.0, 50.0, 0.0]}, index=["a", "b", "c"])
    c = BD.survival(ion, frag, {"f": {"x": 1.0}}, {"f": ["x"]})
    assert list(c["f"]["x"][:2]) == [1.0, 0.5] and math.isnan(c["f"]["x"]["c"])   # 0/0
    # S is unobserved where F is (0/0), not 0: a 0 would be read as a crossing
    assert list(c["s"]["x"][:2]) == [1.0, 0.5] and math.isnan(c["s"]["x"]["c"])
    gone = BD.survival(pd.DataFrame({"x": [0.0, 5.0]}), pd.DataFrame(index=[0, 1]), {}, {})
    assert gone["s"]["x"].isna().all() and list(gone["f"]["x"].fillna(-1)) == [-1, 1.0]
    assert BD.onset([1, 5, 9], [False, True, True]) == 5.0
    assert math.isnan(BD.onset([1, 5], [False, False]))


def test_window_ions_target_and_pad():
    """The target is the named ion NEAREST the located precursor within 10 ppm;
    the co-isolates are the named ions within width/2 + 5 mDa of the key."""
    ions = {"A": 150.0, "B": 150.0008, "far": 150.0016}
    assert AN.window_ions(150.0, 150.0001, ions, 0.4, 0.3)[0] == "A"
    assert AN.window_ions(150.0, 150.0007, ions, 0.4, 0.3)[0] == "B"
    t, co, note = AN.window_ions(150.0, 149.9980, ions, 0.4, 0.3)     # 13 ppm from A
    assert (t, co) == (None, []) and note.startswith("skipped: no named ion matches")
    edge = {"T": 150.0, "in": 150.0 + 0.2 + 0.0049, "out": 150.0 - 0.2 - 0.0051}
    assert AN.window_ions(150.0, 150.0, edge, 0.4, 0.3)[:2] == ("T", ["in"])
    t, co, note = AN.window_ions(150.0, 150.0, edge, NAN, 0.3)
    assert (t, co) == ("T", []) and note.startswith("isolation width unknown")


# ------------------------------------------------------- ions, run folder ----
def test_named_ions_are_checked():
    assert AN.parse_ion_spec("137.1323=C10H17+")[0] == "C10H17+"
    assert AN.parse_ion_spec(" 137.13 = C10H17+ ")[2] == pytest.approx(_mz("C10H17+"))
    for bad, msg in (("C10H17+", "expected MZ=FORMULA"), ("x=C10H17+", "is not an m/z"),
                     ("137.2=C10H17+", "a typo?"), ("137.13=C10H17", "not a singly-charged")):
        with pytest.raises(ValueError, match=msg):
            AN.parse_ion_spec(bad)
    with pytest.raises(ValueError, match="given twice"):
        AN.collect_ions([AN.parse_ion_spec("137.13=C10H17+")] * 2)
    with pytest.raises(ValueError, match="no ions given"):
        AN.collect_ions([])
    with pytest.raises(ValueError, match="both polarities"):
        AN.collect_ions([AN.parse_ion_spec("137.13=C10H17+"),
                         AN.parse_ion_spec("62.9854=^NO3-")])


def test_a_window_whose_target_is_not_named_is_skipped():
    res = AN.analyze_run(_tables("ms2_monoterpene_a"), _ions("C10H17+"))
    st = {w.parent_mz: (w.status, w.target) for w in res.windows}
    assert st == {137.1321: ("analysed", "C10H17+"), 155.1429: ("skipped", ""),
                  155.1559: ("skipped", "")}
    skipped = [w for w in res.windows if w.status == "skipped"][0]
    assert skipped.notes[0].startswith("skipped: no named ion matches the precursor located at")
    assert set(res.breakdown["parent_mz"]) == {137.1321}


def _write_tables(d, t, *, energy_resolved=True):
    """Write RunTables as the `ms2 extract` folder they stand for."""
    d.mkdir(parents=True, exist_ok=True)
    t.spectra.to_parquet(d / "ms2_spectra.parquet", index=False)
    t.parents.to_csv(d / "ms2_parents.csv", index=False)
    t.window.to_csv(d / "ms2_window.csv", index=False)
    man = {"sample_id": "SAMPLE2xxxxxxxxx", "status": "ok", "energy_resolved": energy_resolved,
           "parameters": {"isolation_width_used":
                          (t.manifest.get("parameters") or {}).get("isolation_width_used")}}
    (d / "ms2_manifest.json").write_text(json.dumps(man))
    return d


def _write_run(d, stem, iso=None):
    return _write_tables(d, _tables(stem, iso=iso))


def _extract(cent, summ, ms1=None):
    """What `ms2 extract` builds from a centroid answer + summary (+ MS1)."""
    spectra, parents = S.spectra_table("S", cent, summ), S.parents_table("S", cent, summ)
    width = (summ or {}).get("isolation_width")
    window = (S.window_table("S", _load(ms1), sorted(set(parents["parent_mz"])), width)
              if ms1 and width else S._frame({}, S.WINDOW_COLUMNS))
    return AN.RunTables(spectra=spectra, parents=parents, window=window,
                        manifest={"sample_id": "S",
                                  "parameters": {"isolation_width_used": width}})


def test_energy_averaged_spectra_are_read_but_not_broken_down(mta):
    """A parent with energy-resolved AND energy-averaged spectra: the averaged
    one is read (NCE NaN) but has no place on the energy axis -- the breakdown
    is exactly the resolved spectra's."""
    mixed = dict(_load("ms2_monoterpene_a_centroids.json"))
    mixed.update(_load("ms2_monoterpene_a_centroids_averaged.json"))
    res = AN.analyze_run(_extract(mixed, _load("ms2_monoterpene_a_summary.json"),
                                  "ms2_monoterpene_a_ms1.json"), _ions(*MT_IONS))
    for name in ("breakdown", "e50", "fragments", "branching"):
        assert getattr(res, name).equals(getattr(mta, name)), name
    # the offset is the lowest ENERGY's located precursor, never the averaged
    # spectrum's (-1.2138 ppm instead of -1.0651 in the 137.1321 window), so
    # the resolved spectra read exactly as in the resolved-only run
    assert [(w.parent_mz, w.precursor_mz, w.offset_ppm) for w in res.windows] == \
        [(w.parent_mz, w.precursor_mz, w.offset_ppm) for w in mta.windows]
    assert res.readings[res.readings["nce"].notna()].reset_index(drop=True).equals(mta.readings)
    avg = res.readings[res.readings["nce"].isna()]
    assert set(avg["spectrum"]) == {"137.1321", "155.1429", "155.1559"} and (avg["rank"] > 0).any()
    assert all(w.notes[0] == "1 energy-averaged spectrum(s) read but left out of the breakdown"
               for w in res.windows)


def test_load_run_reads_what_extract_writes(tmp_path):
    d = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    t = AN.load_run(str(d))
    want = _tables("ms2_monoterpene_a")
    assert t.spectra.equals(want.spectra)
    assert {c: str(x) for c, x in t.parents.dtypes.items()} == S.PARENTS_COLUMNS
    assert np.allclose(t.parents["precursor_mz"], want.parents["precursor_mz"])
    assert len(t.window) == len(want.window) and set(t.inputs) == set(AN.RUN_FILES)
    # the CSV copy of the spectra is read when the parquet is missing
    want.spectra.to_csv(d / "ms2_spectra.csv", index=False)
    (d / "ms2_spectra.parquet").unlink()
    t2 = AN.load_run(str(d))
    assert len(t2.spectra) == len(want.spectra) and (t2.spectra["activation"] != "").all()
    with pytest.raises(FileNotFoundError):
        AN.load_run(str(tmp_path / "nowhere"))


# ------------------------------------------------------------------ CLI ----
def _sub(*path):
    p = cli.build_parser()
    for name in path:
        p = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction"
                 ).choices[name]
    return p


def test_cli_parses_analyze():
    P = cli.build_parser()
    a = P.parse_args(["ms2", "analyze", "--run-dir", "D", "--ion", "137.13=C10H17+"])
    assert a.func is cli.cmd_ms2 and a.ms2_verb == "analyze"
    assert (a.run_dir, a.ion, a.ions, a.k_max, a.no_ladder_k4) == ("D", ["137.13=C10H17+"], None,
                                                                   None, False)
    assert (a.tol_ppm, a.tol_mda, a.gas_mass, a.iso_width, a.no_figures, a.out_dir) == \
        (5.0, 0.3, BD.N2_MASS, None, False, None)
    with pytest.raises(SystemExit):
        P.parse_args(["ms2", "analyze", "--run-dir", "D", "--k-max", "4", "--no-ladder-k4"])
    with pytest.raises(SystemExit):
        P.parse_args(["ms2", "analyze"])
    help_text = _sub("ms2", "analyze").format_help()
    assert "--no-ladder-k4" in help_text
    # the metavars docs/MS2.md §11.7 writes
    assert "--k-max K " in help_text and "--out-dir DIR" in help_text and "K_MAX" not in help_text


@pytest.fixture
def no_server(monkeypatch):
    """`ms2 analyze` must not need credentials or a client."""
    def boom(*a, **k):
        raise AssertionError("ms2 analyze asked for Mascope")
    from peaky.io import io_mascope as IO
    monkeypatch.setattr(cli, "_require_creds", boom)
    monkeypatch.setattr(IO, "connect", boom)


def test_cli_analyze_writes_the_tables_figures_and_manifest(tmp_path, no_server, capsys):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
                     "--ion", "155.1558=C10H20^N+", "--ion", "155.1428=C10H19O+",
                     "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    names = {p.name for p in out.iterdir()}
    assert names == {"ms2_readings.csv", "ms2_breakdown.csv", "ms2_e50.csv", "ms2_fragments.csv",
                     "ms2_branching.csv", "ms2_analyze_manifest.json", "breakdown_137.1321.png",
                     "breakdown_155.1429.png", "breakdown_155.1559.png"}
    for png in out.glob("*.png"):
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 20_000
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert man["sample_id"] == "SAMPLE2xxxxxxxxx"
    assert man["parameters"]["gate"] == "ladder-k4" and man["parameters"]["k_max_read"] == 4
    assert man["parameters"]["isolation_width_used"] == 0.4
    assert man["parameters"]["isolation_width_source"] == "extract manifest"
    assert man["parameters"]["gas_mass"] == 28.0061
    win = {w["parent_mz"]: w for w in man["windows"]}
    assert win[137.1321]["offset_ppm"] == pytest.approx(-1.0651, abs=1e-4)
    assert win[155.1559]["co_isolates"] == ["C10H19O+"] and win[155.1559]["volts_known"]
    assert man["counts"]["four_piece_readings"] == 0 and man["counts"]["analysed"] == 3
    assert set(man["inputs"]) == set(AN.RUN_FILES)
    from peaky.reporting import provenance as PV
    assert man["files"]["ms2_e50.csv"] == PV.sha1_file(str(out / "ms2_e50.csv"))
    assert man["code"]["analyze_version"] == AN.__version__
    assert AN.S_NOTE in man["notes"]
    e = pd.read_csv(out / "ms2_e50.csv", keep_default_na=False, na_values=[""])
    assert list(e.columns) == list(AN.E50_COLUMNS)
    r = e[(e["parent_mz"] == 137.1321) & (e["curve"] == "F")].iloc[0]
    assert r["e50_nce"] == pytest.approx(13.891, abs=1e-3)
    for name, spec in (("ms2_readings.csv", RD.READINGS_COLUMNS),
                       ("ms2_breakdown.csv", AN.BREAKDOWN_COLUMNS),
                       ("ms2_fragments.csv", AN.FRAGMENT_COLUMNS),
                       ("ms2_branching.csv", AN.BRANCHING_COLUMNS)):
        assert list(pd.read_csv(out / name, nrows=0).columns) == list(spec), name
    assert "parent 137.1321: analysed; target C10H17+; offset -1.07 ppm; E50 NCE 13.9" in printed
    # the console names every file written, the figures among them
    (wrote,) = [ln for ln in printed.splitlines() if ln.startswith("[ms2] wrote ")]
    assert wrote == ("[ms2] wrote ms2_readings.csv, ms2_breakdown.csv, ms2_e50.csv, "
                     "ms2_fragments.csv, ms2_branching.csv, breakdown_137.1321.png, "
                     "breakdown_155.1429.png, breakdown_155.1559.png, "
                     f"ms2_analyze_manifest.json -> {out}")
    assert "(6.33 V, E_cm 1.073 eV)" in printed
    for f in out.iterdir():
        assert "://" not in f.read_bytes().decode("latin-1"), f.name


def test_cli_analyze_gate_modes_and_defaults(tmp_path, no_server, capsys):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_b")
    # default out-dir: the run folder itself; no width known: target only, said
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.13=C10H17+",
                     "--no-figures"]) == 0
    printed = capsys.readouterr().out
    assert (run / "ms2_e50.csv").exists() and not list(run.glob("*.png"))
    assert "isolation width unknown" in printed and "E50 NCE 15.8" in printed
    for argv, gate, k in ((["--k-max", "4"], "ungated", 4), (["--no-ladder-k4"], "strict-k3", 3),
                          (["--k-max", "3"], "strict-k3", 3)):
        out = tmp_path / gate / str(k)
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.13=C10H17+",
                         "--no-figures", "--out-dir", str(out)] + argv) == 0
        man = json.loads((out / "ms2_analyze_manifest.json").read_text())
        assert (man["parameters"]["gate"], man["parameters"]["k_max_read"]) == (gate, k)
        if gate == "ungated":
            assert "ungated k_max 4" in man["parameters"]["gate_note"]
    # --iso-width replaces an unknown width
    out = tmp_path / "iso"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.13=C10H17+",
                     "--iso-width", "0.4", "--no-figures", "--out-dir", str(out)]) == 0
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert (man["parameters"]["isolation_width_used"],
            man["parameters"]["isolation_width_source"]) == (0.4, "explicit")


@pytest.mark.parametrize("argv, message", [
    (["--ion", "137.2=C10H17+"], "a typo?"),
    ([], "no ions given"),
    (["--ion", "137.13=C10H17+", "--k-max", "9"], "k_max must be an integer"),
    (["--ion", "137.13=C10H17+", "--tol-ppm", "-1"], "--tol-ppm must be"),
    (["--ion", "137.13=C10H17+", "--gas-mass", "0"], "--gas-mass must be"),
    (["--ion", "137.13=C10H17+", "--iso-width", "0"], "--iso-width must be"),
])
def test_cli_analyze_checks_its_arguments(tmp_path, no_server, argv, message):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_b")
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--no-figures"] + argv)
    assert message in str(exc.value.code)
    assert not (run / "ms2_e50.csv").exists()


def test_cli_analyze_exits_1_when_no_window_has_its_target(tmp_path, no_server):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_b")
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "155.1428=C10H19O+",
                  "--no-figures"])
    assert "no window was analysed" in str(exc.value.code)
    man = json.loads((run / "ms2_analyze_manifest.json").read_text())
    assert man["windows"][0]["status"] == "skipped"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(tmp_path / "missing"), "--ion",
                  "137.13=C10H17+"])
    assert "not a folder" in str(exc.value.code)


MT_ARGS = ["--ion", "137.1323=C10H17+", "--ion", "155.1558=C10H20^N+",
           "--ion", "155.1428=C10H19O+", "--no-figures"]


def test_cli_analyze_exit_codes_follow_the_window_statuses(tmp_path, no_server, capsys):
    """Energy-averaged spectra (a server older than Mascope 1.8, or an averaged
    extract): every window is read, none has an energy axis -- the readings
    are written and the exit is 0, with no --ion advice. No MS2 parent at all,
    or no precursor located anywhere: exit 1, each said as it is."""
    for name, cent, summ in (("old", "ms2_old_server_centroids.json",
                              "ms2_old_server_summary.json"),
                             ("avg", "ms2_monoterpene_a_centroids_averaged.json",
                              "ms2_monoterpene_a_summary.json")):
        run = _write_tables(tmp_path / name, _extract(_load(cent), _load(summ),
                                                      "ms2_monoterpene_a_ms1.json"),
                            energy_resolved=False)
        assert cli.main(["ms2", "analyze", "--run-dir", str(run)] + MT_ARGS) == 0, name
        printed = capsys.readouterr().out
        tail = printed.split("[ms2] no window has energy-resolved spectra")[1]
        assert "the readings are written" in tail and "--ion" not in tail
        man = json.loads((run / "ms2_analyze_manifest.json").read_text())
        assert [(w["status"], w["reason"]) for w in man["windows"]] == \
            [("readings only", AN.NOT_RESOLVED)] * 3
        assert [w["target"] for w in man["windows"]] == ["C10H17+", "C10H19O+", "C10H20^N+"]
        assert (man["counts"]["readings_only"], man["counts"]["breakdown_rows"]) == (3, 0)
        rd = pd.read_csv(run / "ms2_readings.csv")
        assert (rd["rank"] > 0).any() and rd["nce"].isna().all()
    # an MS1-only sample: no parent -- nothing to analyse, and no --ion advice
    run = _write_tables(tmp_path / "empty", _extract({}, _load("ms2_empty_scaffold_summary.json")),
                        energy_resolved=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run)] + MT_ARGS)
    assert "no MS2 parent in this folder" in str(exc.value.code)
    assert "--ion" not in str(exc.value.code)
    # no precursor located in any spectrum: that is the cause given
    t = _tables("ms2_monoterpene_b")
    t.parents["precursor_mz"] = NAN
    run = _write_tables(tmp_path / "unlocated", t)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.13=C10H17+",
                  "--no-figures"])
    assert str(exc.value.code) == f"[ms2] no window was analysed: {AN.SKIP_NOT_LOCATED}"
    man = json.loads((run / "ms2_analyze_manifest.json").read_text())
    assert (man["windows"][0]["status"], man["windows"][0]["reason"]) == \
        ("skipped", AN.SKIP_NOT_LOCATED)


def test_cli_analyze_mixed_folder_says_which_windows_are_which(tmp_path, no_server, capsys):
    """A folder whose 137.1321 parent is energy-resolved (five NCEs) and whose
    155 parents are energy-averaged, with only the 155 ions named: 137.1321 is
    skipped for want of a named target, the 155 windows are readings only. The
    summary must not say that no window has an energy axis (137.1321 was
    never looked at) and must give the --ion hint for the skipped window; the
    readings are written, so the exit is 0."""
    cent = {k: v for k, v in _load("ms2_monoterpene_a_centroids.json").items()
            if k.startswith("137.1321@")}
    cent.update({k: v for k, v in _load("ms2_monoterpene_a_centroids_averaged.json").items()
                 if k.startswith("155.")})
    run = _write_tables(tmp_path / "mixed", _extract(cent, _load("ms2_monoterpene_a_summary.json"),
                                                     "ms2_monoterpene_a_ms1.json"))
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "155.1558=C10H20^N+",
                     "--ion", "155.1428=C10H19O+", "--no-figures"]) == 0
    printed = capsys.readouterr().out
    assert "no window has energy-resolved spectra" not in printed
    assert ("[ms2] no window was analysed: 2 of 3 windows (155.1429, 155.1559) hold "
            "energy-averaged spectra only") in printed
    assert ("[ms2] skipped with no named ion at the located precursor: 137.1321 -- name each "
            "window's target ion with --ion MZ=FORMULA (or --ions CSV)") in printed
    man = json.loads((run / "ms2_analyze_manifest.json").read_text())
    assert [(w["parent_mz"], w["status"], w["reason"]) for w in man["windows"]] == [
        (137.1321, "skipped", AN.SKIP_NO_TARGET), (155.1429, "readings only", AN.NOT_RESOLVED),
        (155.1559, "readings only", AN.NOT_RESOLVED)]


def test_cli_gate_modes_reach_the_readings(tmp_path, no_server):
    """The gate options change the numbers, not only the manifest: the hexose
    run's fourth step C6H7O3+ at NCE 10 is read gated (with its support) and
    ungated (without), not strict; F at NCE 10 follows."""
    run = _write_tables(tmp_path / "hex", _synthetic(round(_mz(HEX), 4), HEX_RUN))
    for argv, gate, n4, support, f10 in (([], "ladder-k4", 1, "H7NO2", 0.625),
                                         (["--no-ladder-k4"], "strict-k3", 0, None, 1.0),
                                         (["--k-max", "4"], "ungated", 1, "", 0.625)):
        out = tmp_path / gate
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion",
                         f"{_mz(HEX):.4f}={HEX}", "--no-figures", "--out-dir", str(out)]
                        + argv) == 0
        man = json.loads((out / "ms2_analyze_manifest.json").read_text())
        assert (man["parameters"]["gate"], man["counts"]["four_piece_readings"]) == (gate, n4)
        rd = pd.read_csv(out / "ms2_readings.csv", keep_default_na=False,
                         dtype={"ladder_support": str, "formula": str})
        four = rd[rd["loss_pieces"] == 4]
        assert list(four["formula"]) == [CHAIN[4]] * n4
        assert list(four["ladder_support"]) == [support] * n4
        fr = pd.read_csv(out / "ms2_fragments.csv")
        assert (CHAIN[4] in set(fr["fragment"])) is bool(n4)
        bd = pd.read_csv(out / "ms2_breakdown.csv").sort_values("nce")
        assert bd["f"].iloc[1] == pytest.approx(f10), gate


def test_cli_tolerances_and_gas_mass_reach_the_numbers(tmp_path, no_server):
    """--tol-ppm, --tol-mda and --gas-mass are used, not only recorded: each
    tolerance alone changes how many peaks standard A's spectra read (84 at
    5 ppm / 0.3 mDa), as `analyze_run` does with the same values; E_cm on
    argon is the volts x 39.9624 / (39.9624 + m/z)."""
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    t, ii = AN.load_run(str(run)), _ions(*MT_IONS)

    def go(name, *argv):
        out = tmp_path / name
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--out-dir", str(out)]
                        + MT_ARGS + list(argv)) == 0
        return out, json.loads((out / "ms2_analyze_manifest.json").read_text())["parameters"]

    def explained(out):
        return int((pd.read_csv(out / "ms2_readings.csv")["rank"] > 0).sum())
    assert explained(go("base")[0]) == 84
    for argv, kw, n in ((["--tol-ppm", "20"], {"tol_ppm": 20.0}, 86),
                        (["--tol-mda", "2"], {"tol_mda": 2.0}, 91),
                        (["--tol-ppm", "2", "--tol-mda", "0.1"], {"tol_ppm": 2.0, "tol_mda": 0.1},
                         76)):
        out, p = go("_".join(argv).replace("-", ""), *argv)
        assert explained(out) == n == int((AN.analyze_run(t, ii, **kw).readings["rank"] > 0).sum())
        assert (p["tol_ppm"], p["tol_mda"]) == (kw.get("tol_ppm", 5.0), kw.get("tol_mda", 0.3))
    out, p = go("argon", "--gas-mass", "39.9624")
    assert p["gas_mass"] == 39.9624
    e = pd.read_csv(out / "ms2_e50.csv")
    r = e[(e["parent_mz"] == 137.1321) & (e["curve"] == "F")].iloc[0]
    assert r["e50_ecm"] == pytest.approx(r["e50_v"] * 39.9624 / (39.9624 + _mz("C10H17+")))
    assert r["e50_ecm"] == pytest.approx(1.4277, abs=1e-4)


def test_ions_from_a_csv(tmp_path, no_server):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    p = tmp_path / "ions.csv"
    p.write_text("mz,formula\n137.1323,C10H17+\n,C10H20^N+\n155.1428,C10H19O+\n")
    assert [f for f, *_ in AN.read_ions_csv(str(p))] == ["C10H17+", "C10H20^N+", "C10H19O+"]
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(p),
                     "--no-figures"]) == 0
    man = json.loads((run / "ms2_analyze_manifest.json").read_text())
    assert [i["formula"] for i in man["parameters"]["ions"]] == ["C10H17+", "C10H20^N+",
                                                                 "C10H19O+"]
    bad = tmp_path / "bad.csv"
    bad.write_text("mz,ion\n137.13,C10H17+\n")
    with pytest.raises(ValueError, match="needs a 'formula' column"):
        AN.read_ions_csv(str(bad))
    # the mz column is checked as --ion's is: a typo is refused, with its row
    typo = tmp_path / "typo.csv"
    typo.write_text("mz,formula\n137.1323,C10H17+\n137.2,C10H17O+\n")
    with pytest.raises(ValueError,
                       match=r"typo\.csv row 3: the m/z given \(137\.2\) is .* a typo\?"):
        AN.read_ions_csv(str(typo))
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(typo), "--no-figures",
                  "--out-dir", str(tmp_path / "typo_out")])
    assert "a typo?" in str(exc.value.code) and not (tmp_path / "typo_out").exists()


def test_ion_and_ions_are_used_together(tmp_path, no_server):
    """--ion and --ions add up: the --ion target of the 137.1321 window and the
    CSV's two 155 ions -- all three windows analysed. A formula named by both
    is refused."""
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    p = tmp_path / "ions.csv"
    p.write_text("mz,formula\n155.1558,C10H20^N+\n155.1428,C10H19O+\n")
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
                     "--ions", str(p), "--no-figures", "--out-dir", str(out)]) == 0
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert [i["formula"] for i in man["parameters"]["ions"]] == ["C10H17+", "C10H20^N+",
                                                                 "C10H19O+"]
    assert [(w["parent_mz"], w["status"], w["target"]) for w in man["windows"]] == [
        (137.1321, "analysed", "C10H17+"), (155.1429, "analysed", "C10H19O+"),
        (155.1559, "analysed", "C10H20^N+")]
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "155.1428=C10H19O+",
                  "--ions", str(p), "--no-figures", "--out-dir", str(tmp_path / "twice")])
    assert "C10H19O+ is given twice" in str(exc.value.code)


def test_figure_legends_sit_below_the_panels(tmp_path):
    """The legends are outside (below) both axes: no legend box overlaps an
    axes' data area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = AN.analyze_run(_tables("ms2_monoterpene_a"), _ions(*MT_IONS))
    w = [x for x in res.windows if x.parent_mz == 155.1559][0]
    captured = {}
    real_savefig = plt.Figure.savefig

    def grab(fig, path, **kw):
        captured["fig"] = fig
        return real_savefig(fig, path, **kw)
    plt.Figure.savefig = grab
    try:
        assert AN.render_window(w, str(tmp_path / "b.png"))
    finally:
        plt.Figure.savefig = real_savefig
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = captured["fig"]
    canvas = FigureCanvasAgg(fig)            # closing the figure detached its canvas
    canvas.draw()
    r = canvas.get_renderer()
    axes = [a for a in fig.axes if a.get_legend() is not None]
    assert len(axes) == 2
    for ax in axes:
        leg = ax.get_legend().get_window_extent(r)
        box = ax.get_window_extent(r)
        assert leg.y1 < box.y0, "legend overlaps its panel"


# ------------------------------------------- weights chosen per candidate ----
def _shared_pair(p_share=None):
    """X and Y share the in-source fragment C7H11+ (it grows by what both
    lose); their lowest-energy survivors are 1000 : 500, their MS1 window
    shares 0.5 / 0.5. `p_share`: an MS1 centroid at C11H23+ (in the window,
    in no MS2 spectrum; it could make C7H11+ by composition) with that share."""
    steps = [(e, {X: x, Y: y, "C7H11+": 50.0 + (1000 - x) + (500 - y)})
             for e, x, y in zip(NCE, (1000.0, 800.0, 500.0, 200.0), (500.0, 450.0, 350.0, 200.0))]
    t = _synthetic(round(_mz(X), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    rows = [(_mz(X) + 0.0001, 0.5), (_mz(Y) - 0.0001, 0.5)]
    if p_share is not None:
        rows.append((_mz("C11H23+"), p_share))
    t.window = _window_rows(p, rows)
    return t, p


def test_naming_an_absent_ion_moves_no_other_ions_weights():
    """A shared ion list names P (C11H23+), which this sample lacks. Each
    candidate's weight comes from what is measured of IT: X and Y keep their
    survivor split (2/3 : 1/3) and their F and E50s, P weighs 0. Before, one
    candidate without a survivor switched EVERY candidate to its MS1 window
    share: the split became 0.5 / 0.5 / 0 and both E50s moved. With an MS1
    trace of P (1e-4 of the window), P takes that share on the survivors'
    scale -- and X and Y move by no more than it."""
    t, p = _shared_pair()
    two = AN.analyze_run(t, _ions(X, Y))
    three = AN.analyze_run(t, _ions(X, Y, "C11H23+"))
    assert three.windows[0].co_isolates == [Y, "C11H23+"]
    a = two.fragments.set_index("fragment").loc["C7H11+"]
    b = three.fragments.set_index("fragment").loc["C7H11+"]
    assert b["candidates"] == f"{X};{Y};C11H23+"
    assert (a["weight_source"], b["weight_source"]) == ("ms2 survivors", "ms2 survivors")
    assert _weights(a["weights"]) == pytest.approx({X: 2 / 3, Y: 1 / 3})
    assert _weights(b["weights"]) == pytest.approx({X: 2 / 3, Y: 1 / 3, "C11H23+": 0.0})
    cols = ["ion", "curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(two.e50[cols], three.e50[three.e50["ion"] != "C11H23+"][cols]
                                  .reset_index(drop=True))
    same = ["f", "f_lo", "branching_sum", "branching_flag"]
    pd.testing.assert_frame_equal(two.breakdown[same], three.breakdown[
        three.breakdown["ion"] != "C11H23+"][same].reset_index(drop=True))
    # an MS1 trace: P's weight is its share on the survivors' scale, said
    t, p = _shared_pair(p_share=1e-4)
    trace = AN.analyze_run(t, _ions(X, Y, "C11H23+"))
    c = trace.fragments.set_index("fragment").loc["C7H11+"]
    assert c["weight_source"] == "ms2 survivors + ms1 window"
    w = _weights(c["weights"])
    assert w["C11H23+"] == pytest.approx(1e-4 / (1.0 + 1e-4), rel=1e-6)
    assert (w[X], w[Y]) == pytest.approx((2 / 3, 1 / 3), rel=2e-4)
    assert "no MS2 survivor at the lowest energy for C11H23+ (share 0.0001)" in c["test_note"]
    for ion in (X, Y):
        assert _e50_row(trace, p, ion)["e50_nce"] == pytest.approx(
            _e50_row(two, p, ion)["e50_nce"], abs=0.01)
    # an ion present in the MS2 spectrum keeps its survivor when the MS1
    # window lacks a centroid for it: never 0 for want of an MS1 peak
    assert AT.share_weights([X, Y], {X: 1000.0, Y: 500.0}, {X: 0.6, Y: 0.0})[:2] == \
        (pytest.approx({X: 2 / 3, Y: 1 / 3}), "ms2 survivors")


#: an N-bearing co-isolate of the organonitrate ON; ON_CO, the CHO ion, is
#: named by a shared ion list but absent from this sample (inside the window
#: by m/z, in no MS2 spectrum, no MS1 centroid)
ON_N = "C13H17N^NO3+"
ON_KEY = "C4H8N^N+ | C5H9O+"


def _alias_edge_window(t_i, c_i, frag, own=None):
    """The shared fragment sits at the exact C4H8N^N+ m/z at NCE 10 -- read
    there by the two N-bearing ions only -- and 1.2 ppm above C5H9O+ at NCE
    20 and 30, where every named ion reads it, the absent CHO one too (the
    0.44 mDa CO / N^N alias, at the edge of the tolerance). `own`: a fragment
    only the co-isolate can make (C13H15N^NO2+, its -H2O)."""
    lo, hi = _mz("C4H8N^N+"), _mz("C5H9O+") * (1 + 1.2e-6)
    steps = []
    for i, e in enumerate(NCE):
        pk = {ON: t_i[i], ON_N: c_i[i]}
        if frag[i] > 0:
            pk[lo if e == 10 else hi] = frag[i]
        if own is not None and own[i] > 0:
            pk["C13H15N^NO2+"] = own[i]
        steps.append((e, pk))
    t = _synthetic(round(_mz(ON), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(ON) + 0.0001, 0.6), (_mz(ON_N) + 0.0001, 0.4)])
    return t, p


@pytest.mark.parametrize("t_i, c_i, own, frag, call", [
    # the co-isolate falls only after NCE 10 (into its own fragment): read
    # at NCE 10, the target has fallen and the co-isolate is intact -- the
    # target's by the intact test. Waiting for the absent ion moved the onset
    # to NCE 20, where both had fallen: an ambiguous split, the co-isolate's
    # branching flagged and both E50s moved.
    ((1000.0, 900.0, 500.0, 200.0), (300.0, 300.0, 200.0, 100.0), (0.0, 0.0, 100.0, 200.0),
     (0.0, 5.0, 500.0, 800.0), ON),
    # both intact at NCE 10: no test, the survivor split. Waiting for the
    # absent ion moved the onset to NCE 20 and called the target.
    ((1000.0, 990.0, 500.0, 200.0), (300.0,) * 4, None, (0.0, 10.0, 500.0, 800.0),
     "ambiguous"),
])
def test_an_absent_named_ion_does_not_move_the_intact_tests_onset(t_i, c_i, own, frag, call):
    """An ion list shared across windows names a CHO ion this sample lacks.
    It reads the shared fragment only where it sits at C5H9O+, not at its
    NCE 10 onset: `shared_presence` waited for it there, so the intact test
    read its onset at NCE 20 (a 'stray' before it) although every ion the
    sample holds reads the fragment from NCE 10. Set aside before the intact
    test, it is not waited for: the call, basis, weights, F, F_lo, E50s and
    branching flags are those without the name, and its note moves nothing."""
    t, p = _alias_edge_window(t_i, c_i, frag, own)
    two = AN.analyze_run(t, _ions(ON, ON_N))
    three = AN.analyze_run(t, _ions(ON, ON_N, ON_CO))
    assert three.windows[0].co_isolates == [ON_N, ON_CO]
    a = two.fragments.set_index("fragment").loc[ON_KEY]
    b = three.fragments.set_index("fragment").loc[ON_KEY]
    assert (a["candidates"], b["candidates"]) == (f"{ON};{ON_N}", f"{ON};{ON_N};{ON_CO}")
    assert a["call"] == call
    for col in ("call", "basis", "weight_source", "excluded"):
        assert a[col] == b[col], col
    assert _weights(b["weights"]) == pytest.approx({**_weights(a["weights"]), ON_CO: 0.0})
    # set aside, said -- and no onset moved for it
    aside = (f"{ON_CO}: in no MS2 spectrum of the window nor in the MS1 window (absent "
             "from this sample): weight 0, set aside before the intact test")
    assert b["test_note"] == "; ".join(s for s in (aside, a["test_note"]) if s)
    assert "reads its onset at" not in b["test_note"]
    cols = ["ion", "curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(two.e50[cols], three.e50[three.e50["ion"] != ON_CO][cols]
                                  .reset_index(drop=True))
    same = ["ion", "f", "f_lo", "branching_sum", "branching_flag"]
    pd.testing.assert_frame_equal(two.breakdown[same], three.breakdown[
        three.breakdown["ion"] != ON_CO][same].reset_index(drop=True))


def test_own_fixture_an_extra_named_ion_leaves_the_adduct_split():
    """Standard A's adduct window with C11H23+ named too: in no MS2 spectrum,
    a 0.025 % MS1 trace. The shared fragments keep the survivor split of
    C10H20^N+ and C10H19O+ (0.719352 / 0.280648, the trace's share aside) and
    their E50s (8.606 / 6.771); the old rule switched all five fragments to
    the MS1 window shares and moved the E50s by 2.2 and 3.4 NCE. Without the
    trace's MS1 centroid the window reads exactly as without the name."""
    ions = _ions(*MT_IONS, "C11H23+")
    res = AN.analyze_run(_tables("ms2_monoterpene_a"), ions)
    fr = res.fragments[res.fragments["parent_mz"] == 155.1559]
    assert len(fr) == 5 and set(fr["weight_source"]) == {"ms2 survivors + ms1 window"}
    for text in fr["weights"]:
        w = _weights(text)
        assert w["C10H20^N+"] == pytest.approx(0.719352, abs=2e-4)
        assert w["C10H19O+"] == pytest.approx(0.280648, abs=2e-4)
        assert w["C11H23+"] < 3e-4
    assert _e50_row(res, 155.1559, "C10H20^N+")["e50_nce"] == pytest.approx(8.606, abs=5e-3)
    assert _e50_row(res, 155.1559, "C10H19O+")["e50_nce"] == pytest.approx(6.771, abs=5e-3)
    t = _tables("ms2_monoterpene_a")
    far = np.abs(t.window["mz"] - _mz("C11H23+")) < 1e-3      # in two windows' rows
    assert far.any()
    t.window = t.window[~far].reset_index(drop=True)
    gone = AN.analyze_run(t, ions)
    base = AN.analyze_run(_tables("ms2_monoterpene_a"), _ions(*MT_IONS))
    keep = gone.fragments["parent_mz"] == 155.1559
    assert set(gone.fragments.loc[keep, "weight_source"]) == {"ms2 survivors"}
    cols = ["ion", "curve", "e50_nce", "flag"]
    e = gone.e50[(gone.e50["parent_mz"] == 155.1559) & (gone.e50["ion"] != "C11H23+")]
    pd.testing.assert_frame_equal(e[cols].reset_index(drop=True),
                                  base.e50[base.e50["parent_mz"] == 155.1559][cols]
                                  .reset_index(drop=True))


def test_an_ion_beyond_the_ms1_inventory_has_no_share():
    """`ms2 extract` inventoried the MS1 window +-0.205 Da (width 0.4); the
    analysis uses 2.5 Da, so C9H16N+, 0.995 Da above the target, is a
    co-isolate whose MS1 region was never looked at. Its share is unknown,
    not a measured 0: with no MS2 survivor either, C7H11+ is split equally
    (said), never given wholly to the target as 'the MS1 window holds no
    centroid at it'. Inside the inventory an ion with no centroid is still a
    measured 0."""
    T_, far, near = "C10H17+", "C9H16N+", "C8H13N2+"
    steps = [(e, {T_: x, "C7H11+": 50.0 + (1000 - x)}) for e, x in zip(NCE, X_I)]
    t = _synthetic(round(_mz(T_), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(T_) + 0.0001, 1.0)])
    t.manifest = {"parameters": {"isolation_width_used": 0.4, "window_pad_da": 0.005}}
    assert AN.inventory_half_width(t, p) == pytest.approx(0.205)
    res = AN.analyze_run(t, _ions(T_, far), iso_width=2.5)
    (w,) = res.windows
    assert w.co_isolates == [far]
    r = res.fragments.set_index("fragment").loc["C7H11+"]
    assert (r["candidates"], r["weight_source"]) == (f"{T_};{far}", "equal")
    assert _weights(r["weights"]) == {T_: 0.5, far: 0.5}
    assert not any("holds no centroid" in n for n in w.notes)
    assert any(n.startswith(f"{far} is named inside the window, but no MS2 spectrum holds") and
               "beyond the MS1 window ms2 extract inventoried (+-0.205 Da)" in n
               for n in w.notes)
    assert any(n.startswith("the isolation width used here (2.5 Da, explicit) reaches past the "
                            "MS1 window ms2 extract inventoried (+-0.205 Da") for n in res.notes)
    assert AN._ms1_shares(t.window, p, {T_: _mz(T_), far: _mz(far)}, 0.3, 0.205)[far] != \
        AN._ms1_shares(t.window, p, {T_: _mz(T_), far: _mz(far)}, 0.3, 0.205)[far]   # NaN
    # the control: an ion inside the inventory with no centroid is a measured 0
    res = AN.analyze_run(t, _ions(T_, near), iso_width=2.5)
    r = res.fragments.set_index("fragment").loc["C7H11+"]
    assert (r["call"], r["weight_source"], _weights(r["weights"])) == \
        (T_, "composition", {T_: 1.0, near: 0.0})
    assert any("the MS1 window holds no centroid at it" in n for n in res.windows[0].notes)
    # the extract's own width: no run note
    assert not any("reaches past" in n for n in AN.analyze_run(t, _ions(T_, near)).notes)


# ------------------------------------------------- unobserved energies ----
def test_an_unobserved_energy_is_a_gap_for_s_too():
    """A spectrum of the window holds only an unrelated peak (a failed scan):
    X is unobserved there -- no precursor, no fragment. F and F_lo are NaN
    and their E50 a 'gap'; S is NaN too, and its E50 the same 'gap' (a 0
    there read as a crossing at NCE 3 through a hole F calls unobserved);
    and X's branching there is NaN, not a negative sum of everything it
    held."""
    steps = [(1, {X: 1000.0, "C10H17+": 100.0}), (5, {120.5: 30.0}),
             (10, {X: 200.0, "C10H17+": 800.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X))
    (w,) = res.windows
    _, _, g = _curve(res, w.parent_mz, X)
    for col in ("f", "f_lo", "s", "branching_sum"):
        assert math.isnan(g[col].iloc[1]), col
    assert not g["branching_flag"].any()
    assert g["s"].iloc[2] == pytest.approx(0.2)
    for curve in ("F", "F_lo", "S"):
        assert _e50_row(res, w.parent_mz, X, curve)["flag"] == "gap", curve
    b = res.branching[res.branching["nce"] == 5]
    assert len(b) and b[["frag_gain", "ion_loss", "branching"]].isna().all().all()
    assert np.isfinite(res.branching.loc[res.branching["nce"] == 10, "branching"]).all()


# --------------------------------------------- the branching-breach note ----
_MOVED = r"([+-]\d\.\d{4}) at NCE (\d+(?:\.\d+)?)"


def test_the_breach_note_labels_every_value_with_its_energy(nh4):
    """The note names the flagged steps with a finite sum first, then those
    with no ratio; every movement of the unexplained share and every TIC
    ratio carries its own NCE, so none is read against another step's.
    C11H13O+ (161.096): named NCE 21, 41, then 11 -- and -0.0225 is NCE 11's."""
    w = [x for x in nh4.windows if x.parent_mz == 161.096][0]
    (note,) = [n for n in w.notes if n.startswith("branching of C11H13O+")]
    bd = nh4.breakdown[np.isclose(nh4.breakdown["parent_mz"], 161.096)
                       & (nh4.breakdown["ion"] == "C11H13O+")].sort_values("nce")
    u = dict(zip(bd["nce"], bd["budget_unexplained"]))
    tic = dict(zip(bd["nce"], bd["tic"]))
    moved = note[note.index("unexplained share of the TIC moved by"):note.index("the TIC was")]
    got = [(float(v), float(e)) for v, e in __import__("re").findall(_MOVED, moved)]
    assert [e for _, e in got] == [21.0, 41.0, 11.0]
    for v, e in got:
        assert v == pytest.approx(u[e] - u[1.0], abs=5e-5), e
    assert "-0.0225 at NCE 11" in moved
    ratios = note[note.index("the TIC was"):]
    for e in (21, 41, 11):
        assert f"{tic[float(e)] / tic[1.0]:.3f} at NCE {e}" in ratios


def test_the_breach_note_pairs_values_when_a_no_ratio_step_comes_first():
    """Synthetic: the target lost nothing at NCE 10 (no ratio) and something
    at 20 and 30 (sums above the flag), while a named co-isolate loses to a
    channel the spectrum does not hold -- the TIC falls (0.8, 0.65, 0.55)
    and every fraction that stays inflates. The note names 20, 30, then 10,
    pairs each TIC ratio and each unexplained share (flat: nothing
    unexplained) with its own NCE, and names products outside the observed
    m/z range as a cause."""
    steps = [(1, {X: 1000.0, Y: 1000.0}),
             (10, {X: 800.0, "C5H10^N+": 200.0, Y: 600.0}),
             (20, {X: 600.0, "C5H10^N+": 400.0, Y: 300.0}),
             (30, {X: 400.0, "C5H10^N+": 600.0, Y: 100.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    _, _, g = _curve(res, w.parent_mz, X)
    assert list(g["branching_flag"]) == [False, True, True, True]
    assert math.isnan(g["branching_sum"].iloc[1]) and np.isfinite(g["branching_sum"].iloc[2:]).all()
    (note,) = [n for n in w.notes if n.startswith(f"branching of {X}")]
    assert note.startswith(f"branching of {X} sums to ")
    assert "; no ratio at NCE 10 (" in note
    assert "products left the observed m/z range" in note
    moved = note[note.index("unexplained share of the TIC moved by"):note.index("the TIC was")]
    assert [e for _, e in __import__("re").findall(_MOVED, moved)] == ["20", "30", "10"]
    assert "the TIC was 0.650 at NCE 20, 0.550 at NCE 30, 0.800 at NCE 10 of the lowest " \
        "energy's" in note


# ------------------------------------- products outside the observed range ----
def test_a_product_outside_the_observed_range_is_seen_by_s_only():
    """An ion whose only channel releases a product below the scan's
    low-mass limit: the spectrum holds no fragment of it. F (and F_lo) see
    only observed products, so F stays 1 and its E50 reads above range,
    while S -- valid with a steady precursor supply -- falls through 0.5
    (NCE 20). Said in the docs (§11.3, §11.9) and in the run's S note."""
    steps = [(e, {X: x, 120.5: 30.0}) for e, x in zip(NCE, (1000.0, 800.0, 500.0, 200.0))]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X))
    (w,) = res.windows
    _, f, g = _curve(res, w.parent_mz, X)
    assert np.allclose(f, 1.0) and np.allclose(g["f_lo"], 1.0)
    assert np.allclose(g["s"], [1.0, 0.8, 0.5, 0.2])
    assert _e50_row(res, w.parent_mz, X)["flag"] == "above_range"
    assert _e50_row(res, w.parent_mz, X, "S")["e50_nce"] == pytest.approx(20.0)
    assert "outside the observed m/z range" in AN.S_NOTE
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = {k: doc[doc.index(f"### 11.{k} "):doc.index(f"### 11.{k + 1} ")] for k in (3, 5, 9)}
    for k in (3, 5, 9):
        assert "outside the observed m/z range" in " ".join(sec[k].split()), k


def test_the_intact_test_reads_only_observed_products():
    """A known limit, documented (§11.5): X loses 6 % of its signal to a
    product outside the observed range and 1 % to the shared C6H9+, the
    co-isolate Y 5 % to C6H9+. The TIC falls with X's unobserved loss, so
    Y's TIC fraction holds (1.000 of its lowest energy's) though it
    fragments: the intact test excludes Y and C6H9+ is called X's. Nothing
    is flagged: X's products do not outgrow its loss. Only S sees Y fall."""
    steps = [(1, {X: 1000.0, Y: 200.0}), (10, {X: 930.0, Y: 190.0, "C6H9+": 20.0}),
             (20, {X: 800.0, Y: 170.0, "C6H9+": 60.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    r = res.fragments.set_index("fragment").loc["C6H9+"]
    assert (r["call"], r["basis"], r["excluded"]) == (X, "intact test", Y)
    _, fy, gy = _curve(res, w.parent_mz, Y)
    assert np.allclose(fy, 1.0) and np.allclose(gy["s"], [1.0, 0.95, 0.85])
    assert not res.breakdown["branching_flag"].any()
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = " ".join(doc[doc.index("### 11.5 "):doc.index("### 11.6 ")].split())
    assert "the intact test, the guard and the branching see only observed products" in sec


def test_s_rests_on_a_steady_supply_not_on_the_injection_time(mta):
    """Orbitrap intensities are reported per unit injection time: S is not
    said to need a fixed injection time, but a steady precursor supply --
    in the run note and on every S row."""
    assert "steady precursor supply" in AN.S_NOTE
    assert "scale with" not in AN.S_NOTE and "was fixed" not in AN.S_NOTE
    s = mta.e50[mta.e50["curve"] == "S"]
    assert len(s) and s["note"].str.startswith(AN.S_STEADY).all()
    assert not s["note"].str.contains("injection time").any()
    assert "s" in AN.BREAKDOWN_COLUMNS and "s_fixed_it" not in AN.BREAKDOWN_COLUMNS


# ------------------------------------------- resolved fragments stay two ----
def test_a_weak_centroid_never_merges_two_resolved_fragments():
    """m/z 57: C2H4N^N+ (the target's only) and C3H5O+ (either ion's) are
    0.443 mDa apart -- beyond the 0.3 mDa floor -- and resolved in the same
    NCE 20 and 30 spectra (+-0.5 ppm jitter). A 5-count centroid at NCE 1
    midway is within the tolerance of both: single linkage once chained
    them into one fragment 'C2H4N^N+ | C3H5O+', split 0.5 / 0.5 with the
    co-isolate and its branching flagged. Two peaks of one spectrum more
    than the tolerance apart are never one group: C2H4N^N+ stays the
    target's by composition, the co-isolate is not its candidate, and no
    row names both formulas (the centroid reads as C3H5O+ for both ions,
    and joins that fragment). Placed nearer C2H4N^N+ it joins that one, and
    every E50 is what it is without it."""
    n, o = _mz("C2H4N^N+"), _mz("C3H5O+")

    def steps(bridge=None):
        out = []
        for i, (e, x, y) in enumerate(zip(NCE, (1000.0, 1000.0, 600.0, 200.0),
                                          (1000.0, 700.0, 400.0, 100.0))):
            s = (-1) ** i * 0.5e-6
            pk = {ON: x, ON_CO: y}
            if x < 1000:
                pk[n * (1 - s)] = 1000 - x
            if y < 1000:
                pk[o * (1 + s)] = 1000 - y
            if e == 1 and bridge is not None:
                pk[bridge] = 5.0
            out.append((e, pk))
        return out
    p = round(_mz(ON), 4)
    clean = AN.analyze_run(_synthetic(p, steps()), _ions(ON, ON_CO))
    mid = AN.analyze_run(_synthetic(p, steps((n + o) / 2)), _ions(ON, ON_CO))
    fr = mid.fragments.set_index("fragment")
    assert set(fr.index) == {"C2H4N^N+", "C3H5O+"}
    assert (fr.loc["C2H4N^N+", "candidates"], fr.loc["C2H4N^N+", "basis"]) == (ON, "composition")
    assert (fr.loc["C2H4N^N+", "onset_nce"], fr.loc["C2H4N^N+", "n_spectra"]) == (20.0, 2)
    assert fr.loc["C3H5O+", "candidates"] == f"{ON};{ON_CO}"
    near = AN.analyze_run(_synthetic(p, steps(n - 0.00013)), _ions(ON, ON_CO))
    fr = near.fragments.set_index("fragment")
    assert set(fr.index) == {"C2H4N^N+", "C3H5O+"}
    assert (fr.loc["C3H5O+", "call"], fr.loc["C3H5O+", "basis"]) == (ON_CO, "intact test")
    cols = ["ion", "curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(near.e50[cols], clean.e50[cols])
    assert not near.breakdown["branching_flag"].any()


def test_merge_fragment_peaks_never_joins_two_peaks_of_one_spectrum():
    """The cannot-link, alone: A at s2 / s3 and B at s2 / s3 are 0.443 mDa
    apart at m/z 57 (tolerance 0.3 mDa). A peak in s1 between them, within
    the tolerance of both, joins neither when it reads as both formulas
    (its own name), and joins the one it reads as when it reads one only.
    Two peaks of one spectrum within the tolerance are still one fragment."""
    n, o = _mz("C2H4N^N+"), _mz("C3H5O+")
    base = [("s2", "fragment", "C2H4N^N+", n, "A"), ("s3", "fragment", "C2H4N^N+", n, "A"),
            ("s2", "fragment", "C3H5O+", o, "A;B"), ("s3", "fragment", "C3H5O+", o, "A;B")]
    cols = ["spectrum", "role", "key", "mz_cal", "candidates"]
    both = pd.DataFrame(base + [("s1", "fragment", "C2H4N^N+ | C3H5O+", (n + o) / 2, "A;B")],
                        columns=cols)
    out = AT.merge_fragment_peaks(both, tol_ppm=5.0, tol_mda=0.3)
    assert list(out["key"]) == ["C2H4N^N+", "C2H4N^N+", "C3H5O+", "C3H5O+",
                                "C2H4N^N+ | C3H5O+"]
    one = pd.DataFrame(base + [("s1", "fragment", "C3H5O+", (n + o) / 2, "A;B")], columns=cols)
    assert list(AT.merge_fragment_peaks(one, tol_ppm=5.0, tol_mda=0.3)["key"]) == \
        ["C2H4N^N+", "C2H4N^N+", "C3H5O+", "C3H5O+", "C3H5O+"]
    split = pd.DataFrame([("s1", "fragment", "C3H5O+", o, "A;B"),
                          ("s1", "fragment", "C3H5O+", o + 0.0002, "A;B"),
                          ("s2", "fragment", "C3H5O+", o + 0.0001, "A;B")], columns=cols)
    assert set(AT.merge_fragment_peaks(split, tol_ppm=5.0, tol_mda=0.3)["key"]) == {"C3H5O+"}


# ------------------------------------------ the C2F3 = H2^NO4 alias ----
def test_an_organic_reading_outranks_a_carbon_poor_polyoxy_one():
    """In 15N-nitrate data C2F3 = H2^NO4 within 0.21 mDa: CF3COO- (C2F3O2-)
    reads as H2^NO6-, a sub-formula of every [M+^NO3]- cluster. The
    fluorinated co-isolate makes it (in-source and on); the nitrate cluster
    never fragments. Where another window ion reads the peak as a
    carbon-bearing organic ion, an ion that can read it only as a
    carbon-poor polyoxy H/N/O composition is not its candidate, said: the
    fragment is the co-isolate's by composition and the cluster's F stays 1
    (as a candidate it took a survivor share of the in-source fragment)."""
    T_, FC = "C6H12^NO9-", "C9H14F3O4-"
    steps = [(e, {T_: 1000.0, FC: y, "C2F3O2-": 20.0 + (1000 - y)})
             for e, y in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    res = AN.analyze_run(_synthetic(round(_mz(T_), 4), steps), _ions(T_, FC))
    (w,) = res.windows
    assert (w.target, w.co_isolates) == (T_, [FC])
    (r,) = res.fragments.itertuples()
    assert (r.fragment, r.candidates, r.call, r.basis) == \
        ("C2F3O2- | H2^NO6-", FC, FC, "composition")
    assert r.test_note == (f"not a candidate: {T_} reads it only as H2^NO6- -- a carbon-poor "
                           "polyoxy composition, while C2F3O2- (organic) is read there too")
    _, f, _ = _curve(res, w.parent_mz, T_)
    assert np.allclose(f, 1.0) and _e50_row(res, w.parent_mz, T_)["flag"] == "above_range"
    f10, f20 = 700 / 1020, 400 / 1020            # FC / (FC + its whole fragment)
    assert _e50_row(res, w.parent_mz, FC)["e50_nce"] == pytest.approx(
        10 + (f10 - 0.5) / (f10 - f20) * 10)
    for f_, want in (("H2^NO6-", True), ("CH4^NO6-", True), ("HN^NO6-", True),
                     ("^NO3-", False), ("CHO3-", False), ("C2F3O2-", False),
                     ("C3H2F3O2-", False), ("CH5N2O+", False), ("C2H4NO4-", False)):
        assert AT.polyoxy_inorganic(f_) is want, f_


def test_an_absent_organic_reader_takes_nothing_by_the_polyoxy_rule():
    """The polyoxy rule needs an organic reader the sample HOLDS. The
    nitrate cluster declusters into ^NO3- and H2^NO6-; a shared ion list
    also names C9H14F3O4-, inside the window but absent from this sample (no
    MS2 survivor, no MS1 centroid: a measured share of 0), which reads the
    H2^NO6- peak as C2F3O2-. It cannot have made the peak, so the cluster
    stays a candidate and keeps its fragment: calls, weights, F, F_lo, S and
    their E50s are those of the run without it. Dropping the cluster for an
    absent organic reader gave that ion the fragment with weight 1 and moved
    the cluster's F E50 from 16.67 to 21.8, while the window note said the
    absent ion weighs 0."""
    T_, FC = "C6H12^NO9-", "C9H14F3O4-"
    steps = []
    for e, x in zip(NCE, (1000.0, 700.0, 400.0, 100.0)):
        pk = {T_: x}
        if x < 1000:
            pk.update({"^NO3-": 0.5 * (1000 - x), "H2^NO6-": 0.5 * (1000 - x)})
        steps.append((e, pk))
    t = _synthetic(round(_mz(T_), 4), steps)
    p = float(t.parents["parent_mz"].iloc[0])
    t.window = _window_rows(p, [(_mz(T_) + 0.0001, 1.0)])
    one = AN.analyze_run(t, _ions(T_))
    two = AN.analyze_run(t, _ions(T_, FC))
    assert two.windows[0].co_isolates == [FC]
    a = one.fragments.sort_values("fragment", key=lambda k: k.map(len)).reset_index(drop=True)
    b = two.fragments.sort_values("fragment", key=lambda k: k.map(len)).reset_index(drop=True)
    assert list(a["fragment"]) == ["^NO3-", "H2^NO6-"]
    assert list(b["fragment"]) == ["^NO3-", "C2F3O2- | H2^NO6-"]
    assert list(a["call"]) == list(b["call"]) == [T_, T_]
    assert b.loc[1, "candidates"] == f"{T_};{FC}"
    assert _weights(b.loc[1, "weights"]) == {T_: 1.0, FC: 0.0}
    assert f"{FC} reads it as C2F3O2- (organic) but is absent from this sample: {T_}, which " \
        "reads it only as carbon-poor polyoxy, stays a candidate" in b.loc[1, "test_note"]
    same = ["nce", "f", "f_lo", "s", "branching_sum", "branching_flag"]
    pd.testing.assert_frame_equal(one.breakdown[same], two.breakdown[
        two.breakdown["ion"] == T_][same].reset_index(drop=True))
    cols = ["curve", "e50_nce", "flag"]
    pd.testing.assert_frame_equal(one.e50[cols], two.e50[two.e50["ion"] == T_][cols]
                                  .reset_index(drop=True))
    assert _e50_row(two, p, T_)["e50_nce"] == pytest.approx(10 + 2 / 3 * 10)
    assert any(n.startswith(f"{FC} is named inside the window") and "weighs 0" in n
               for n in two.windows[0].notes)
    # the same ion PRESENT (an MS1 centroid, no MS2 survivor) still outranks it
    t.window = _window_rows(p, [(_mz(T_) + 0.0001, 0.9), (_mz(FC), 0.1)])
    r = AN.analyze_run(t, _ions(T_, FC)).fragments.set_index("fragment").loc[
        "C2F3O2- | H2^NO6-"]
    assert (r["candidates"], r["call"], r["basis"]) == (FC, FC, "composition")


def _lobe_steps(lobe_da=-0.0022, share=0.01):
    """The organonitrate target (1000 counts at every energy: it never
    fragments) and its CHO co-isolate falling into C11H13O+, whose line
    carries a weak centroid `lobe_da` beside it holding `share` of it --
    2.2 mDa below, it reads as C7H14^NO3+, a sub-formula of the target only
    (+0.23 ppm, loss C3H7NO2)."""
    lobe = _mz("C11H13O+") + lobe_da
    steps = []
    for e, y in zip(NCE, (1000.0, 700.0, 400.0, 100.0)):
        pk = {ON: 1000.0, ON_CO: y}
        if y < 1000:
            pk["C11H13O+"] = (1 - share) * (1000 - y)
            pk[lobe] = share * (1000 - y)
        steps.append((e, pk))
    return steps


def test_a_side_lobe_of_a_brighter_line_is_not_a_fragment():
    """A weak centroid 2.2 mDa below a line 99x brighter (2.3 line widths at
    a resolving power of 170 000: where an Orbitrap's negative side lobe
    sits) is not an ion. Read against the window ions it is C7H14^NO3+, a
    sub-formula of the target only, and was the target's by composition:
    its F fell below 1 while it never lost a count, and its branching was
    flagged at every energy. Every peak of that fragment sits within 5 widths
    of a peak >= 50x brighter, so it is set aside: its peaks stay in the TIC,
    unexplained, and the window says so. A fragment one of whose peaks
    clears the band or the ratio is kept; a run whose centroids report no
    resolution is not tested."""
    steps = _lobe_steps()
    res = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps, resolution=170000.0),
                         _ions(ON, ON_CO))
    (w,) = res.windows
    assert list(res.fragments["fragment"]) == ["C11H13O+"]
    (note,) = [n for n in w.notes if n.startswith("side lobe: ")]
    assert note.startswith("side lobe: C7H14^NO3+: each of its 3 peaks (3 spectra) sits within 5 "
                           "line widths of a peak >= 50x brighter in the same spectrum (up to "
                           "2.20 mDa = 2.32 widths away, from a peak 99x brighter)")
    _, f, g = _curve(res, w.parent_mz, ON)
    assert np.allclose(f, 1.0) and not g["branching_flag"].any()
    assert (g["n_unexplained"].to_numpy() == [0, 1, 1, 1]).all()
    _, f_co, _ = _curve(res, w.parent_mz, ON_CO)          # CO / (CO + its 99 %)
    y = np.array([1000.0, 700.0, 400.0, 100.0])
    assert np.allclose(f_co, y / (y + 0.99 * (1000 - y)))
    assert not res.breakdown["branching_flag"].any()
    # no resolution reported: no width, no test -- the old reading
    bare = AN.analyze_run(_synthetic(round(_mz(ON), 4), steps), _ions(ON, ON_CO))
    r = bare.fragments.set_index("fragment").loc["C7H14^NO3+"]
    assert (r["call"], r["basis"]) == (ON, "composition")
    assert not any(n.startswith("side lobe") for n in bare.windows[0].notes)
    _, f_bare, g_bare = _curve(bare, w.parent_mz, ON)
    assert f_bare[-1] < 0.992 and g_bare["branching_flag"].iloc[1:].all()
    # beyond 5 widths (a resolving power of 400 000: 5.5 widths), or one
    # peak only 40x weaker than its line: an ion, kept
    louder = [(e, {**pk, **({k: v * 2.5 for k, v in pk.items() if isinstance(k, float)}
                            if e == 30 else {})}) for e, pk in steps]
    for kw, st in (({"resolution": 400000.0}, steps), ({"resolution": 170000.0}, louder)):
        kept = AN.analyze_run(_synthetic(round(_mz(ON), 4), st, **kw), _ions(ON, ON_CO))
        assert "C7H14^NO3+" in set(kept.fragments["fragment"]), kw
    # the helper, by hand: the width is the BRIGHTER peak's m/z / resolution
    roles = pd.DataFrame({"spectrum": ["s"] * 3, "peak": [0, 1, 2],
                          "mz_cal": [100.0, 100.003, 100.5], "intensity": [1000.0, 10.0, 5.0],
                          "role": ["fragment"] * 3, "key": ["A", "B", "C"]})
    assert set(AT.satellite_fragments(roles, {("s", 0): 1e5})) == {"B"}       # 3 widths
    assert AT.satellite_fragments(roles, {("s", 0): 2e5}) == {}               # 6 widths
    assert AT.satellite_fragments(roles, {("s", 1): 1e5}) == {}               # no width for A
    gone = AT.set_aside_fragments(roles, ["B"])
    assert list(gone["role"]) == ["fragment", "unexplained", "fragment"]
    assert list(gone["key"]) == ["A", "", "C"]


def test_our_own_files_hold_no_side_lobe_fragment(nh4, mta):
    """The side-lobe test sets aside no fragment of our own three files: every
    fragment there is an ion (their centroids report a resolution, so the
    test ran)."""
    b = AN.analyze_run(_tables("ms2_monoterpene_b"), _ions(*MT_IONS))
    for res in (nh4, mta, b):
        assert not any(n.startswith("side lobe") for w in res.windows for n in w.notes)
    for stem in ("ms2_nh4_exploratory", "ms2_monoterpene_a", "ms2_monoterpene_b"):
        assert _tables(stem).spectra["resolution"].gt(0).all(), stem


# ------------------------------------------------------ docs and figures ----
def test_the_outputs_section_names_every_column():
    """§11.7's row of each table names every column that is not a key."""
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = doc[doc.index("### 11.7 "):doc.index("### 11.8 ")]
    keys = {"parent_mz", "spectrum", "ion", "fragment", "curve"}
    for name, spec in (("ms2_breakdown.csv", AN.BREAKDOWN_COLUMNS),
                       ("ms2_e50.csv", AN.E50_COLUMNS),
                       ("ms2_fragments.csv", AN.FRAGMENT_COLUMNS),
                       ("ms2_branching.csv", AN.BRANCHING_COLUMNS)):
        (row,) = [ln for ln in sec.splitlines() if ln.startswith(f"| `{name}` |")]
        missing = [c for c in spec if c not in keys and f"`{c}`" not in row]
        assert not missing, (name, missing)


def test_panel_b_breaks_the_targets_line_where_it_is_unobserved(tmp_path):
    """At NCE 20 the spectrum holds no peak the target X can be or make (only
    an oxygen ion X cannot): F is a gap there (0/0), and panel B's dashed
    target line breaks there too instead of dropping to a fraction of 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [(1, {X: 1000.0}), (10, {X: 700.0, "C10H17+": 300.0}),
             (20, {"C2H3O+": 50.0}), (30, {X: 100.0, "C10H17+": 900.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X))
    (w,) = res.windows
    f = w.breakdown.set_index("nce")["f"]
    assert math.isnan(f.loc[20]) and f.loc[10] == pytest.approx(0.7)
    captured = {}
    real_savefig = plt.Figure.savefig

    def grab(fig, path, **kw):
        captured["fig"] = fig
        return real_savefig(fig, path, **kw)
    plt.Figure.savefig = grab
    try:
        assert AN.render_window(w, str(tmp_path / "b.png"))
    finally:
        plt.Figure.savefig = real_savefig
    _, axb = [a for a in captured["fig"].axes if a.get_legend() is not None]
    (own,) = [ln for ln in axb.get_lines() if ln.get_linestyle() == "--"]
    y = dict(zip(own.get_xdata(), own.get_ydata()))
    assert math.isnan(y[20]) and y[10] == pytest.approx(0.7)


def test_panel_b_reads_0_where_the_target_is_gone_not_unobserved(tmp_path):
    """X stays intact at the fragment's onset, so the intact test gives the
    shared C10H17+ to Y; at NCE 30 X's precursor is gone (to a product the
    spectrum does not hold). F is NaN there (0 / 0 weighted), F_lo is 0 (the
    fragment X could make is there): X is observed, gone -- panel B's dashed
    target line reads 0 there, it does not break."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [(1, {X: 500.0, Y: 500.0}), (10, {X: 500.0, Y: 300.0, "C10H17+": 200.0}),
             (20, {X: 480.0, Y: 100.0, "C10H17+": 400.0, "C11H19+": 20.0}),
             (30, {Y: 50.0, "C10H17+": 450.0, "C11H19+": 500.0})]
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), steps), _ions(X, Y))
    (w,) = res.windows
    assert res.fragments.set_index("fragment").loc["C10H17+", "call"] == Y
    xb = w.breakdown[w.breakdown["ion"] == X].set_index("nce")
    assert math.isnan(xb.loc[30, "f"]) and xb.loc[30, "f_lo"] == 0.0
    captured = {}
    real_savefig = plt.Figure.savefig

    def grab(fig, path, **kw):
        captured["fig"] = fig
        return real_savefig(fig, path, **kw)
    plt.Figure.savefig = grab
    try:
        assert AN.render_window(w, str(tmp_path / "b.png"))
    finally:
        plt.Figure.savefig = real_savefig
    _, axb = [a for a in captured["fig"].axes if a.get_legend() is not None]
    (own,) = [ln for ln in axb.get_lines() if ln.get_linestyle() == "--"]
    y = dict(zip(own.get_xdata(), own.get_ydata()))
    assert y[30] == 0.0


def test_panel_b_never_reuses_a_window_ions_colour(tmp_path):
    """One colour, one ion: panel B's fragments take colours no window ion
    wears in panel A (only the target's own dashed line repeats its colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    res = AN.analyze_run(_synthetic(round(_mz(X), 4), _mixture()), _ions(X, Y))
    (w,) = res.windows
    captured = {}
    real_savefig = plt.Figure.savefig

    def grab(fig, path, **kw):
        captured["fig"] = fig
        return real_savefig(fig, path, **kw)
    plt.Figure.savefig = grab
    try:
        assert AN.render_window(w, str(tmp_path / "b.png"))
    finally:
        plt.Figure.savefig = real_savefig
    axa, axb = [a for a in captured["fig"].axes if a.get_legend() is not None]
    a = {to_hex(ln.get_color()) for ln in axa.get_lines() if ln.get_label()[:1] != "_"}
    b = {to_hex(ln.get_color()) for ln in axb.get_lines()
         if ln.get_label()[:1] != "_" and ln.get_linestyle() != "--"}
    assert len(a) == 2 and len(b) >= 3
    assert not a & b, a & b
