"""The merged-ledger isotopologue rows (`batch/iso_checks.satellite_rows`).

A merged row whose line is another merged ion's 13C / 18O / ... isotopologue keeps
its own formula when every per-file arbitration decided by score. The gate reads
the line against the parent's expected ratio spectrum by spectrum, from the peak
AREAS of the batch time series: in band (median 0.4-1.4x, no upper tail) the row
leaves the merged ledger and its pair is an isotope-check veto; a line a real ion
shares (median above the band, or an upper tail) is MIXED -- a note only.

All series here are synthetic: invented readings, exact isotope ratios.

Run: pytest tests/test_isotopologue_rows.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution

ORBI = {"coef": 1.2e-7, "exponent": 1.65}          # R ~ 266 000 at m/z 200: 18O apart from 13C2
TOF = {"coef": 0.02, "exponent": 1.0}              # R = 50 at any m/z: TOF-class
SCALE = {"sigma_ppm": 0.1, "stamp_ppm": 3.0, "merge_ppm": 3.0}
P = ("C8H12O7", "[M+NO3]-")                         # the parent: an invented nitrate cluster
R = ("C14H10O5", "[M-H]-")                          # an unrelated reading put on its line
X = ("C6H10O5", "[M-H]-")                           # a bystander far from both


def quiet(*a):
    pass


def _line(parent, label):
    """(m/z, expected ratio) of the parent's line whose main component is `label`."""
    pmz = C.ion_mz(*parent)
    cnt = IC.ion_counts(*parent, None)
    for sh, e, tags in IC._parent_lines(cnt, Resolution.from_dict(ORBI).fwhm(pmz + 1.0)):
        if IC._sat_label(tags) == label:
            return pmz + sh, e
    raise AssertionError(f"no {label} line")


def _merged(rows):
    """rows: (mz, (neutral, adduct), tier[, tier_reason])."""
    out = []
    for r in rows:
        mz, (n, a), tier = r[:3]
        out.append(dict(mz=float(mz), neutral_formula=n, adduct=a, tier=tier, ion_score=0.95,
                        tier_reason=r[3] if len(r) > 3 else pd.NA, alternatives="", ion_only_of=pd.NA))
    return pd.DataFrame(out)


def _series(lines, n=9, area=True):
    """lines: [(mz, [area per spectrum] or a scalar)]; heights 0.75x the areas; a
    row per (spectrum, line) where the area is finite and > 0, plus a noise floor."""
    rows = []
    for i in range(n):
        for mz, a in lines:
            v = float(a[i] if np.ndim(a) else a)
            if np.isfinite(v) and v > 0:
                rows.append(dict(sample_item_id=f"s{i:02d}", mz=float(mz), height=0.75 * v, area=v))
        rows += [dict(sample_item_id=f"s{i:02d}", mz=500.0 + 0.41 * j, height=5.0, area=6.0) for j in range(20)]
    t = pd.DataFrame(rows)
    return t if area else t.drop(columns="area")


def _wave(n=9):
    return 1e5 * (1.5 + np.sin(np.arange(n) / 2.0))


def _case(rho, label="13C", n=9, r_reading=R, extra=None):
    """A merged parent, the reading on its `label` line at rho x expected per spectrum,
    and a bystander; the series to match."""
    pmz = C.ion_mz(*P)
    lmz, e = _line(P, label)
    ap = _wave(n)
    rho = np.broadcast_to(np.asarray(rho, float), (n,))
    merged = _merged([(pmz, P, "Assigned"), (lmz, r_reading, "Assigned"), (C.ion_mz(*X), X, "Candidate")]
                     + (extra or []))
    ts = _series([(pmz, ap), (lmz, rho * e * ap), (C.ion_mz(*X), 3e4)], n=n)
    return merged, ts, lmz, e


def _gate(merged, ts, **kw):
    kw = {"resolution": ORBI, "mass_scale": SCALE, "klass": "orbitrap", "log": quiet, **kw}
    return IC.satellite_rows(merged, ts, **kw)


def _keys(df):
    return set(zip(df["neutral_formula"], df["adduct"]))


# --------------------------------------------------------------------------- the strip
def test_true_13c_satellite_is_stripped_and_vetoed():
    rho = 1.0 + 0.1 * np.sin(np.arange(9))                # 0.9-1.1 in every spectrum
    merged, ts, lmz, e = _case(rho)
    kept, tab, summ = _gate(merged, ts)
    assert _keys(kept) == {P, X}, kept
    assert summ["ran"] and summ["n_stripped"] == 1 and summ["n_stripped_assigned"] == 1, summ
    row = tab.iloc[0]
    assert row["verdict"] == "isotopologue" and row["action"] == "stripped" and row["label"] == "13C"
    assert row["parent_ion"] == "C8H12NO10-" and row["parent_tier"] == "Assigned"
    assert row["expected"] == pytest.approx(e) and 0.9 <= row["rho_area"] <= 1.1 and row["share"] == pytest.approx(
        min(1.0, 1.0 / row["rho_area"]))
    assert row["rho_height"] == pytest.approx(row["rho_area"]) and row["n_both"] == 9 and row["n_tail"] == 0
    # the parent's row says where its line went
    note = str(kept.loc[kept["neutral_formula"] == P[0], "tier_reason"].iloc[0])
    assert f"the line at m/z {lmz:.4f} (was {R[0]} {R[1]}) is its 13C isotopologue" in note, note
    # ... and the pooled pair of the stripped reading is an isotope-check veto
    v = IC.veto_rows(tab)
    assert list(v.columns) == list(IC.TABLE_COLUMNS) and len(v) == 1 and bool(v["veto"].iloc[0])
    vv = IC.veto(v)
    assert set(vv) == {R} and vv[R].startswith("isotopologue: the line is the 13C isotopologue of C8H12NO10-")
    assert IC.summary(v, ORBI)["SAT"] == {"vetoed": 1} and IC.summary(v, ORBI)["tested"] == 0


def test_18o_line_is_resolved_from_13c2_and_stripped():
    lmz, e = _line(P, "18O")
    l2, e2 = _line(P, "13C2")
    assert abs(l2 - lmz) > 2e-3                           # 2.5 mDa apart: two lines at this width
    assert e == pytest.approx(10 * 0.002050 / 0.997570, rel=0.05)   # 18O alone, no 13C2 in it
    merged, ts, _lmz, _e = _case(np.full(9, 0.95), label="18O")
    kept, tab, summ = _gate(merged, ts)
    assert summ["n_stripped"] == 1 and tab.iloc[0]["label"] == "18O" and R not in _keys(kept)


def test_height_bias_does_not_decide():
    """The area decides: areas at 1x expected strip whatever the heights read; areas at
    1.6x do not, though heights at 1.2x would sit in the band."""
    merged, ts, lmz, e = _case(np.full(9, 1.0))
    sel = np.isclose(ts["mz"], lmz)
    ts.loc[sel, "height"] = ts.loc[sel, "area"] * 0.7 * 0.75
    assert _gate(merged, ts)[2]["n_stripped"] == 1
    merged, ts, lmz, e = _case(np.full(9, 1.6))
    sel = np.isclose(ts["mz"], lmz)
    ts.loc[sel, "height"] = ts.loc[sel, "area"] * 0.75 * 0.75
    kept, tab, summ = _gate(merged, ts)
    assert summ["n_stripped"] == 0 and R in _keys(kept) and tab.iloc[0]["verdict"] == "mixed"


# --------------------------------------------------------------------------- kept
def test_intermittent_real_ion_on_a_13c_line_is_mixed_not_stripped():
    """A real ion present in 4 of 10 spectra at 2-3x: the median reads 1.1 (in band),
    the upper tail says the line is shared -- a note, no strip."""
    rho = np.array([1.0, 1.05, 1.1, 1.0, 1.1, 1.15, 2.2, 2.6, 3.0, 2.4])
    merged, ts, lmz, e = _case(rho, n=10)
    assert 1.0 <= float(np.median(rho)) <= 1.4
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and summ["n_mixed"] == 1, summ
    row = tab.iloc[0]
    assert row["verdict"] == "mixed" and row["action"] == "note" and row["n_tail"] == 4
    note = str(kept.loc[kept["neutral_formula"] == R[0], "tier_reason"].iloc[0])
    # the median share, and the spectra that hold more -- never "100% of this line"
    assert "% of this line is the" not in note.split("in the median spectrum")[0], note
    share = f"{min(1.0, 1.0 / float(np.median(rho))):.0%}"
    assert f"in the median spectrum {share} of this line is the 13C isotopologue of C8H12NO10-" in note, note
    assert "but 4 of 10 spectra hold more than 2x the expected line: a real ion shares it" in note, note
    assert kept.loc[kept["neutral_formula"] == R[0], "tier"].iloc[0] == "Assigned"
    assert len(IC.veto_rows(tab)) == 0


def test_mixed_line_above_the_band_is_a_note_with_its_share():
    merged, ts, lmz, e = _case(np.full(9, 2.5))
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_mixed"] == 1
    assert tab.iloc[0]["share"] == pytest.approx(0.4)
    assert "40% of this line is the 13C isotopologue" in str(
        kept.loc[kept["neutral_formula"] == R[0], "tier_reason"].iloc[0])


@pytest.mark.parametrize("why", ["few", "position", "low", "high"])
def test_no_action(why):
    if why == "few":            # line and parent together in 2 spectra only
        merged, ts, lmz, e = _case(np.r_[np.ones(2), np.full(7, np.nan)])
    elif why == "position":     # 2 ppm off the line, the window max(1 ppm, 4 x 0.1 ppm)
        merged, ts, lmz, e = _case(np.ones(9))
        shift = lmz * 2e-6
        merged.loc[np.isclose(merged["mz"], lmz), "mz"] += shift
        ts.loc[np.isclose(ts["mz"], lmz), "mz"] += shift
    elif why == "low":
        merged, ts, lmz, e = _case(np.full(9, 0.2))
    else:
        merged, ts, lmz, e = _case(np.full(9, 6.0))
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and len(tab) == 0, (why, summ, tab)


# --------------------------------------------------------------------------- exempt
def test_known_species_and_labelled_standard_are_exempt():
    merged, ts, lmz, e = _case(np.ones(9))
    i = merged.index[np.isclose(merged["mz"], lmz)][0]
    merged.at[i, "tier_reason"] = IC.KNOWN_LOCK_MARK + " (2 files)"
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_exempt"] == 1 and tab.iloc[0]["verdict"] == "exempt"
    assert "kept: a known-species decision" in str(kept.loc[kept["neutral_formula"] == R[0], "tier_reason"].iloc[0])
    lab = ("C5H8^NO4", "[M-H]-")                          # an isotope-labelled reading on the line
    merged, ts, lmz, e = _case(np.ones(9), r_reading=lab)
    kept, tab, summ = _gate(merged, ts)
    assert lab in _keys(kept) and summ["n_exempt"] == 1 and "isotope-labelled" in tab.iloc[0]["note"]
    assert len(IC.veto_rows(tab)) == 0


def test_a_curated_reading_is_exempt_as_in_the_element_signature_removal():
    """One exempt policy for both merged-ledger removal gates: a reading a curated list
    stands behind (`exempt`, passes.curated_formulas -- what remove_signature_vetoed
    spares) is a note-only 'exempt' here too, with no SAT veto."""
    merged, ts, lmz, e = _case(np.ones(9))
    kept, tab, summ = _gate(merged, ts, exempt=frozenset({R[0]}))
    assert R in _keys(kept) and summ["n_exempt"] == 1 and summ["n_stripped"] == 0
    assert tab.iloc[0]["verdict"] == "exempt" and "kept: a curated formula" in tab.iloc[0]["note"]
    assert len(IC.veto_rows(tab)) == 0
    # a curated PARENT still strips the reading on its line: the policy spares a reading, not its lines
    kept, tab, summ = _gate(merged, ts, exempt=frozenset({P[0]}))
    assert R not in _keys(kept) and summ["n_stripped"] == 1


def test_a_stripped_row_whose_parent_is_removed_reads_parent_removed():
    """The element-signature removal runs after the gate: a row stripped as the
    satellite of a reading that then leaves the merged ledger reads 'parent removed',
    and its pooled pair no longer carries the SAT veto."""
    merged, ts, lmz, e = _case(np.ones(9))
    kept, tab, summ = _gate(merged, ts)
    assert len(IC.veto_rows(tab)) == 1
    same, n = IC.parent_removed(tab, [{"neutral_formula": X[0], "adduct": X[1]}], log=quiet)
    assert n == 0 and same is tab
    out, n = IC.parent_removed(tab, [{"neutral_formula": P[0], "adduct": P[1]}], log=quiet)
    assert n == 1 and out.iloc[0]["verdict"] == IC.SAT_PARENT_REMOVED
    assert "then left the merged ledger" in out.iloc[0]["note"] and out.iloc[0]["action"].startswith("stripped")
    assert len(IC.veto_rows(out)) == 0 and tab.iloc[0]["verdict"] == "isotopologue"   # the input is untouched


def test_ion_only_row_is_neither_parent_nor_stripped():
    merged, ts, lmz, e = _case(np.ones(9))
    merged.loc[np.isclose(merged["mz"], lmz), "ion_only_of"] = "C9H9O4"
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and tab.iloc[0]["verdict"] == "exempt"
    merged, ts, lmz, e = _case(np.ones(9))
    merged.loc[merged["neutral_formula"] == P[0], "ion_only_of"] = "C8H12O7"     # the parent is ion-only
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and len(tab) == 0


# --------------------------------------------------------------------------- scope
def test_scope_skips():
    merged, ts, lmz, e = _case(np.ones(9))
    for kw, why in (({"klass": "tof"}, "instrument class tof"),
                    ({"klass": "auto", "resolution": TOF}, "instrument class tof"),
                    ({"klass": None}, "instrument class unknown"),
                    ({"resolution": None}, "no width model")):
        kept, tab, summ = _gate(merged, ts, **kw)
        assert not summ["ran"] and summ["skipped"].startswith(why) and len(kept) == len(merged), (kw, summ)
    kept, tab, summ = _gate(merged, None)
    assert summ["skipped"] == "no time series"
    # a height-only series: skipped, never read on heights
    for t in (ts.drop(columns="area"), ts.assign(area=np.nan)):
        kept, tab, summ = _gate(merged, t)
        assert not summ["ran"] and "no peak areas" in summ["skipped"] and len(kept) == len(merged)
        assert list(tab.columns) == list(IC.SAT_TABLE_COLUMNS) and not len(tab)
    assert _gate(merged, ts, klass="auto")[2]["n_stripped"] == 1           # 'auto' reads the width model


# --------------------------------------------------------------------------- chains, order, re-entry
def test_chain_a_stripped_row_is_never_a_parent():
    """P0 -> P1 (P0's 13C line) -> R (P1's 13C line = P0's 13C2 line): P1 leaves, and R
    is judged against P0's 13C2 line only."""
    p0 = P
    l1, e1 = _line(p0, "13C")
    l2, e2 = _line(p0, "13C2")
    ap = _wave()
    p1 = ("C15H8O4", "[M-H]-")
    merged = _merged([(C.ion_mz(*p0), p0, "Assigned"), (l1, p1, "Assigned"), (l2, R, "Candidate")])
    ts = _series([(C.ion_mz(*p0), ap), (l1, e1 * ap), (l2, e2 * ap)])
    kept, tab, summ = _gate(merged, ts)
    assert _keys(kept) == {p0} and summ["n_stripped"] == 2, tab
    r = tab[tab["neutral_formula"] == R[0]].iloc[0]
    assert r["parent_ion"] == "C8H12NO10-" and r["label"] == "13C2" and r["parents"].count(";") == 0


def test_deterministic_and_restrips_a_row_that_re_enters():
    merged, ts, lmz, e = _case(1.0 + 0.1 * np.sin(np.arange(9)))
    kept, tab, summ = _gate(merged, ts)
    # shuffled rows (ledger and series) give the same table and the same kept set
    k2, t2, s2 = _gate(merged.sample(frac=1.0, random_state=3), ts.sample(frac=1.0, random_state=5))
    pd.testing.assert_frame_equal(tab, t2)
    assert _keys(k2) == _keys(kept)
    # the second merge (cover + residual) re-adds the reading as an M0 row: stripped again
    back = pd.concat([kept, merged[merged["neutral_formula"] == R[0]]], ignore_index=True)
    back["tier_reason"] = pd.NA
    k3, t3, s3 = _gate(back, ts)
    assert _keys(k3) == _keys(kept) and s3["n_stripped"] == 1
    pd.testing.assert_frame_equal(t3, tab)
    # the merged schema is unchanged by the strip
    assert list(kept.columns) == list(merged.columns)


def test_reagent_parent():
    """A merged reading on a reagent ion's 18O line, at the line's ratio, is the reagent's."""
    rg_mz = C.ion_mz("HNO3", "[M-H]-")                    # NO3-
    lines = IC._parent_lines({"N": 1, "O": 3}, Resolution.from_dict(ORBI).fwhm(rg_mz + 1))
    sh, e, tags = next(x for x in lines if IC._sat_label(x[2]) == "18O")
    ap = 1e7 * (1.5 + np.sin(np.arange(9) / 2.0))
    on = ("CH3O3", "[M-H]-")                               # an invented reading
    merged = _merged([(rg_mz + sh, on, "Candidate"), (C.ion_mz(*X), X, "Assigned")])
    ts = _series([(rg_mz, ap), (rg_mz + sh, e * ap), (C.ion_mz(*X), 3e4)])
    # the reagent's M0 and its own 18O line, both under the reagent's ion formula: the
    # heavy line is no parent reading (a median over both would sit ~1 Da off the M0)
    reag = pd.DataFrame({"mz": [rg_mz, rg_mz + sh], "role": "reagent",
                         "ion_formula": ["NO3-", "NO3-"], "iso_label": [None, "18O"]})
    kept, tab, summ = _gate(merged, ts, reagents=reag)
    assert on not in _keys(kept) and summ["n_stripped"] == 1, (summ, tab)
    row = tab.iloc[0]
    assert row["parent_tier"] == "reagent" and row["parent_ion"] == "NO3-" and row["label"] == "18O"
    assert row["parent_mz"] == pytest.approx(rg_mz, abs=1e-6)
    assert summ["n_parents"] == 3                          # two merged rows + one reagent ion (its heavy line no parent)
    # without the reagent rows there is no parent
    assert _gate(merged, ts)[2]["n_stripped"] == 0


def test_role_votes_are_recorded():
    merged, ts, lmz, e = _case(np.ones(9))
    pmz = C.ion_mz(*P)
    a = pd.DataFrame({"mz": [pmz, lmz], "role": ["M0", "M0"], "peak_id": ["a1", "a2"], "parent_peak_id": [None, None]})
    b = pd.DataFrame({"mz": [pmz, lmz], "role": ["M0", "iso_child"], "peak_id": ["b1", "b2"],
                      "parent_peak_id": [None, "b1"]})
    kept, tab, summ = _gate(merged, ts, per_file={"f1": a, "f2": b, "f3": b})
    row = tab.iloc[0]
    assert (row["files_m0"], row["files_iso_parent"], row["files_other"]) == (1, 2, 0)


# --------------------------------------------------------------------------- boundaries
@pytest.mark.parametrize("n, n_high, verdict", [(9, 0, "isotopologue"), (9, 1, "mixed"),
                                                 (20, 1, "isotopologue"), (20, 2, "mixed")])
def test_tail_boundary(n, n_high, verdict):
    """Fewer than max(1, 10 % of n) spectra above 2x: one high spectrum of 9 is
    already a shared line, one of 20 is not (two of 20 are)."""
    rho = np.ones(n)
    rho[:n_high] = 2.5
    merged, ts, lmz, e = _case(rho, n=n)
    kept, tab, summ = _gate(merged, ts)
    row = tab.iloc[0]
    assert row["verdict"] == verdict and row["n_tail"] == n_high and row["n_both"] == n, row.to_dict()
    assert (R in _keys(kept)) == (verdict == "mixed")
    if verdict == "mixed":
        note = str(kept.loc[kept["neutral_formula"] == R[0], "tier_reason"].iloc[0])
        assert not note.startswith("100%") and f"but {n_high} of {n} spectra hold more than 2x" in note, note


@pytest.mark.parametrize("rho, stripped", [(1.3, True), (1.38, True), (1.45, False)])
def test_upper_band_edge(rho, stripped):
    """A median at 1.3x expected is in band (stripped); above 1.4x it is MIXED."""
    merged, ts, lmz, e = _case(np.full(9, rho))
    kept, tab, summ = _gate(merged, ts)
    assert tab.iloc[0]["rho_area"] == pytest.approx(rho, rel=1e-6)
    assert (R not in _keys(kept)) == stripped and tab.iloc[0]["verdict"] == ("isotopologue" if stripped else "mixed")


# --------------------------------------------------------------------------- secondary paths
def test_two_parents_summed_on_one_line():
    """P's 13C line and P2's 37Cl2 line are one line (0.2 ppm apart): the expectation
    is the sum e1 A_P + e2 A_P2, spectrum by spectrum -- either parent alone reads
    the line at ~2x and would leave it MIXED."""
    p2 = ("C12H18O3Cl2", "[M-H]-")
    l1, e1 = _line(P, "13C")
    l2, e2 = _line(p2, "37Cl2")
    assert abs(l2 - l1) / l1 * 1e6 < 0.5
    pmz, p2mz = C.ion_mz(*P), C.ion_mz(*p2)
    ap = _wave()
    ap2 = (e1 / e2) * 1e5 * (1.5 + np.cos(np.arange(9) / 2.0))     # e2 A_P2 ~ e1 A_P
    merged = _merged([(pmz, P, "Assigned"), (p2mz, p2, "Assigned"), (l1, R, "Assigned"),
                      (C.ion_mz(*X), X, "Candidate")])
    ts = _series([(pmz, ap), (p2mz, ap2), (l1, e1 * ap + e2 * ap2), (C.ion_mz(*X), 3e4)])
    kept, tab, summ = _gate(merged, ts)
    assert _keys(kept) == {P, p2, X} and summ["n_stripped"] == 1, tab
    row = tab.iloc[0]
    assert row["rho_area"] == pytest.approx(1.0, rel=1e-3) and row["n_tail"] == 0
    parents = row["parents"].split("; ")
    assert len(parents) == 2 and {x.split()[0] for x in parents} == {"C8H12NO10-", "C12H17Cl2O3-"}, parents
    # ... and each parent's tier_reason is left to the main contributor only
    assert sum("is its" in str(x) for x in kept["tier_reason"]) == 1


def test_two_parents_on_one_peak_are_read_once():
    """A reagent ion and a merged row on the same peak: the peak's area enters the
    expectation once, not twice (twice would halve rho and strip a mixed line)."""
    rg_mz = C.ion_mz("HNO3", "[M-H]-")
    lines = IC._parent_lines({"N": 1, "O": 3}, Resolution.from_dict(ORBI).fwhm(rg_mz + 1))
    sh, e, tags = next(x for x in lines if IC._sat_label(x[2]) == "18O")
    ap = 1e7 * (1.5 + np.sin(np.arange(9) / 2.0))
    on = ("CH3O3", "[M-H]-")
    merged = _merged([(rg_mz, ("HNO3", "[M-H]-"), "Assigned"), (rg_mz + sh, on, "Candidate"),
                      (C.ion_mz(*X), X, "Assigned")])
    ts = _series([(rg_mz, ap), (rg_mz + sh, 2.5 * e * ap), (C.ion_mz(*X), 3e4)])
    reag = pd.DataFrame({"mz": [rg_mz], "role": "reagent", "ion_formula": ["NO3-"], "iso_label": [None]})
    kept, tab, summ = _gate(merged, ts, reagents=reag)
    row = tab.iloc[0]
    assert row["parents"].count(";") == 1 and row["rho_area"] == pytest.approx(2.5, rel=1e-3), row.to_dict()
    assert row["verdict"] == "mixed" and on in _keys(kept)


def test_position_is_the_trace_centroid():
    """The merged m/z 3 ppm off the line, the re-centred trace (mz_trace) on it: the
    row is read at its trace and stripped; without the trace it is out of reach."""
    merged, ts, lmz, e = _case(np.ones(9))
    i = merged.index[np.isclose(merged["mz"], lmz)][0]
    merged["mz_trace"] = merged["mz"]
    merged.at[i, "mz"] = lmz * (1 + 3e-6)
    kept, tab, summ = _gate(merged, ts)
    assert R not in _keys(kept) and summ["n_stripped"] == 1
    assert tab.iloc[0]["mz"] == pytest.approx(lmz) and abs(tab.iloc[0]["offset_ppm"]) < 0.01
    kept, tab, summ = _gate(merged.drop(columns="mz_trace"), ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and len(tab) == 0


def test_labelled_pair_channel_on_a_15n_line_is_exempt():
    """A profile clustering on both the 14N and the 15N-labelled nitrate: a 14N-cluster
    reading on a parent's 15N line is the labelled pair's business (exempt); on the
    13C line, or without the labelled channel, it is judged as any other."""
    from types import SimpleNamespace
    sib = ("C9H10O4", "[M+NO3]-")
    pair = SimpleNamespace(adducts=("[M-H]-", "[M+NO3]-", "[M+^NO3]-"))
    plain = SimpleNamespace(adducts=("[M-H]-", "[M+NO3]-"))
    merged, ts, lmz, e = _case(np.ones(9), label="15N", r_reading=sib)
    kept, tab, summ = _gate(merged, ts, prof=pair)
    assert sib in _keys(kept) and summ["n_exempt"] == 1 and tab.iloc[0]["label"] == "15N"
    assert "kept: a reading on the labelled pair's channel" in tab.iloc[0]["note"]
    for prof in (plain, None):
        kept, tab, summ = _gate(merged, ts, prof=prof)
        assert sib not in _keys(kept) and summ["n_stripped"] == 1, prof
    merged, ts, lmz, e = _case(np.ones(9), label="13C", r_reading=sib)
    kept, tab, summ = _gate(merged, ts, prof=pair)
    assert sib not in _keys(kept) and summ["n_stripped"] == 1


@pytest.mark.parametrize("adduct, labelled", [
    ("[M+15NO3]-", True), ("[M+13CO3]-", True), ("[M+D]+", True), ("[M+H+D2O]+", True), ("[M-D]-", True),
    ("[M+^NO3]-", True), ("[M+NO3]-", False), ("[M+Cu]+", False), ("[M+2H]2+", False), ("[M+Dy]+", False),
    ("[M+13Cl]-", False), ("[M+H]+", False), ("[M-H]-", False), ("", False)])
def test_labelled_adduct_pattern(adduct, labelled):
    assert IC._is_labelled("C5H8O4", adduct) is labelled


def test_collapsed_and_untiered_rows_are_neither_parent_nor_candidate():
    merged, ts, lmz, e = _case(np.ones(9))
    merged["trace_role"] = ""
    merged.loc[merged["neutral_formula"] == P[0], "trace_role"] = "collapsed"     # a collapsed trace label
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_stripped"] == 0 and len(tab) == 0 and summ["n_parents"] == 2
    merged["trace_role"] = ""
    merged.loc[np.isclose(merged["mz"], lmz), "trace_role"] = "collapsed"
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_matched"] == 0
    merged, ts, lmz, e = _case(np.ones(9))
    merged.loc[np.isclose(merged["mz"], lmz), "tier"] = "Unassigned"           # no Assigned / Candidate tier
    kept, tab, summ = _gate(merged, ts)
    assert R in _keys(kept) and summ["n_matched"] == 0 and len(tab) == 0


# --------------------------------------------------------------------------- through the batch
def test_batch_strips_stamps_and_levels(tmp_path, monkeypatch):
    """Through assign_batch.run: the reading on the parent's 13C line leaves the merged
    ledger, the line is stamped as the parent's isotopologue, its pooled pair is an
    isotope-check veto (5b) and the table is written; --no-isotopologue-gate keeps it."""
    import json
    import os

    from peaky.assignment import assign as A_
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    pmz = C.ion_mz(*P)
    lmz, e = _line(P, "13C")
    xmz = C.ion_mz(*X)
    xl, ex = _line(X, "13C")                              # the bystander's own 13C line (rule C's bias)
    t0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
    rows = []
    for i in range(14):
        w = 1e5 * (1.5 + np.sin(i / 2.0))
        lines = [(pmz, w), (lmz, e * w * (1.0 + 0.05 * np.cos(i))), (xmz, 4e4), (xl, ex * 4e4)] + \
                [(60.0 + 3.7 * j, 50.0) for j in range(80)]
        rows += [dict(sample_item_id=f"s{i:02d}", sample_item_name=f"n{i:02d}",
                      datetime_utc=t0 + pd.Timedelta(minutes=10 * i), peak_id=f"s{i:02d}_{k}", mz=float(m),
                      height=0.75 * float(h), area=float(h)) for k, (m, h) in enumerate(lines)]
    pk = pd.DataFrame(rows)

    def _ionf(n, a):
        return C.format_formula(T._ion_counts(n, a)) + "-"

    def fake_assign(sid, context="ambient-air", **kw):
        led = L.new_ledger(pd.DataFrame([("p1", pmz, 1e5), ("p2", lmz, 1e4), ("p3", xmz, 3e4)],
                                        columns=["peak_id", "mz", "height"]))
        for pid, (n, a) in (("p1", P), ("p2", R), ("p3", X)):
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=_ionf(n, a), ion_score=0.9,
                                compound_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                                confidence="High", commentary="stub")
        T.apply_tiers(led)
        led.loc[led["role"] == L.ROLE_M0, "tier"] = T.TIER_ASSIGNED
        return {"ledger": led, "stats": {"noise_edge_cps": 50.0, "height_gate_cps": 50.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pk[pk["sample_item_id"] == sid]
                        [["peak_id", "mz", "height"]].reset_index(drop=True))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A_, "run", fake_assign)

    def run(d, **kw):
        AB.run(peaks=pk, ts_peaks=pk, reagent="NO3", batch="test batch", out_dir=str(d), k_min=2, k_max=3,
               min_gain=0.0, n_jobs=1, resolving_power=Resolution.from_dict(ORBI), mass_axis="off",
               residual=False, log=quiet, **kw)
        summ = json.load(open(os.path.join(d, "batch_summary.json")))
        merged = pd.read_csv(os.path.join(d, "merged_ledger.csv"))
        tab = pd.read_csv(os.path.join(d, "tables", "isotopologue_rows.csv"))
        lev = pd.read_csv(os.path.join(d, "tables", "evidence_levels.csv"))
        ts = pd.read_parquet(os.path.join(d, "per_file", "_batch_ts.parquet"))
        return summ, merged, tab, lev, ts

    summ, merged, tab, lev, ts = run(tmp_path / "a")
    g = summ["merge_gates"]["isotopologue"]
    assert g["ran"] and g["n_stripped"] == 1 and g["instrument"] == "orbitrap", g
    assert R not in _keys(merged) and {P, X} <= _keys(merged)
    assert len(tab) == 1 and tab["verdict"].iloc[0] == "isotopologue" and tab["label"].iloc[0] == "13C"
    # the line is the parent's isotopologue in the stamp, not an analyte
    on = ts[np.isclose(ts["mz"], lmz)]
    assert len(on) == 14 and (on["neutral_formula"].isna() | (on["neutral_formula"].astype(str) == "")).all()
    assert (on["ion_formula"] == _ionf(*P)).all() and (on["role"] == "iso_child").all(), on[["role", "ion_formula"]]
    # the pooled pair reads the veto, as the merged ledger does
    # (this tiny batch calibrates no degeneracy window, so no level is assessed: the
    # pair's hard fact is what the scale's step 0 reads -- 5b wherever levels run)
    r = lev[(lev["neutral_formula"] == R[0]) & (lev["adduct"] == R[1])].iloc[0]
    assert bool(r["iso_veto"]) and str(r["iso_note"]).startswith("isotopologue: the line is the 13C"), r.to_dict()
    # the parent's 13C slot is free now: rule C reads its carbons and agrees
    assert not lev.loc[(lev["neutral_formula"] == P[0]) & (lev["adduct"] == P[1]), "iso_veto"].astype(bool).any()
    iso = pd.read_csv(os.path.join(tmp_path / "a", "tables", "iso_checks.csv"))
    sat = iso[iso["check"] == "SAT"]
    assert len(sat) == 1 and sat["neutral_formula"].iloc[0] == R[0] and bool(sat["veto"].iloc[0])
    rc = iso[(iso["check"] == "C") & (iso["neutral_formula"] == P[0])].iloc[0]
    assert rc["verdict"] == "agree", rc.to_dict()
    assert summ["evidence_levels"]["iso_checks"]["SAT"] == {"vetoed": 1}
    # off: the row stays and the table is a header
    summ, merged, tab, lev, ts = run(tmp_path / "b", isotopologue_rows=False)
    assert R in _keys(merged) and not len(tab)
    assert summ["merge_gates"]["isotopologue"]["skipped"] == "--no-isotopologue-gate"


def test_batch_parent_removed_by_the_element_signature_gate(tmp_path, monkeypatch):
    """Both merged-ledger gates in one batch: R sits on the 13C line of a bromine reading
    PB whose own 81Br line the batch never shows. The isotopologue gate strips R as PB's
    satellite at the merge; REQ then refutes PB on its 81Br line and the element-signature
    removal takes PB out. The table reconciles: R reads 'parent removed', its pooled pair
    carries no SAT veto (iso_checks.csv, evidence_levels.csv), and the gates' summary
    counts it. Both gates were handed the same curated exempt set."""
    import json
    import os

    from peaky.assignment import assign as A_
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    PB = ("C8H13BrO4", "[M-H]-")                         # an invented bromine reading
    pmz = C.ion_mz(*PB)
    lmz, e = _line(PB, "13C")
    t0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
    rows = []
    for i in range(14):
        w = 1e5 * (1.5 + np.sin(i / 2.0))
        lines = [(pmz, w), (lmz, e * w * (1.0 + 0.05 * np.cos(i)))] + [(60.0 + 3.7 * j, 50.0) for j in range(80)]
        rows += [dict(sample_item_id=f"s{i:02d}", sample_item_name=f"n{i:02d}",
                      datetime_utc=t0 + pd.Timedelta(minutes=10 * i), peak_id=f"s{i:02d}_{k}", mz=float(m),
                      height=0.75 * float(h), area=float(h)) for k, (m, h) in enumerate(lines)]
    pk = pd.DataFrame(rows)

    def _ionf(n, a):
        return C.format_formula(T._ion_counts(n, a)) + "-"

    def fake_assign(sid, context="ambient-air", **kw):
        led = L.new_ledger(pd.DataFrame([("p1", pmz, 1e5), ("p2", lmz, 1e4)], columns=["peak_id", "mz", "height"]))
        for pid, (n, a) in (("p1", PB), ("p2", R)):
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=_ionf(n, a), ion_score=0.9,
                                compound_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                                confidence="High", commentary="stub")
        T.apply_tiers(led)
        led.loc[led["role"] == L.ROLE_M0, "tier"] = T.TIER_ASSIGNED
        return {"ledger": led, "stats": {"noise_edge_cps": 50.0, "height_gate_cps": 50.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    seen = {}
    sat_rows, remove = IC.satellite_rows, IC.remove_signature_vetoed

    def spy_sat(*a, **kw):
        seen["sat"] = kw.get("exempt")
        return sat_rows(*a, **kw)

    def spy_remove(*a, **kw):
        seen["remove"] = kw.get("exempt")
        return remove(*a, **kw)

    monkeypatch.setattr(IC, "satellite_rows", spy_sat)
    monkeypatch.setattr(IC, "remove_signature_vetoed", spy_remove)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pk[pk["sample_item_id"] == sid]
                        [["peak_id", "mz", "height"]].reset_index(drop=True))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A_, "run", fake_assign)
    d = tmp_path / "a"
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3", batch="test batch", out_dir=str(d), k_min=2, k_max=3,
           min_gain=0.0, n_jobs=1, resolving_power=Resolution.from_dict(ORBI), mass_axis="off",
           residual=False, log=quiet)
    summ = json.load(open(os.path.join(d, "batch_summary.json")))
    merged = pd.read_csv(os.path.join(d, "merged_ledger.csv"))
    tab = pd.read_csv(os.path.join(d, "tables", "isotopologue_rows.csv"))
    iso = pd.read_csv(os.path.join(d, "tables", "iso_checks.csv"))
    lev = pd.read_csv(os.path.join(d, "tables", "evidence_levels.csv"))
    gates = summ["merge_gates"]
    assert gates["isotopologue"]["n_stripped"] == 1 and gates["isotopologue"]["n_parent_removed"] == 1, gates
    assert gates["element_signature"]["removed"] == 1, gates["element_signature"]
    assert not ({PB, R} & _keys(merged)), merged
    assert len(tab) == 1 and tab["verdict"].iloc[0] == IC.SAT_PARENT_REMOVED, tab.to_dict("records")
    assert not (iso["check"] == "SAT").any()
    req = iso[(iso["check"] == "REQ") & (iso["neutral_formula"] == PB[0])].iloc[0]
    assert bool(req["veto"]) and "81Br" in str(req["line"])
    r = lev[(lev["neutral_formula"] == R[0]) & (lev["adduct"] == R[1])]
    assert not len(r) or not r["iso_veto"].astype(bool).any(), r.to_dict("records")
    assert seen["sat"] == seen["remove"] and isinstance(seen["sat"], frozenset) and len(seen["sat"]) > 0
