"""Rule K's 13C-shadow line test and its merged-ledger gate (peaky/batch/label_twins.py
`_shadows` / `shadow_gate`; docs/EVIDENCE_LEVELS.md §3 `label_veto`).

A 14N line CxHyNOz- of a labelled-nitrate batch sits 8.106 mDa (13C + H - N) below the
13C line of P = C(x+1)H(y+1)Oz-. On an Orbitrap-class batch, when P's ion is committed,
the line can be a fixed fraction of that 13C line rather than X's cluster: per spectrum
it follows P's 13C line (median ratio <= 1/3, sd(log) <= 0.15, r(log) >= 0.9 over >= 50
spectra, the +8.1 mDa line confirmed as P's 13C line by its ratio to P's 12C line) more
tightly than it follows its own 15N twin (the cluster-k sd, else the raw sd of
log(h14 / h15)). Such a line is `shadow`: refuted (veto) and alien like an absent one,
never untied, out of the cluster-k reference pool, and every Assigned merged reading on it
is Candidate (`shadow_gate`, no re-vote).

Synthetic series only (tests/test_label_twins.py's helpers).

Run: pytest tests/test_label_twins_shadow.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from tests.test_evidence import ledger, m0
from tests.test_label_twins import LABEL, N_SPECTRA, NO3, NO3L, REFS, _refs_spec, _row, _series, k_of

QUIET = lambda *a: None   # noqa: E731
ACID = "[M-H]-"
R13 = 0.0107              # per-carbon 13C/12C height ratio (chem.isotopes.R_13C_PER_C)

#: the shadow: X's 14N line [X+NO3]- = C9H16NO7- runs at 0.11x the 13C line of P = C10H17O7-
X = "C9H16O4"
P = "C10H18O7"
#: a real cluster whose 13C neighbour co-varies (sd 0.10) but which follows its own twin tighter
X_TWIN, P_TWIN = "C8H14O4", "C9H16O7"
#: the +8.1 mDa line is not P's 13C line: it reads 3x what P's 12C line predicts
X_NOT13C, P_NOT13C = "C7H12O4", "C8H14O7"
#: P's ion is not committed
X_NOPARENT, P_NOPARENT = "C11H18O5", "C12H20O8"


def _parent_rows(i, when, sid, parent, nC, *, scale, phase, c13_x=1.0, rng=None, sd13=0.02, amp=1.0):
    """P's [M-H]- 12C line and its 13C line in spectrum i; returns the 13C height. `amp`: P's
    dynamic range, the sine's amplitude on 1.5."""
    h12 = scale * (1.5 + amp * np.sin(2 * np.pi * i / 17.0 + phase))
    h13 = h12 * nC * R13 * c13_x * float(np.exp(rng.normal(0.0, sd13)))
    m12 = C.ion_mz(parent, ACID)
    return h13, [dict(sample_item_id=sid, datetime_utc=when, mz=m12, height=h12, area=h12, role="M0",
                      neutral_formula=parent, adduct=ACID),
                 dict(sample_item_id=sid, datetime_utc=when, mz=m12 + LT.D13C, height=h13, area=h13,
                      role="iso_child", neutral_formula=None, adduct=None)]


def _shadow_series(cases=None, n=N_SPECTRA) -> pd.DataFrame:
    """The reference clusters (TL._series) plus, per case, P's lines and X's two nitrate lines.

    case: (x, parent, kind, kw). kind 'shadow': h14 = s x h13(P) x noise(sd14), the 15N line
    h15 = h14 / k(t) x noise(sd15) (so h14 / h15 also follows the cluster ratio, looser), P's
    dynamic range `amp`, the 15N line in the first `share15` of the spectra only (0 = never);
    'twin': h14 = k(t) x h15 x noise(0.03), P's 13C line = h14 / s x noise(0.10)."""
    cases = cases if cases is not None else DEFAULT_CASES
    ts = _series(_refs_spec())
    t0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        sid, when = f"s{i:03d}", t0 + pd.Timedelta(minutes=20 * i)
        for x, parent, kind, kw in cases:
            nC = C.parse_formula(parent)["C"]
            m14, m15 = C.ion_mz(x, NO3), C.ion_mz(x, NO3L)
            if kind == "shadow":
                h13, pr = _parent_rows(i, when, sid, parent, nC, scale=kw.get("scale", 2e5), phase=kw.get("phase", 0.3),
                                       c13_x=kw.get("c13_x", 1.0), rng=rng, amp=kw.get("amp", 1.0))
                h14 = kw.get("s", 0.11) * h13 * float(np.exp(rng.normal(0.0, kw.get("sd14", 0.05))))
                h15 = h14 / k_of(i) * float(np.exp(rng.normal(0.0, kw.get("sd15", 0.15))))
            else:
                h15 = 3e4 * (1.5 + np.sin(2 * np.pi * i / 23.0 + kw.get("phase", 1.1)))
                h14 = k_of(i) * h15 * float(np.exp(rng.normal(0.0, 0.03)))
                h13 = h14 / kw.get("s", 0.11) * float(np.exp(rng.normal(0.0, 0.10)))
                m12 = C.ion_mz(parent, ACID)
                h12 = h13 / (nC * R13)
                pr = [dict(sample_item_id=sid, datetime_utc=when, mz=m12, height=h12, area=h12, role="M0",
                           neutral_formula=parent, adduct=ACID),
                      dict(sample_item_id=sid, datetime_utc=when, mz=m12 + LT.D13C, height=h13, area=h13,
                           role="iso_child", neutral_formula=None, adduct=None)]
            rows += pr
            rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=m14, height=h14, area=h14, role="M0",
                             neutral_formula=x, adduct=NO3))
            if i < kw.get("share15", 1.0) * n:
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=m15, height=h15, area=h15, role="M0",
                                 neutral_formula=x, adduct=NO3L))
    return pd.concat([ts, pd.DataFrame(rows)], ignore_index=True)


DEFAULT_CASES = [
    (X, P, "shadow", {}),
    (X_TWIN, P_TWIN, "twin", {}),
    (X_NOT13C, P_NOT13C, "shadow", {"c13_x": 3.0, "phase": 1.7}),
    (X_NOPARENT, P_NOPARENT, "shadow", {"phase": 2.4}),
]


def _frames(cases=None, parents=None, only14=()) -> dict:
    """Every reference and every case's X committed on both nitrate adducts (those in `only14`
    on [M+NO3]- alone); the parents in `parents` (default: every case's but P_NOPARENT's)
    committed on [M-H]-."""
    cases = cases if cases is not None else DEFAULT_CASES
    parents = parents if parents is not None else [p for _, p, _, _ in cases if p != P_NOPARENT]
    rows = [m0(f"r{k}", n, adduct=NO3L, ion=n + "^NO3", mz=C.ion_mz(n, NO3L)) for k, n in enumerate(REFS)]
    for k, (x, _, _, _) in enumerate(cases):
        if x not in only14:
            rows.append(m0(f"a{k}", x, adduct=NO3L, ion=x + "^NO3", mz=C.ion_mz(x, NO3L)))
        rows.append(m0(f"b{k}", x, adduct=NO3, ion=x + "NO3", mz=C.ion_mz(x, NO3)))
    rows += [m0(f"p{k}", p, adduct=ACID, mz=C.ion_mz(p, ACID)) for k, p in enumerate(parents)]
    return {"f1": EV.trim(ledger(rows))}


def _measure(ts=None, frames=None, klass="orbitrap"):
    return LT.measure(_shadow_series() if ts is None else ts, _frames() if frames is None else frames, LABEL,
                      klass=klass, log=QUIET)


@pytest.fixture(scope="module")
def table():
    return _measure()


# =========================================================================== the line test
def test_the_shadow_offset_is_13c_plus_h_minus_n():
    """SHADOW_DMZ: the 13C line of C(x+1)H(y+1)Oz- sits 8.1059 mDa above CxHyNOz-."""
    assert LT.SHADOW_DMZ == pytest.approx(LT.D13C + C.M["C"] + C.M["H"] - C.M["N"])
    assert LT.SHADOW_DMZ == pytest.approx(0.0081059, abs=1e-7)
    assert C.ion_mz(P, ACID) + LT.D13C - C.ion_mz(X, NO3) == pytest.approx(LT.SHADOW_DMZ, abs=1e-9)
    assert (LT.SHADOW_S_MAX, LT.SHADOW_SD_MAX, LT.SHADOW_R_MIN, LT.SHADOW_N_MIN) == (pytest.approx(1 / 3), 0.15, 0.9, 50)
    assert LT.SHADOW_C13 == (0.5, 1.6)


def test_a_line_that_follows_its_parents_13c_line_is_a_shadow(table):
    """The shadow: a fixed 0.11x of P's 13C line, its twin looser. It would track the cluster
    ratio (k_sd under 0.25), but 'shadow' outranks 'tracks': refuted, alien, never untied."""
    r = _row(table, X)
    assert r.cluster_k                                   # the cluster-k test alone reads it as X's cluster
    assert r.line_verdict == "shadow" and r.shadow
    assert r.shadow_parent == "C10H17O7-"
    assert r.shadow_mz == pytest.approx(C.ion_mz(X, NO3) + LT.SHADOW_DMZ)
    assert r.shadow_n == N_SPECTRA
    assert r.shadow_ratio == pytest.approx(0.11, rel=0.05)
    assert r.shadow_sd < 0.08 and r.shadow_r > 0.95
    assert r.shadow_c13 == pytest.approx(1.0, abs=0.05)
    assert r.shadow_sd < r.shadow_twin_sd
    assert r.committed and r.veto and r.alien and not r.untie
    assert "13C shadow" in r.note and "C10H17O7-" in r.note
    assert (X, NO3) in LT.veto(table) and (X, NO3) in LT.facts(table)["alien"]


def test_a_line_that_follows_its_own_twin_tighter_is_not_a_shadow(table):
    """A cluster co-varying with a neighbour's 13C line (sd 0.10) that follows its own 15N
    twin at sd 0.03: the twin explains it better -- it keeps 'tracks'."""
    r = _row(table, X_TWIN)
    assert r.shadow_sd <= LT.SHADOW_SD_MAX and r.shadow_r >= LT.SHADOW_R_MIN
    assert r.shadow_ratio <= LT.SHADOW_S_MAX and 0.5 <= r.shadow_c13 <= 1.6
    assert r.shadow_twin_sd < r.shadow_sd
    assert not r.shadow and r.line_verdict == "tracks" and not r.veto


def test_a_line_8_mda_above_that_is_not_the_parents_13c_line_makes_no_shadow(table):
    """The +8.1 mDa line reads 3x the 13C line P's 12C line predicts: it is not P's 13C line."""
    r = _row(table, X_NOT13C)
    assert r.shadow_c13 == pytest.approx(3.0, rel=0.05)
    assert r.shadow_sd < 0.08 and r.shadow_r > 0.95
    assert not r.shadow and r.line_verdict != "shadow"


def test_a_parent_that_is_not_committed_makes_no_shadow(table):
    r = _row(table, X_NOPARENT)
    assert not r.shadow and r.line_verdict != "shadow"
    assert pd.isna(r.shadow_parent) or r.shadow_parent == ""


@pytest.mark.parametrize("cut,kw", [
    ("ratio", {"s": 0.5}),                      # 0.5x the 13C line: above 1/3
    ("sd", {"sd14": 0.25}),                     # sd(log) 0.25 > 0.15
])
def test_each_cut_of_the_line_test_can_fail(cut, kw):
    cases = [(X, P, "shadow", kw)]
    t = _measure(_shadow_series(cases), _frames(cases))
    r = _row(t, X)
    assert not r.shadow and r.line_verdict != "shadow", cut


@pytest.mark.parametrize("sd14", [0.17, 0.20])
def test_the_sd_cut_alone_keeps_a_loose_follower_out(sd14):
    """sd(log h14 / h13) between SHADOW_SD_MAX and the twin's sd, every other cut passed (n, the
    ratio, r, the 13C check, the twin comparison): the sd cut alone says no. A real cluster line
    can follow a neighbour's 13C line this loosely and its own twin only a little looser."""
    cases = [(X, P, "shadow", {"sd14": sd14, "sd15": 0.35})]
    r = _row(_measure(_shadow_series(cases), _frames(cases)), X)
    assert LT.SHADOW_SD_MAX < r.shadow_sd < LT.SHADOW_SD_MAX + 0.07
    assert r.shadow_sd < r.shadow_twin_sd
    assert r.shadow_n >= LT.SHADOW_N_MIN and r.shadow_ratio <= LT.SHADOW_S_MAX
    assert r.shadow_r >= LT.SHADOW_R_MIN and LT.SHADOW_C13[0] <= r.shadow_c13 <= LT.SHADOW_C13[1]
    assert not r.shadow and r.line_verdict != "shadow" and not r.veto


def test_the_r_cut_alone_keeps_a_line_beside_a_flat_parent_out():
    """P barely varies (its sine's amplitude 0.1 on 1.5): a line at a fixed share of its 13C line
    reads a small sd, but nothing co-varies -- r 0.74, every other cut passed. A constant ratio
    to a constant line is no evidence that the line is that line's shadow."""
    cases = [(X, P, "shadow", {"amp": 0.1, "sd14": 0.05})]
    r = _row(_measure(_shadow_series(cases), _frames(cases)), X)
    assert r.shadow_r < LT.SHADOW_R_MIN - 0.1
    assert r.shadow_sd <= LT.SHADOW_SD_MAX and r.shadow_sd < r.shadow_twin_sd
    assert r.shadow_n >= LT.SHADOW_N_MIN and r.shadow_ratio <= LT.SHADOW_S_MAX
    assert LT.SHADOW_C13[0] <= r.shadow_c13 <= LT.SHADOW_C13[1]
    assert not r.shadow and r.line_verdict == "tracks" and not r.veto


def test_a_twin_too_sparse_for_the_cluster_k_test_is_judged_by_its_raw_sd():
    """The 15N twin is co-detected in 60 spectra, under the cluster-k test's CODETECT_MIN: its
    k_sd is NaN, and the raw sd(log h14 / h15) over those 60 spectra stands in -- the line is
    still judged, and follows P's 13C line tighter: a shadow."""
    cases = [(X, P, "shadow", {"share15": 0.5})]
    ts = _shadow_series(cases)
    r = _row(_measure(ts, _frames(cases)), X)
    assert pd.isna(r.k_sd) and r.n_codetected == N_SPECTRA // 2 < LT.CODETECT_MIN
    h = ts[ts["neutral_formula"] == X].pivot_table(index="sample_item_id", columns="adduct", values="height")
    h = h.dropna()
    assert len(h) == N_SPECTRA // 2 >= LT.PARTNER_MIN_SPECTRA
    assert r.shadow_twin_sd == pytest.approx(float(np.std(np.log(h[NO3] / h[NO3L]), ddof=1)))
    assert r.shadow_sd < r.shadow_twin_sd
    assert r.shadow and r.line_verdict == "shadow" and r.veto


def test_a_line_with_no_15n_twin_is_never_a_shadow():
    """No 15N line at all (X committed on [M+NO3]- alone): there is no twin to compare with, so
    the line is not a shadow however tightly it follows P's 13C line -- rule K reads it as
    absent, and the gate leaves every reading of it alone (a 14N organonitrate reading of the
    same line among them)."""
    cases = [(X, P, "shadow", {"share15": 0.0})]
    t = _measure(_shadow_series(cases), _frames(cases, only14=(X,)))
    r = _row(t, X)
    assert pd.isna(r.shadow_twin_sd) and pd.isna(r.k_sd) and r.n_codetected == 0
    assert r.shadow_n >= LT.SHADOW_N_MIN and r.shadow_sd <= LT.SHADOW_SD_MAX and r.shadow_r >= LT.SHADOW_R_MIN
    assert not r.shadow and r.line_verdict == "absent"
    m14 = C.ion_mz(X, NO3)
    merged = _merged([[m14, X, NO3, "Assigned", pd.NA], [m14, "C9H17NO7", ACID, "Assigned", pd.NA]])
    out = LT.shadow_gate(merged, t, mass_scale=None, log=QUIET)
    assert out["lines"] == 0 and out["demoted"] == 0 and list(merged["tier"]) == ["Assigned", "Assigned"]


def test_only_a_cho_x_has_a_parent():
    """P = C(x+1)H(y+1)Oz- of the 14N line CxHyNOz-: X CHO only. An X with an N of its own (its
    ion holds two N) or another element has none -- its 13C neighbour would keep that N."""
    assert LT._parent_of(X) == {"C": 10, "H": 17, "O": 7}
    assert LT._parent_of("C9H15NO4") is None
    assert LT._parent_of("C9H16O4S") is None


def test_fifty_codetected_spectra_are_needed():
    cases = [(X, P, "shadow", {})]
    ts = _shadow_series(cases)
    keep = ~((ts["neutral_formula"] == X) & (ts["adduct"] == NO3) & (ts["sample_item_id"] >= "s049"))
    r = _row(_measure(ts[keep], _frames(cases)), X)
    assert r.shadow_n == 49 and not r.shadow


def test_a_shadow_leaves_the_cluster_k_reference_pool():
    """X is a bright CHO cluster committed on [M+^NO3]-: a reference by the cluster-k rule.
    As a shadow it is out of the pool, and every other line's k is the pool's without it."""
    t = _measure()
    assert not _row(t, X).reference
    assert all(_row(t, n).reference for n in REFS)
    tr = LT._Traces(_shadow_series())
    neutrals = sorted(set(REFS) | {c[0] for c in DEFAULT_CASES})
    pool = [n for n in neutrals if n != X]
    ref = LT._cluster_k(tr, neutrals, ref_pool=pool).set_index("neutral_formula")
    for n in REFS + [X_TWIN]:
        assert _row(t, n).k_sd == pytest.approx(ref.at[n, "k_sd"]), n
    full = LT._cluster_k(tr, neutrals, ref_pool=neutrals).set_index("neutral_formula")
    assert full.at[X, "reference"]                       # with the pool as it was


def test_the_line_test_runs_on_an_orbitrap_class_batch_only():
    """Out of class (a TOF, or no class) nothing is measured: the statistics are empty, the
    verdicts are the cluster-k test's."""
    for klass in (None, "tof"):
        t = _measure(klass=klass)
        r = _row(t, X)
        assert not r.shadow and r.line_verdict == "tracks" and pd.isna(r.shadow_ratio), klass
        assert not t["shadow"].any()


def test_the_table_carries_the_shadow_columns_and_the_summary_counts_them(table):
    cols = ["shadow_parent", "shadow_mz", "shadow_n", "shadow_ratio", "shadow_sd", "shadow_r", "shadow_c13",
            "shadow_twin_sd", "shadow"]
    assert [c for c in LT.TABLE_COLUMNS if c.startswith("shadow")] == cols
    assert list(table.columns) == list(LT.TABLE_COLUMNS)
    assert table["shadow"].dtype == bool
    assert "shadow" in LT.LINE_VERDICTS and "shadow" in LT.FOREIGN and "shadow" in LT.REFUTED
    s = LT.summary(table, LABEL)
    assert s["lines_shadow"] == 1
    assert list(LT._empty().columns) == list(LT.TABLE_COLUMNS)


# =========================================================================== the merged gate
def _merged(rows):
    return pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "tier_reason"])


def test_the_gate_makes_every_assigned_reading_on_a_shadow_line_candidate(table):
    m14 = C.ion_mz(X, NO3)
    merged = _merged([
        [m14 * (1 + 0.4e-6), X, NO3, "Assigned", pd.NA],                      # the shadow reading
        [m14 * (1 - 1.0e-6), "C8H16NO6", ACID, "Assigned", "kept"],         # another reading of the same line
        [m14 * (1 + 0.2e-6), "C9H17NO7", "[M]-.", "Candidate", pd.NA],      # already Candidate: untouched
        [C.ion_mz(P, ACID), P, ACID, "Assigned", pd.NA],                      # the parent: untouched
        [C.ion_mz(X_TWIN, NO3), X_TWIN, NO3, "Assigned", pd.NA],             # a tracking line: untouched
        [m14 * (1 + 9e-6), "C9H19NO2S", ACID, "Assigned", pd.NA],            # outside the merge window
    ])
    out = LT.shadow_gate(merged, table, mass_scale={"merge_ppm": 3.0}, log=QUIET)
    assert list(merged["tier"]) == ["Candidate", "Candidate", "Candidate", "Assigned", "Assigned", "Assigned"]
    assert "13C shadow" in merged.at[0, "tier_reason"] and "C10H17O7-" in merged.at[0, "tier_reason"]
    assert merged.at[1, "tier_reason"].startswith("kept | Candidate")
    assert pd.isna(merged.at[2, "tier_reason"])
    assert out == {"ran": True, "lines": 1, "demoted": 2,
                   "demoted_rows": [f"{X} {NO3}", f"C8H16NO6 {ACID}"]}


@pytest.mark.parametrize("scale,ppm", [
    ({"merge_ppm": 3.0}, 3.0),
    ({"merge_ppm": 6.0, "tol_ppm": 3.0}, 6.0),
    ({"tol_ppm": 4.0}, 4.0),          # no merge window measured: the binning tolerance
    (None, 6.0),                      # no scale: 6 ppm
])
def test_the_gate_window_is_the_batchs_merge_window(table, scale, ppm):
    """A reading 0.9x the merge window off the shadow line is on it; one 1.1x off is not."""
    m14 = C.ion_mz(X, NO3)
    merged = _merged([[m14 * (1 + 0.9 * ppm * 1e-6), "C9H17NO7", ACID, "Assigned", pd.NA],
                      [m14 * (1 - 1.1 * ppm * 1e-6), "C9H17NO7", ACID, "Assigned", pd.NA]])
    out = LT.shadow_gate(merged, table, mass_scale=scale, log=QUIET)
    assert list(merged["tier"]) == ["Candidate", "Assigned"] and out["demoted"] == 1


def test_the_gate_is_a_no_op_without_a_shadow_line():
    m14 = C.ion_mz(X, NO3)
    merged = _merged([[m14, X, NO3, "Assigned", pd.NA]])
    for t in (None, LT._empty(), _measure(klass=None)):
        out = LT.shadow_gate(merged, t, mass_scale=None, log=QUIET)
        assert out["demoted"] == 0 and list(merged["tier"]) == ["Assigned"]
    # a table written before the shadow columns existed reads no shadow line
    old = _measure(klass=None).drop(columns=[c for c in LT.TABLE_COLUMNS if c.startswith("shadow")])
    assert LT.shadow_gate(merged, old, log=QUIET)["demoted"] == 0


# =========================================================================== the batch, end to end
def test_a_labelled_nitrate_batch_demotes_the_shadow_reading(tmp_path, monkeypatch):
    """assign_batch.run on NO3+NO3_15N, Orbitrap-class: the shadow line's [X+NO3]- reading is
    Candidate on the merged ledger with the gate's note, the gate's counts are in
    merge_gates['label_shadow'], and the pooled pair is refuted (5b) by its label veto."""
    import json

    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    cases = [(X, P, "shadow", {}), (X_TWIN, P_TWIN, "twin", {})]
    ts = _shadow_series(cases)
    pk = ts[["sample_item_id", "datetime_utc", "mz", "height"]].copy()
    pk["sample_item_name"] = "n_" + pk["sample_item_id"]
    commits = ([(n, NO3L) for n in REFS] + [(x, a) for x, _, _, _ in cases for a in (NO3L, NO3)]
               + [(p, ACID) for _, p, _, _ in cases])

    def fake_assign(sid, context="ambient-air", **kw):
        ids = [f"P{j}" for j in range(len(commits))]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": [C.ion_mz(n, a) for n, a in commits],
                                         "height": [5000.0] * len(ids)}))
        for pid, (n, a) in zip(ids, commits):
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=n, ion_score=0.95,
                                eff_score=0.9, tied=False, alternatives=None, compound_score=0.95, ppm_error=0.1,
                                pass_no=1, method="cheminfo", confidence="High", commentary=f"Pass 1: {n} {a}")
        T.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    seen = {}
    real_gate = LT.shadow_gate

    def spy(merged, table, *, mass_scale=None, log=print):
        seen["mass_scale"] = mass_scale
        return real_gate(merged, table, mass_scale=mass_scale, log=log)

    monkeypatch.setattr(LT, "shadow_gate", spy)
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3+NO3_15N", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=100_000, log=lambda *a: None)
    table = pd.read_csv(tmp_path / "tables" / "label_twins.csv")
    assert list(table.columns) == list(LT.TABLE_COLUMNS)
    assert _row(table, X).line_verdict == "shadow" and _row(table, X_TWIN).line_verdict == "tracks"
    summ = json.load(open(tmp_path / "batch_summary.json"))
    gate = summ["merge_gates"]["label_shadow"]
    assert gate["ran"] and gate["lines"] == 1 and gate["demoted"] == 1 and gate["demoted_rows"] == [f"{X} {NO3}"]
    assert summ["evidence_levels"]["label_twins"]["lines_shadow"] == 1
    # the gate's window is the batch's own merge window
    from peaky.batch.iso_checks import _merge_ppm
    assert seen["mass_scale"] is not None
    assert _merge_ppm(seen["mass_scale"]) == pytest.approx(summ["mass_scale"]["merge_ppm"])
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False).set_index(["neutral_formula", "adduct"])
    assert merged.at[(X, NO3), "tier"] == "Candidate"
    assert "13C shadow" in merged.at[(X, NO3), "tier_reason"]
    assert merged.at[(X, NO3), "evidence_level"] == "5b"
    assert merged.at[(X_TWIN, NO3), "tier"] == "Assigned"
    assert merged.at[(P, ACID), "tier"] == "Assigned"
