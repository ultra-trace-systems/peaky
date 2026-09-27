"""Rule K: the 15N twin arbitrates labelled-nitrate readings (EVIDENCE_LEVELS §3
`label_untie` / `label_veto`, §4 row 1).

`batch/label_twins.measure` reads the stamped batch time series and the pooled
per-file ledgers of a 15N-labelled nitrate channel. In one direction the 14N line
[X+NO3]- is X's cluster (not the same-ion organonitrate [X'-H]-) when its 14N/15N
ratio follows the batch's cluster ratio k_cl(t) -- the per-spectrum median over
the bright CHO acid clusters, the pair under test left out -- and then the
arbiter's alias-only tie on it is cleared (the pooled level reads the acid
branch). In the other the labelled [Y+^NO3]- reading is refuted when the 14N twin
its reagent's impurity must carry, (1 - purity) / purity of its height, is absent
in the spectra where the batch's own detection curve says it would be seen: a new
hard input (5b). Both facts exist only on the pooled batch of a labelled-nitrate
profile; a leak would move the golden vectors, and a test here says so.

Run: pytest tests/test_label_twins.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from tests.test_evidence import GOLDEN, _pooled, _vector, child, ledger, m0

NO3, NO3L = LT.NO3, LT.NO3L
N_SPECTRA = 120
REFS = ["C10H16O4", "C9H14O3", "C8H12O4", "C7H10O4", "C10H18O4", "C8H12O5"]
LABEL = P.resolve("NO3+NO3_15N")


def k_of(i: int) -> float:
    """The batch's cluster 14N/15N ratio in spectrum i (it follows the reagent)."""
    return 0.15 * (1.4 + np.sin(2 * np.pi * i / 40.0))


def _series(spec: dict) -> pd.DataFrame:
    """A stamped batch time series of labelled-nitrate clusters. spec[neutral]:
    h15 (the 15N line's scale), phase, q (the 14N line's factor on k_cl; None = no
    14N line), noise (sd of a log-normal factor on the 14N line), flat (the 14N
    line at a constant ratio instead of following k), share (the 14N line in the
    first `share` of the spectra only), share15 (the 15N line in the first
    `share15` of the spectra only; 0 = never), twin_shift (Da added to the 14N
    line's position), c13 (write the 15N line's 13C line; default True)."""
    t0 = pd.Timestamp("2026-08-11 00:00", tz="UTC")
    rng = np.random.default_rng(7)
    rows = []
    for i in range(N_SPECTRA):
        sid = f"s{i:03d}"
        when = t0 + pd.Timedelta(minutes=20 * i)
        for n, o in spec.items():
            h15 = o.get("h15", 5000.0) * (1.5 + np.sin(2 * np.pi * i / 25.0 + o.get("phase", 0.0)))
            mz15 = C.ion_mz(n, NO3L)
            has15 = i < o.get("share15", 1.0) * N_SPECTRA
            if has15:
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=mz15, height=h15, area=h15, role="M0",
                                 neutral_formula=n, adduct=NO3L))
            if has15 and o.get("c13", True):
                h13 = h15 * LT.C13_PER_C * C.parse_formula(n).get("C", 0)
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=mz15 + LT.D13C, height=h13, area=h13,
                                 role="iso_child", neutral_formula=None, adduct=None))
            q = o.get("q", 1.0)
            if q is None or i >= o.get("share", 1.0) * N_SPECTRA:
                continue
            k = 0.2 if o.get("flat") else k_of(i)
            h14 = q * k * h15 * float(np.exp(rng.normal(0.0, o.get("noise", 0.03))))
            rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=C.ion_mz(n, NO3) + o.get("twin_shift", 0.0),
                             height=h14, area=h14, role="M0", neutral_formula=n, adduct=NO3))
    return pd.DataFrame(rows)


def _frames(neutrals, *, no3=(), tied=(), only14=()) -> dict:
    """One per-file ledger committing every neutral on [M+^NO3]- (those in
    `only14` on [M+NO3]- alone) and those in `no3` on [M+NO3]- (tied on the
    same-ion organonitrate alias when in `tied`)."""
    rows = []
    for k, n in enumerate(neutrals):
        if n in only14:
            rows.append(m0(f"b{k}", n, adduct=NO3, ion=n + "NO3", mz=C.ion_mz(n, NO3), tied=n in tied))
            continue
        rows.append(m0(f"a{k}", n, adduct=NO3L, ion=n + "^NO3", mz=C.ion_mz(n, NO3L)))
        if n in no3:
            rows.append(m0(f"b{k}", n, adduct=NO3, ion=n + "NO3", mz=C.ion_mz(n, NO3), tied=n in tied))
    return {"f1": EV.trim(ledger(rows))}


def _alias_ledger(neutral, *, other=None, margin=0.03) -> pd.DataFrame:
    """A per-file ledger with one tied [X+NO3]- row whose runner-up is the
    same-ion organonitrate [X+HNO3-H]- (or `other`, a different ion)."""
    alt_n = C.format_formula({k: C.parse_formula(neutral).get(k, 0) + d
                              for k, d in (("C", 0), ("H", 1), ("N", 1), ("O", 3))})
    alt = {"formula": alt_n, "adduct": "[M-H]-", "eff_score": 0.80 - margin}
    alts = [alt] + ([{"formula": other[0], "adduct": other[1], "eff_score": 0.80 - margin}] if other else [])
    row = m0("b0", neutral, adduct=NO3, ion=neutral + "NO3", mz=C.ion_mz(neutral, NO3), tied=True)
    led = ledger([row])
    led["alternatives"] = [json.dumps(alts)]
    led["eff_score"] = [0.80]
    led["commentary"] = ["Pass 1: tie"]
    return led


def _measure(spec, frames=None, alias=None, prof=LABEL):
    return LT.measure(_series(spec), frames if frames is not None else _frames(list(spec)), prof,
                      alias_ties=alias, log=lambda *a: None)


def _refs_spec(**extra) -> dict:
    spec = {n: {"phase": 0.6 * k} for k, n in enumerate(REFS)}
    spec.update(extra)
    return spec


def _row(table, neutral, adduct=NO3) -> pd.Series:
    t = table[(table.neutral_formula == neutral) & (table.adduct == adduct)]
    assert len(t) == 1, (neutral, adduct)
    return t.iloc[0]


# --------------------------------------------------------------------------- scope
def test_only_a_labelled_nitrate_profile_is_in_scope():
    assert LT.in_scope(P.PROFILES["NO3_15N"]) and LT.in_scope(LABEL)
    for name in ("NO3", "Br", "Ur", "NH4_15N"):
        assert not LT.in_scope(P.PROFILES[name]), name
    assert not LT.in_scope(P.resolve("Br+NO3"))
    spec = _refs_spec()
    for prof in (P.PROFILES["NO3"], P.PROFILES["Br"], P.PROFILES["Ur"]):
        t = _measure(spec, prof=prof)
        assert t.empty and list(t.columns) == list(LT.TABLE_COLUMNS)
        assert LT.untie(t) == set() and LT.veto(t) == {}
        assert LT.summary(t, prof)["in_scope"] is False
    assert LT.measure(None, _frames(REFS), LABEL).empty


def test_the_twin_fraction_is_the_reagent_impurity():
    assert LT.twin_fraction(P.PROFILES["NO3_15N"]) == pytest.approx(0.02 / 0.98)
    assert LT.twin_fraction(LABEL) == pytest.approx(0.02 / 0.98)
    from types import SimpleNamespace
    assert LT.twin_fraction(SimpleNamespace(purity=0.99)) == pytest.approx(0.01 / 0.99)
    from peaky.chem import isotopes
    p0 = isotopes.LABEL_PURITY_15N
    assert LT.twin_fraction(SimpleNamespace(purity=None)) == pytest.approx((1 - p0) / p0)


# --------------------------------------------------------------------------- the untie (cluster-k)
def test_a_cluster_following_k_passes_and_unties():
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0}})
    alias = {"f1": LT.alias_only_ties(_alias_ledger(x))}
    t = _measure(spec, _frames(list(spec), no3=[x], tied=[x]), alias)
    r = _row(t, x)
    assert r.cluster_k and r.alias_only_tie and r.untie
    assert 0.5 <= r.k_ratio <= 2.0 and r.k_sd <= 0.25 and r.k_r >= 0.8 and r.k_n >= 50
    assert LT.untie(t) == {(x, NO3)}
    s = LT.summary(t, LABEL)
    assert s["references"] == len(REFS) + 1 and s["untie"] == 1 and s["in_scope"] is True   # x is a bright CHO pair too
    assert list(t.columns) == list(LT.TABLE_COLUMNS)


def test_each_clause_of_the_cluster_k_test_can_fail():
    spec = _refs_spec(**{
        "C6H10O4": {"phase": 1.0, "q": 3.0},                  # ratio 3x k_cl: outside the band
        "C6H10O5": {"phase": 1.2, "q": 0.3},                  # ratio 0.3x: outside the band
        "C7H12O4": {"phase": 1.4, "noise": 0.6},              # scatter: sd of log > 0.25
        "C7H12O5": {"phase": 1.6, "flat": True},              # constant ratio: r(log q, log k) < 0.8
        "C9H16O4": {"phase": 1.8, "share": 0.6},              # co-detected in 72 < 100 spectra
    })
    t = _measure(spec)
    assert not _row(t, "C6H10O4").cluster_k and _row(t, "C6H10O4").k_ratio > 2.0
    assert not _row(t, "C6H10O5").cluster_k and _row(t, "C6H10O5").k_ratio < 0.5
    assert not _row(t, "C7H12O4").cluster_k and _row(t, "C7H12O4").k_sd > 0.25
    assert not _row(t, "C7H12O5").cluster_k and _row(t, "C7H12O5").k_r < 0.8
    r = _row(t, "C9H16O4")
    assert not r.cluster_k and r.n_codetected < LT.CODETECT_MIN and pd.isna(r.k_ratio)
    for n in REFS:
        assert _row(t, n).cluster_k, n


def test_references_are_bright_cho_pairs_and_leave_the_tested_pair_out():
    spec = _refs_spec(**{
        "C10H16O4S": {"phase": 1.0},                          # not C/H/O: never a reference
        "C8H14O4": {"phase": 1.3, "h15": 300.0},              # 15N line median below 1000: never a reference
    })
    t = _measure(spec)
    assert not _row(t, "C10H16O4S").reference and not _row(t, "C8H14O4").reference
    assert all(_row(t, n).reference for n in REFS)
    # a reference whose own ratio runs at 5x the others fails: its own line is left out of its k
    spec = _refs_spec()
    spec[REFS[0]] = {"phase": 0.0, "q": 5.0}
    t = _measure(spec)
    assert not _row(t, REFS[0]).cluster_k and _row(t, REFS[0]).k_ratio > 2.0


def test_the_untie_needs_an_alias_only_tie_and_a_committed_14n_reading():
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0}})
    frames = _frames(list(spec), no3=[x], tied=[x])
    # the runner-up is a different ion within the tie margin: the cluster-k test cannot settle that
    other = {"f1": LT.alias_only_ties(_alias_ledger(x, other=("C11H22O5", "[M+Cl]-")))}
    t = _measure(spec, frames, other)
    assert _row(t, x).cluster_k and not _row(t, x).alias_only_tie and not _row(t, x).untie
    # no alias table at all: nothing unties
    assert not _row(_measure(spec, frames, None), x).untie
    # one file alias-only, another not: every file's tie must be alias-only
    both = {"f1": LT.alias_only_ties(_alias_ledger(x)),
            "f2": LT.alias_only_ties(_alias_ledger(x, other=("C11H22O5", "[M+Cl]-")))}
    assert not _row(_measure(spec, frames, both), x).untie
    # the 14N reading is not committed: the line passes but there is nothing to untie
    t = _measure(spec, _frames(list(spec)), {"f1": LT.alias_only_ties(_alias_ledger(x))})
    assert _row(t, x).cluster_k and not _row(t, x).untie


def test_alias_only_ties_reads_the_tier_engines_own_test():
    x = "C10H18O4"
    a = LT.alias_only_ties(_alias_ledger(x))
    assert list(a.itertuples(index=False, name=None)) == [(x, NO3, True)]
    b = LT.alias_only_ties(_alias_ledger(x, other=("C11H22O5", "[M+Cl]-")))
    assert list(b.alias_only) == [False]
    # a different ion outside the tie margin does not keep the tie
    c = LT.alias_only_ties(_alias_ledger(x, other=("C11H22O5", "[M+Cl]-"), margin=0.2))
    assert list(c.alias_only) == [True]
    untied = _alias_ledger(x)
    untied["tied"] = False
    assert LT.alias_only_ties(untied).empty
    assert LT.alias_only_ties(ledger([m0("z", x, adduct="[M-H]-")])).empty
    assert LT.alias_only_ties(pd.DataFrame()).empty


# --------------------------------------------------------------------------- the tracking verdicts (K03)
def test_every_committed_14n_line_gets_a_tracking_verdict():
    lines = {
        "C10H18O3": ({"phase": 2.0}, "tracks"),
        "C7H12O5": ({"phase": 1.4, "noise": 0.6}, "consistent"),        # in band, too noisy to track
        "C6H10O5": ({"phase": 1.2, "q": 0.15}, "consistent"),           # below the cluster share: not an organonitrate
        "C6H10O4": ({"phase": 1.0, "q": 3.0}, "excess"),                # above it: cluster or organonitrate
        "C11H18O6": ({"phase": 0.9, "share15": 0.0}, "absent"),         # no 15N partner at all
        "C11H18O5": ({"phase": 0.7, "share15": 0.15}, "absent"),        # partner in 15 % of the 14N line's spectra
        "C9H16O4": ({"phase": 1.8, "share15": 0.25}, "untestable"),     # 25 %: neither absent nor co-detected enough
        "C9H14O6": ({"phase": 1.6, "share15": 0.5}, "untestable"),
    }
    spec = _refs_spec(**{n: o for n, (o, _) in lines.items()})
    frames = _frames(list(spec), no3=list(lines), only14=["C11H18O6", "C11H18O5"])
    t = _measure(spec, frames)
    for n, (_, verdict) in lines.items():
        r = _row(t, n)
        assert r.line_verdict == verdict and r.committed, (n, r.line_verdict)
        assert r.alien == (verdict in ("excess", "absent", "untestable")), n
        assert r.veto == (verdict in ("excess", "absent")), n
    assert "no 15N partner" in _row(t, "C11H18O6").note and "0 of the 120" in _row(t, "C11H18O6").note
    assert "runs 3.0x its cluster share" in _row(t, "C6H10O4").note
    assert _row(t, "C11H18O5").partner_share == pytest.approx(0.15)
    f = LT.facts(t)
    assert f["alien"] == {(n, NO3) for n, (_, v) in lines.items() if v in ("excess", "absent", "untestable")}
    assert {k for k in f["veto"] if k[1] == NO3} == {(n, NO3) for n, (_, v) in lines.items() if v in ("excess", "absent")}
    assert f["untie"] == set()                                        # nothing tied here
    s = LT.summary(t, LABEL)
    assert s["committed_lines"] == len(lines) and s["lines_absent"] == 2 and s["lines_excess"] == 1
    assert s["lines_consistent"] == 2 and s["lines_untestable"] == 2 and s["lines_tracks"] == 1
    assert s["alien"] == 5 and s["lines_refuted"] == 3
    # the references are the committed labelled clusters only: a bright 14N-only neutral never calibrates k
    assert not _row(t, "C11H18O6").reference and not _row(t, "C11H18O5").reference


def test_an_absent_partner_needs_ten_spectra_of_the_14n_line():
    n = "C11H18O6"
    for share, verdict in ((9 / N_SPECTRA, "untestable"), (10 / N_SPECTRA, "absent")):
        spec = _refs_spec(**{n: {"phase": 0.9, "share15": 0.0, "share": share}})
        t = _measure(spec, _frames(list(spec), no3=[n], only14=[n]))
        assert _row(t, n).line_verdict == verdict, share


def test_a_line_not_committed_on_14n_is_never_alien_nor_vetoed():
    spec = _refs_spec(**{"C6H10O4": {"phase": 1.0, "q": 3.0}})
    t = _measure(spec)                                                # committed on [M+^NO3]- only
    r = _row(t, "C6H10O4")
    assert r.line_verdict == "excess" and not r.committed and not r.alien and not r.veto
    assert LT.facts(t)["alien"] == set()


def test_facts_is_none_for_an_empty_table():
    assert LT.facts(None) is None and LT.facts(LT._empty()) is None
    t = _measure(_refs_spec())
    assert set(LT.facts(t)) == {"untie", "veto", "alien"}


# --------------------------------------------------------------------------- the veto (14N twin)
def test_a_bright_cluster_without_its_14n_twin_is_refuted():
    y = "C12H14O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "q": None}})
    t = _measure(spec)
    r = _row(t, y, NO3L)
    assert r.twin_verdict == "refuted" and r.veto and r.obs == 0 and r.E >= LT.E_MIN
    assert r.f == pytest.approx(0.02 / 0.98)
    assert r.mz_partner == pytest.approx(C.ion_mz(y, NO3L) - LT.DELTA_15N)
    assert "no 14N twin" in r.note
    assert LT.veto(t) == {(y, NO3L): r.note}
    for n in REFS:                                         # the references carry their twin
        assert _row(t, n, NO3L).twin_verdict == "passes", n


def test_the_twin_is_looked_for_below_the_labelled_line():
    y = "C12H14O2"
    # a line 0.997 Da ABOVE the labelled cluster is not its 14N twin
    spec = _refs_spec(**{y: {"phase": 1.0, "twin_shift": 2 * LT.DELTA_15N}})
    assert _row(_measure(spec), y, NO3L).veto
    # 3 ppm off the twin position is outside the 2-ppm window
    spec = _refs_spec(**{y: {"phase": 1.0, "twin_shift": C.ion_mz(y, NO3) * 3e-6}})
    assert _row(_measure(spec), y, NO3L).veto
    spec = _refs_spec(**{y: {"phase": 1.0, "twin_shift": C.ion_mz(y, NO3) * 1e-6}})
    assert not _row(_measure(spec), y, NO3L).veto


def test_a_twin_seen_in_between_is_unclear_and_a_dim_cluster_is_untestable():
    y, z = "C12H14O2", "C11H12O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "share": 0.35},
                         z: {"phase": 1.0, "q": None, "h15": 2.0, "c13": False}})
    t = _measure(spec)
    assert _row(t, y, NO3L).twin_verdict == "unclear" and not _row(t, y, NO3L).veto
    rz = _row(t, z, NO3L)
    # a detection curve of nothing but bright lines says nothing about a dim twin
    assert rz.twin_verdict == "untestable" and not rz.veto and rz.E == 0


def _dim_rows(frames, ts):
    """Dim committed [M-H]- M0s whose 13C lines (expected ~40-210 counts) are never seen."""
    dim = ["C10H16O3", "C9H14O4", "C8H10O4", "C10H14O4"]
    extra, rows = [], []
    for k, n in enumerate(dim):
        extra.append(m0(f"d{k}", n, adduct="[M-H]-", ion=n, mz=C.ion_mz(n, "[M-H]-")))
        for i in range(N_SPECTRA):
            h = 1000.0 * (1.2 + 0.8 * np.sin(i / 7.0 + k))
            rows.append(dict(sample_item_id=f"s{i:03d}",
                             datetime_utc=pd.Timestamp("2026-08-11", tz="UTC") + pd.Timedelta(minutes=20 * i),
                             mz=C.ion_mz(n, "[M-H]-"), height=h, area=h, role="M0", neutral_formula=n,
                             adduct="[M-H]-"))
    frames = dict(frames, f2=EV.trim(ledger(extra)))
    return frames, pd.concat([ts, pd.DataFrame(rows)], ignore_index=True)


def test_the_detection_curve_gates_what_is_testable():
    """A cluster whose expected twin falls where the batch's own 13C lines are
    never detected is untestable, not refuted; a brighter one is refuted."""
    y = "C12H14O2"
    for h15, verdict in ((2500.0, "untestable"), (20000.0, "refuted")):
        spec = _refs_spec(**{y: {"phase": 1.0, "q": None, "h15": h15}})
        frames, ts = _dim_rows(_frames(list(spec)), _series(spec))
        r = _row(LT.measure(ts, frames, LABEL, log=lambda *a: None), y, NO3L)
        assert r.twin_verdict == verdict and r.veto == (verdict == "refuted"), h15
    curve = LT._pdet_curve(LT._Traces(ts), LT._committed(frames))
    assert curve[np.searchsorted(LT.PDET_EDGES, 45.0, side="right") - 1] == 0.0
    assert curve[-1] == pytest.approx(1.0)


def test_an_unmeasured_height_is_never_testable():
    """A detection bin the batch holds no 13C line in takes the populated bin
    below it, and 0 below the first: without dim lines, a dim cluster's missing
    twin proves nothing."""
    y = "C12H14O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "q": None, "h15": 2500.0}})
    ts, frames = _series(spec), _frames(list(spec))
    curve = LT._pdet_curve(LT._Traces(ts), LT._committed(frames))
    lowest = np.searchsorted(LT.PDET_EDGES, 0.0107 * 7 * 5000 * 0.5, side="right") - 1
    assert (curve[:lowest] == 0.0).all() and curve[lowest] > 0.5
    r = _row(LT.measure(ts, frames, LABEL, log=lambda *a: None), y, NO3L)
    assert r.twin_verdict == "untestable" and not r.veto
    assert (LT._pdet_curve(LT._Traces(ts), LT._committed({})) == 0.0).all()


def _scan_from(ts, start):
    """The series as a scan that starts at `start`: the lines below it removed and
    a filler line at the start in every spectrum (a real spectrum's lowest peak
    sits at its scan start)."""
    cut = ts[ts["mz"] >= start]
    fill = cut.groupby("sample_item_id", as_index=False).first()[["sample_item_id", "datetime_utc"]]
    fill = fill.assign(mz=start + 0.001, height=50.0, area=50.0, role="", neutral_formula=None, adduct=None)
    return pd.concat([cut, fill], ignore_index=True)


def test_a_twin_below_the_scan_start_was_never_measured():
    """A labelled reading within 0.997 Da of the scan start cannot be refuted by
    a twin the scan never reached."""
    y = "C4H4O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "h15": 20000.0}})
    ts = _series(spec)
    meas = lambda t: _row(LT.measure(t, _frames(list(spec)), LABEL, log=lambda *a: None), y, NO3L)
    assert meas(ts).twin_verdict == "passes"
    r = meas(_scan_from(ts, C.ion_mz(y, NO3L) - 0.5))                 # the scan starts 0.5 Da below Y's line
    assert r.twin_verdict == "untestable" and not r.veto and r.E == 0
    # with no twin anywhere but the twin inside the scan, the reading is refuted
    spec[y] = {"phase": 1.0, "h15": 20000.0, "q": None}
    ts = _series(spec)
    assert meas(_scan_from(ts, C.ion_mz(y, NO3L) - 2.0)).veto
    assert not meas(_scan_from(ts, C.ion_mz(y, NO3L) - 0.5)).veto


def test_the_scan_start_is_capped_at_the_batch_median():
    """A sparse spectrum whose lowest peak sits high keeps the batch's scan start;
    a spectrum that really starts lower keeps its own."""
    y = "C4H4O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "h15": 20000.0, "q": None}})
    ts = _scan_from(_series(spec), C.ion_mz(y, NO3L) - 2.0)
    sparse = ts[~((ts["sample_item_id"] == "s000") & (ts["mz"] < C.ion_mz(y, NO3L) - 1.0))]
    tr = LT._Traces(sparse)
    assert tr.low[0] == pytest.approx(C.ion_mz(y, NO3L) - 2.0 + 0.001)
    deep = pd.concat([ts, pd.DataFrame([dict(sample_item_id="s001", datetime_utc=ts["datetime_utc"].iloc[0],
                                             mz=40.0, height=50.0, area=50.0, role="")])], ignore_index=True)
    tr = LT._Traces(deep)
    assert tr.low[tr.order.index("s001")] == pytest.approx(40.0)


# --------------------------------------------------------------------------- the level
def _nitrate_rows(x="C10H18O4", *, tied=True, iso=True):
    rows = [m0("h", x, adduct="[M-H]-", ion=x[:-1] + str(int(x[-1])), mz=C.ion_mz(x, "[M-H]-")),
            m0("n14", x, adduct=NO3, ion=x + "NO3", mz=C.ion_mz(x, NO3), tied=tied),
            m0("n15", x, adduct=NO3L, ion=x + "^NO3", mz=C.ion_mz(x, NO3L))]
    if iso:
        rows.append(child("c1", "n15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 10))
    return ledger(rows)


def K(untie=(), veto=None, alien=()) -> dict:
    """Rule K's facts as batch/label_twins.facts hands them to level_pooled."""
    return {"untie": set(untie), "veto": dict(veto or {}), "alien": set(alien)}


def _levels(frame, **kw) -> dict:
    out = EV.level_pooled({"f": frame}, **kw)
    return {a: (lv, ax, why) for a, lv, ax, why in
            zip(out.adduct, out.evidence_level, out.evidence_axes, out.level_reason)}


def test_the_untie_lets_the_acid_branch_read():
    x = "C10H18O4"
    base = _levels(_nitrate_rows(x))
    assert base[NO3][0] == "5b" and "near-tie" in base[NO3][2]
    lv = _levels(_nitrate_rows(x), label=K(untie={(x, NO3)}))
    assert lv[NO3][0] == "3b" and "label_untie" in lv[NO3][1] and "acid branch" in lv[NO3][2]
    assert lv["[M-H]-"] == base["[M-H]-"] and lv[NO3L] == base[NO3L]
    # an untie of a pair that is not tied changes nothing and is not recorded
    lv = _levels(_nitrate_rows(x, tied=False), label=K(untie={(x, NO3)}))
    assert "label_untie" not in lv[NO3][1]
    assert lv == _levels(_nitrate_rows(x, tied=False))
    # other hard inputs still hold
    rows = _nitrate_rows(x)
    rows.loc[rows.peak_id == "n14", "below_assignability"] = True
    assert _levels(rows, label=K(untie={(x, NO3)}))[NO3][0] == "5b"


def test_the_veto_is_a_hard_input_with_its_numbers():
    x = "C10H18O4"
    note = "no 14N twin: seen in 0 of 42 expected detections"
    lv = _levels(_nitrate_rows(x, tied=False), label=K(veto={(x, NO3L): note}))
    level, axes, why = lv[NO3L]
    assert level == "5b" and "label_veto" in axes
    assert why == f"5b: the reagent's two isotopologues refute the cluster reading ({note})"
    assert _levels(_nitrate_rows(x, tied=False), label=K(veto={(x, NO3L): ""}))[NO3L][2] == \
        "5b: the reagent's two isotopologues refute the cluster reading"
    # it outranks the curated identity and the branch, as every hard input does
    rows = _nitrate_rows(x, tied=False)
    rows.loc[rows.peak_id == "n15", "method"] = "known:atmospheric"
    assert _levels(rows, label=K(veto={(x, NO3L): note}))[NO3L][0] == "5b"
    # a veto keyed on another adduct leaves this one alone
    assert _levels(_nitrate_rows(x, tied=False), label=K(veto={(x, NO3): note}))[NO3L][0] != "5b"


def test_ion_only_pairs_never_take_the_facts():
    x = "C10H18O4"
    rows = _nitrate_rows(x)
    io = m0("io", x, adduct="[M]-.", ion=x, mz=C.ion_mz(x, "[M-H]-") + 1.0078, method="ion_only:ea", tied=True)
    frame = pd.concat([rows, ledger([io])], ignore_index=True)
    out = EV.level_pooled({"f": frame}, label=K(untie={(x, "[M]-.")}, veto={(x, "[M]-."): "x"})).set_index("adduct")
    assert not out.loc["[M]-.", "label_untie"] and not out.loc["[M]-.", "label_veto"]
    assert out.loc["[M]-.", "evidence_level"] == "5b"            # still tied: the untie never reaches it


def _j1_rows(j="C11H18O6", *, tied=False):
    """A C11 acid seen deprotonated (13C-backed) and as a 14N [M+NO3]- line only."""
    return ledger([m0("h", j, adduct="[M-H]-", ion=j, mz=C.ion_mz(j, "[M-H]-")),
                   child("hc", "h", "13C+1", 1000.0 * EV.C13_PER_CARBON * 11),
                   m0("n14", j, adduct=NO3, ion=j + "NO3", mz=C.ion_mz(j, NO3), tied=tied),
                   child("nc", "n14", "13C+1", 1000.0 * EV.C13_PER_CARBON * 11)])


def test_an_alien_14n_line_gives_its_neutral_nothing_and_takes_nothing():
    j = "C11H18O6"
    base = _levels(_j1_rows(j))
    assert base["[M-H]-"][0] == "3b" and base[NO3][0] == "3b"          # the branch rests on the 14N line
    note = "no 15N partner: seen in 0 of the 305 spectra of the 14N line; cluster or organonitrate undecided"
    lv = _levels(_j1_rows(j), label=K(alien={(j, NO3)}, veto={(j, NO3): note}))
    assert lv["[M-H]-"][0] == "4b" and "branch" not in lv["[M-H]-"][1] and "chan2" not in lv["[M-H]-"][1]
    assert lv[NO3][0] == "5b" and "two isotopologues refute" in lv[NO3][2] and note in lv[NO3][2]
    # an untestable line (alien, no veto) takes nothing either: its own axes decide
    lv = _levels(_j1_rows(j), label=K(alien={(j, NO3)}))
    assert lv[NO3][0] == "4b" and "branch" not in lv[NO3][1] and "chan2" not in lv[NO3][1]
    assert lv["[M-H]-"][0] == "4b"
    # a line that is not alien keeps the branch
    assert _levels(_j1_rows(j), label=K())["[M-H]-"][0] == "3b"


def test_the_two_nitrate_clusters_of_one_neutral_are_one_channel():
    x = "C4H6O4"
    rows = ledger([m0("n14", x, adduct=NO3, ion=x + "NO3", mz=C.ion_mz(x, NO3)),
                   m0("n15", x, adduct=NO3L, ion=x + "^NO3", mz=C.ion_mz(x, NO3L)),
                   child("c1", "n15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 4)])
    base = EV.level_pooled({"f": rows}).set_index("adduct")
    got = EV.level_pooled({"f": rows}, label=K()).set_index("adduct")
    assert base["chan2"].all() and not got["chan2"].any()
    assert base.loc[NO3L, "evidence_level"] == "4b" and got.loc[NO3L, "evidence_level"] == "4b"
    # with the deprotonated line the neutral still has two channels and the branch
    rows = pd.concat([rows, ledger([m0("h", x, adduct="[M-H]-", ion=x, mz=C.ion_mz(x, "[M-H]-"))])],
                     ignore_index=True)
    got = EV.level_pooled({"f": rows}, label=K()).set_index("adduct")
    assert got["chan2"].all() and got["branch"].all()
    # the fold never touches a profile without the facts: label=None is the old engine
    assert EV.level_pooled({"f": rows}, label=None).equals(EV.level_pooled({"f": rows}))


def test_the_reference_script_relabels_the_pools_like_the_engine():
    LL = _ll()
    j, x = "C11H18O6", "C4H6O4"
    rows = pd.concat([_j1_rows(j), ledger([m0("a14", x, adduct=NO3, ion=x + "NO3", mz=C.ion_mz(x, NO3)),
                                           m0("a15", x, adduct=NO3L, ion=x + "^NO3", mz=C.ion_mz(x, NO3L))])],
                     ignore_index=True)
    lab = K(alien={(j, NO3)}, veto={(j, NO3): "n"})
    core = EV.level_pooled({"s1": rows}, label=lab)
    frame = LL.measure_source("s1", rows.assign(__file="s1"), None)
    ref = LL.assign_levels(frame, set(), None, lab)
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == len(core) == 4
    assert (m["evidence_level"] == m["level"]).all()
    assert (m["chan2_x"] == m["chan2_y"]).all() and (m["branch_x"] == m["branch_y"]).all()


def test_the_facts_are_never_axes_nor_in_cross_nor_per_file():
    x = "C10H18O4"
    base = EV.level_pooled({"f": _nitrate_rows(x)})
    out = EV.level_pooled({"f": _nitrate_rows(x)}, label=K(untie={(x, NO3)}, veto={(x, NO3L): "n"}))
    assert (out["n_axes"].to_numpy() == base["n_axes"].to_numpy()).all()
    assert (out["cross"].to_numpy() == base["cross"].to_numpy()).all()
    # the cross set re-levels a source on its own per-file evidence, without rule K (§6.3)
    assert x in EV.source_neutrals({"f": _nitrate_rows(x)})
    import inspect
    assert "label" not in inspect.signature(EV.source_neutrals).parameters
    per_file = EV.compute_levels(_nitrate_rows(x))
    assert "5b" in set(per_file["evidence_level"])                # the per-file tie stands
    assert "label" not in inspect.signature(EV.compute_levels).parameters


# --------------------------------------------------------------------------- the leak guard
def test_a_leaked_fact_would_move_the_goldens():
    """Leak mutants: untie every tied [M+NO3]- pair, veto every [M+^NO3]- pair
    -- the golden vectors catch both. The facts come only from a batch time
    series of a labelled-nitrate profile, which no golden source carries."""
    from tests.test_evidence import _read
    no3, br = _read("tv_nitrate"), _read("tv_bromide")
    cross = EV.source_neutrals({"tv_bromide": br})
    base = EV.level_pooled({"tv_nitrate": no3}, cross=cross)
    tied = set(zip(base.loc[base["tied"] & base["adduct"].eq(NO3), "neutral_formula"],
                   base.loc[base["tied"] & base["adduct"].eq(NO3), "adduct"]))
    assert tied
    moved = EV.level_pooled({"tv_nitrate": no3}, cross=cross, label=K(untie=tied))
    assert _vector(moved.evidence_level) != _vector(base.evidence_level)
    tof, orbi = _pooled("tof"), _pooled("orbi")
    base = EV.level_pooled(orbi, cross=EV.source_neutrals(tof))
    assert (len(base), _vector(base.evidence_level)) == GOLDEN["orbi"]
    labelled = base[base["adduct"].eq(NO3L)]
    moved = EV.level_pooled(orbi, cross=EV.source_neutrals(tof),
                            label=K(veto={(n, NO3L): "" for n in labelled["neutral_formula"]}))
    assert _vector(moved.evidence_level) != GOLDEN["orbi"][1]
    # the one-channel fold alone (an empty fact set that is not None) moves it too
    moved = EV.level_pooled(orbi, cross=EV.source_neutrals(tof), label=K())
    assert _vector(moved.evidence_level) != GOLDEN["orbi"][1]
    # ... and alien 14N lines leaving their neutral's pools
    lines = set(zip(base.loc[base["adduct"].eq(NO3), "neutral_formula"], base.loc[base["adduct"].eq(NO3), "adduct"]))
    moved = EV.level_pooled({"tv_nitrate": no3}, cross=cross, label=K(alien=lines))
    assert _vector(moved.evidence_level) != _vector(EV.level_pooled({"tv_nitrate": no3}, cross=cross).evidence_level)


# --------------------------------------------------------------------------- the reference script
def _ll():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    return LL


def test_the_reference_script_reads_the_table_like_the_engine(tmp_path):
    LL = _ll()
    x, y = "C10H18O4", "C12H14O2"
    run = tmp_path / "out" / "RUN_1"
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    rows = pd.concat([_nitrate_rows(x), ledger([m0("y15", y, adduct=NO3L, ion=y + "^NO3",
                                                   mz=C.ion_mz(y, NO3L)),
                                               child("yc", "y15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 12)])],
                     ignore_index=True)
    rows.to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    pd.DataFrame({"neutral_formula": [x, y, x], "adduct": [NO3, NO3L, NO3L], "untie": [True, False, False],
                  "veto": [False, True, False], "note": ["", "no 14N twin", ""]}).to_csv(
        run / "tables" / "label_twins.csv", index=False)
    off = LL.run([str(run)], []).set_index(["neutral", "adduct"])["level"]
    on = LL.run([str(run)], [], None, "auto").set_index(["neutral", "adduct"])["level"]
    assert off[(x, NO3)] == "5b" and on[(x, NO3)] == "3b"
    assert off[(y, NO3L)] == "4b" and on[(y, NO3L)] == "5b"
    explicit = LL.run([str(run)], [], None, str(run / "tables" / "label_twins.csv"))
    assert list(explicit["level"]) == list(on.reset_index()["level"])
    assert list(LL.run([str(tmp_path / "out")], [], None, "auto")["level"]) == list(on.reset_index()["level"])
    core = EV.level_pooled({"s1": rows}, label=K(untie={(x, NO3)}, veto={(y, NO3L): "no 14N twin"}))
    m = core.merge(on.reset_index(), left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == len(core) and (m["evidence_level"] == m["level"]).all()
    out = tmp_path / "levels.csv"
    assert LL.main([str(run), "--label-twins", "--out", str(out)]) == 0
    got = pd.read_csv(out).set_index(["neutral", "adduct"])["level"]
    assert got[(x, NO3)] == "3b" and got[(y, NO3L)] == "5b"
    assert LL.main([str(run), "--out", str(out)]) == 0
    assert pd.read_csv(out).set_index(["neutral", "adduct"])["level"][(y, NO3L)] == "4b"
    empty = tmp_path / "bare"
    (empty / "per_file").mkdir(parents=True)
    rows.to_csv(empty / "per_file" / "s1_ledger.csv", index=False)
    assert LL.label_twin_facts(str(empty)) is None
    # one named table never reaches a source without a labelled cluster (another channel)
    other = tmp_path / "other"
    (other / "per_file").mkdir(parents=True)
    _j1_rows().to_csv(other / "per_file" / "s1_ledger.csv", index=False)
    table = tmp_path / "t.csv"
    pd.DataFrame({"neutral_formula": ["C11H18O6"], "adduct": [NO3], "untie": [False], "veto": [True],
                  "alien": [True], "note": ["n"]}).to_csv(table, index=False)
    got = LL.run([str(other)], [], None, str(table)).set_index("adduct")["level"]
    assert got["[M-H]-"] == "3b" and got[NO3] == "3b"               # untouched: no [M+^NO3]- pair there


# --------------------------------------------------------------------------- the scorecard
def test_the_scorecard_reads_the_twins_as_the_runs_own_evidence(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "scorecard", Path(__file__).resolve().parents[1] / "scripts" / "scorecard.py")
    SC = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scorecard", SC)
    spec.loader.exec_module(SC)
    seen = {}
    real = SC.EV.level_pooled

    def spy(frames, **kw):
        seen.update(kw)
        return real(frames, **kw)
    monkeypatch.setattr(SC.EV, "level_pooled", spy)
    x = "C10H18O4"
    run_dir = tmp_path / "RUN"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame({"neutral_formula": [x, x], "adduct": [NO3, NO3L], "untie": [True, False],
                  "veto": [False, True], "note": ["", "n"]}).to_csv(run_dir / "tables" / "label_twins.csv",
                                                                    index=False)
    run = types.SimpleNamespace(path=str(run_dir), per_file=_nitrate_rows(x).assign(__file="s1"))
    SC.own_levels_for(run)
    assert seen.get("label") == {"untie": {(x, NO3)}, "veto": {(x, NO3L): "n"}, "alien": set()}


# --------------------------------------------------------------------------- the batch, end to end
def test_a_labelled_nitrate_batch_writes_the_table_and_levels_it(tmp_path, monkeypatch):
    """assign_batch.run on the composed NO3+NO3_15N profile: the table is written
    from the stamped series, the summary carries the funnel, the alias-only tied
    [X+NO3]- row of a cluster that follows k_cl reads 3b on the merged ledger, and
    a labelled cluster without its 14N twin reads 5b."""
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    x, y = "C10H18O3", "C12H14O2"
    spec = _refs_spec(**{x: {"phase": 2.0}, y: {"phase": 1.0, "q": None}})
    ts = _series(spec)
    pk = ts[["sample_item_id", "datetime_utc", "mz", "height"]].copy()
    pk["sample_item_name"] = "n_" + pk["sample_item_id"]
    neutrals = list(spec)

    def fake_assign(sid, context="ambient-air", **kw):
        commits = [(n, NO3L) for n in neutrals] + [(x, NO3), (x, "[M-H]-")]
        ids = [f"P{j}" for j in range(len(commits))]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": [C.ion_mz(n, a) for n, a in commits],
                                         "height": [5000.0] * len(ids)}))
        for pid, (n, a) in zip(ids, commits):
            tie = a == NO3
            alts = ([{"formula": "C10H19NO6", "adduct": "[M-H]-", "ion_score": 0.9, "raw_score": 0.9,
                      "eff_score": 0.88, "ppm": 0.3}] if tie else None)
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=n, ion_score=0.95,
                                eff_score=0.9, tied=tie, alternatives=alts, compound_score=0.95, ppm_error=0.1,
                                pass_no=1, method="cheminfo", confidence="Medium" if tie else "High",
                                commentary=f"Pass 1: {n} {a}" + (" (TIE)" if tie else ""))
        T.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3+NO3_15N", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=lambda *a: None)
    table = pd.read_csv(tmp_path / "tables" / "label_twins.csv")
    assert list(table.columns) == list(LT.TABLE_COLUMNS)
    assert LT.untie(table) == {(x, NO3)} and set(LT.veto(table)) == {(y, NO3L)}
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["label_twins"]
    assert summ["in_scope"] is True and summ["untie"] == 1 and summ["refuted"] == 1
    merged = pd.read_csv(tmp_path / "merged_ledger.csv").set_index(["neutral_formula", "adduct"])
    assert merged.loc[(x, NO3), "evidence_level"] == "3b"
    assert "label_untie" in merged.loc[(x, NO3), "evidence_axes"]
    assert merged.loc[(y, NO3L), "evidence_level"] == "5b"
    assert "two isotopologues refute" in merged.loc[(y, NO3L), "level_reason"]
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv")
    assert {"label_untie", "label_veto", "label_note"} <= set(ev.columns)


def test_an_unlabelled_batch_writes_an_empty_table(tmp_path, monkeypatch):
    from tests.test_neutral_pairs import test_a_uronium_batch_measures_the_pair_and_lifts_the_rows as ur
    ur(tmp_path, monkeypatch)
    table = pd.read_csv(tmp_path / "tables" / "label_twins.csv")
    assert table.empty and list(table.columns) == list(LT.TABLE_COLUMNS)
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["label_twins"]
    assert summ == {"in_scope": False, "lines": 0, "untie": 0, "readings": 0, "refuted": 0}
