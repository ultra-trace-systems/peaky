"""The isotope checks of a batch (C11+a; EVIDENCE_LEVELS §3 `iso_veto`, §4 row 1,
§4.2).

`batch/iso_checks.measure` reads the stamped batch time series and the pooled
per-file ledgers and tests what each committed formula claims about its isotope
lines: rule C (the 13C line's carbon count, Orbitrap-class only), REQ (a required
heavy line absent where it must be seen) and HIGH (a heavy line too high for the
formula, variant V4). A refuted pair carries `iso_veto` with the check's note
-- a hard input of the private pre-0.10.0 decision, a step-0 rejection (5b) of
the evidence scale -- and leaves its neutral's chan2 / branch pools in both
directions. The facts exist only on the pooled batch; a leak would move the
golden fact vectors, and a test here says so. (Before peaky 0.10.0 these tests
pinned the pooled level the veto gave; that decision is private now and never
sees the table, so they pin the facts, the hard inputs, and on a batch the
scale's level on the merged ledger.) The synthetic series below carry exact isotope ratios, so every
threshold is tested at its edge.

Run: pytest tests/test_iso_checks.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import tiers as T
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from peaky.chem.resolution import Resolution
from tests.test_evidence import ORBI_NO_LOCK, _pooled, _vector, child, ledger, m0

N = 40
FILL = 10.0
T0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
ORBI = {"coef": 5e-7, "exponent": 1.5}            # R ~ 140 000 at m/z 200
TOF = Resolution.from_r(10_000)
SCALE_O = {"sigma_ppm": 0.2, "stamp_ppm": 6.0}
SCALE_T = {"sigma_ppm": 3.6, "stamp_ppm": 9.074}
NO3 = P.resolve("NO3")
LABEL = P.resolve("NO3+NO3_15N")
REFS = ["C10H16O4", "C9H14O4", "C8H12O4", "C7H10O5", "C10H16O5", "C9H14O5"]
H = "[M-H]-"
D81 = 1.9979521


def quiet(*a):
    pass


# --------------------------------------------------------------------------- builders
def _ion(n, a):
    return C.format_formula(T._ion_counts(n, a)) + ("-" if a.endswith("-") else "+")


def _row(i, mz, h, a=None, role="", nf="", ad="", label="", ion=""):
    return dict(sample_item_id=f"s{i:03d}", datetime_utc=T0 + pd.Timedelta(minutes=20 * i), mz=float(mz),
                height=float(h), area=float(h if a is None else a), role=role, neutral_formula=nf, adduct=ad,
                iso_label=label, ion_formula=ion)


def _filler(i, low=True):
    """The spectrum's noise: sixty peaks at FILL cps (its noise edge), one at m/z 60 (the scan start)."""
    rows = [_row(i, 700.0 + 0.37 * j, FILL) for j in range(60)]
    return rows + ([_row(i, 60.0, FILL)] if low else [])


def _wave(i, phase=0.0):
    return 1.5 + np.sin(2 * np.pi * i / 13.0 + phase)


def _m0(i, n, a, h, stamp_shift=0.0, role="M0"):
    return _row(i, C.ion_mz(n, a) + stamp_shift, h, role=role, nf=n if role else "", ad=a if role else "",
                ion=_ion(n, a) if role else "")


def _c13(i, n, a, *, c=None, c_h=None, h=1e5, phase=0.0, occupied=False, label="13C"):
    """An M0 stamp and its 13C line reading `c` carbons on area and `c_h` on height."""
    ion = _ion(n, a)
    ic = C.parse_formula(ion)
    mz = C.ion_mz(n, a)
    oth = ic.get("O", 0) * IC.R17O
    ca = ic["C"] if c is None else c
    ch = ca if c_h is None else c_h
    h0 = h * _wave(i, phase)
    wf = ((mz + 1) / mz) ** 1.5
    h1, a1 = h0 * (ch * IC.R13C + oth), h0 * (ca * IC.R13C + oth) * wf
    line = (_row(i, mz + IC.D13C, h1, a1, role="M0", nf="C9H9NO", ad=H) if occupied
            else _row(i, mz + IC.D13C, h1, a1, role="iso_child", label=label))
    return [_row(i, mz, h0, role="M0", nf=n, ad=a, ion=ion), line]


def _series(build, *, low=True) -> pd.DataFrame:
    rows = []
    for i in range(N):
        rows += _filler(i, low=low)
        rows += build(i)
    return pd.DataFrame(rows)


def _refs(i):
    rows = []
    for k, n in enumerate(REFS):
        rows += _c13(i, n, H, phase=0.7 * k)
    return rows


def _frames(pairs, **kw) -> dict:
    rows = [m0(f"p{k}", n, adduct=a, ion=_ion(n, a), mz=C.ion_mz(n, a), **kw) for k, (n, a) in enumerate(pairs)]
    return {"f1": EV.trim(ledger(rows))}


def _measure(ts, pairs, prof=NO3, resolution=ORBI, scale=SCALE_O, x_edge=1.0, frames=None):
    return IC.measure(ts, frames if frames is not None else _frames(pairs), prof, resolution=resolution,
                      mass_scale=scale, x_edge=x_edge, log=quiet)


def _get(table, check, n, a=H) -> pd.Series:
    t = table[(table["check"] == check) & (table["neutral_formula"] == n) & (table["adduct"] == a)]
    assert len(t) == 1, (check, n, a, len(t))
    return t.iloc[0]


def _has(table, check, n, a=H) -> bool:
    return bool(((table["check"] == check) & (table["neutral_formula"] == n) & (table["adduct"] == a)).any())


# --------------------------------------------------------------------------- the instrument gate
def test_the_instrument_class_reads_the_width_model():
    assert IC.instrument_class(None) == "tof"                          # no model: conservative
    assert IC.instrument_class(ORBI) == "orbitrap"
    assert IC.instrument_class(TOF) == "tof"
    assert IC.instrument_class(Resolution.from_r(50_000)) == "orbitrap"
    assert IC.instrument_class(Resolution.from_r(49_999)) == "tof"
    assert IC.instrument_class({"coef": None}) == "tof"


def test_a_tof_class_batch_runs_no_rule_c():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]
    ts = _series(lambda i: _refs(i) + _c13(i, x, H, c=6.0))
    assert _get(_measure(ts, pairs), "C", x)["veto"]
    for res in (TOF, None):
        t = _measure(ts, pairs, resolution=res, scale=SCALE_T)
        assert not (t["check"] == "C").any() and (t["check"] == "HIGH").any()
        assert set(t["instrument"]) == {"tof"}


# --------------------------------------------------------------------------- rule C
def test_rule_c_contradicts_a_carbon_count_the_13c_line_refutes():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]
    t = _measure(_series(lambda i: _refs(i) + _c13(i, x, H, c=6.0)), pairs)
    r = _get(t, "C", x)
    assert r["verdict"] == "contradict" and r["veto"]
    assert r["n_carbon"] == 12 and r["c_area"] == pytest.approx(6.0, abs=1e-6)
    assert r["c_height"] == pytest.approx(6.0, abs=1e-6) and r["n_used"] == N
    assert r["note"].startswith("13C reads 6.0 C for 12 (height 6.0")
    for n in REFS:
        assert _get(t, "C", n)["verdict"] == "agree"
    assert r["bias_area"] == pytest.approx(0.0, abs=1e-9)          # the six references hold the median
    assert IC.veto(t) == {(x, H): "rule C: " + r["note"]}


def test_rule_c_tolerance_is_max_of_1_5_a_quarter_and_3_se():
    x = "C12H20O4"                                                  # tol = 0.25 x 12 = 3.0
    pairs = [(n, H) for n in REFS] + [(x, H)]
    for c, verdict in ((9.1, "agree"), (8.9, "contradict"), (14.9, "agree"), (15.1, "contradict")):
        t = _measure(_series(lambda i: _refs(i) + _c13(i, x, H, c=c)), pairs)
        assert _get(t, "C", x)["verdict"] == verdict, c
    y = "C4H6O4"                                                    # tol = 1.5 (floor) for 4 carbons
    pairs = [(n, H) for n in REFS] + [(y, H)]
    for c, verdict in ((2.6, "agree"), (2.4, "contradict")):
        t = _measure(_series(lambda i: _refs(i) + _c13(i, y, H, c=c)), pairs)
        assert _get(t, "C", y)["verdict"] == verdict, c


def test_rule_c_needs_area_and_height_to_miss():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]
    t = _measure(_series(lambda i: _refs(i) + _c13(i, x, H, c=6.0, c_h=12.0)), pairs)
    r = _get(t, "C", x)
    assert r["verdict"] == "ambiguous" and not r["veto"]
    t = _measure(_series(lambda i: _refs(i) + _c13(i, x, H, c=12.0, c_h=6.0)), pairs)
    assert _get(t, "C", x)["verdict"] == "agree"


def test_rule_c_needs_eight_bright_spectra():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]

    def build(bright):
        # a dim M0 (20 cps: 20 x 12 x R13C < 5 x the noise edge) is not used
        return lambda i: _refs(i) + _c13(i, x, H, c=6.0, h=1e5 if i < bright else 20.0 / _wave(i))
    r = _get(_measure(_series(build(8)), pairs), "C", x)
    assert r["n_used"] == 8 and r["verdict"] == "contradict"
    r = _get(_measure(_series(build(7)), pairs), "C", x)
    assert r["n_used"] == 7 and r["verdict"] == "untestable" and not r["veto"]
    assert "7 of 40 spectra bright enough" in r["note"]


def test_rule_c_bias_is_level_free_over_heteroatom_free_pairs():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]
    # every reference reads 10 % high: the batch's misread, divided out -- the pair agrees
    ts = _series(lambda i: sum((_c13(i, n, H, c=1.1 * C.parse_formula(n)["C"], phase=0.7 * k)
                                for k, n in enumerate(REFS)), []) + _c13(i, x, H, c=1.1 * 12))
    t = _measure(ts, pairs)
    r = _get(t, "C", x)
    assert r["bias_area"] == pytest.approx(0.1, abs=1e-6) and r["c_area"] == pytest.approx(12.0, abs=1e-6)
    assert r["verdict"] == "agree"
    # a heteroatom pair does not set the bias: five sulfur pairs reading half their carbons move nothing
    s_pairs = ["C10H16O4S", "C9H14O4S", "C8H12O4S", "C7H10O5S", "C10H16O5S", "C9H14O5S", "C11H18O4S"]
    ts = _series(lambda i: _refs(i) + _c13(i, x, H, c=6.0)
                 + sum((_c13(i, n, H, c=0.5 * C.parse_formula(n)["C"], phase=0.3 * k) for k, n in enumerate(s_pairs)), []))
    t = _measure(ts, pairs + [(n, H) for n in s_pairs])
    assert _get(t, "C", x)["bias_area"] == pytest.approx(0.0, abs=1e-9)
    assert _get(t, "C", x)["verdict"] == "contradict"


def test_rule_c_too_many_with_an_occupied_slot_is_untestable():
    y = "C8H14O4"                                                   # 8 C, reads 14: 'too many'
    pairs = [(n, H) for n in REFS] + [(y, H)]
    for k_occ, verdict in ((12, "untestable"), (8, "contradict")):  # 30 % / 20 % of the used spectra
        t = _measure(_series(lambda i: _refs(i) + _c13(i, y, H, c=14.0, occupied=i < k_occ)), pairs)
        r = _get(t, "C", y)
        assert r["verdict"] == verdict and r["occupied"] == pytest.approx(k_occ / N)
    assert "another pair's line in 30%" in _get(
        _measure(_series(lambda i: _refs(i) + _c13(i, y, H, c=14.0, occupied=i < 12)), pairs), "C", y)["note"]
    # 'too few' is never excused by an occupied slot
    t = _measure(_series(lambda i: _refs(i) + _c13(i, y, H, c=4.0, occupied=i < 12)), pairs)
    assert _get(t, "C", y)["verdict"] == "contradict"


def test_rule_c_exempts_a_14n_cluster_beside_its_15n_sibling():
    x, a = "C10H18O4", "[M+NO3]-"
    pairs = [(n, H) for n in REFS] + [(x, a)]
    ts = _series(lambda i: _refs(i) + _c13(i, x, a, c=5.0))
    r = _get(_measure(ts, pairs, prof=LABEL), "C", x, a)
    assert r["verdict"] == "exempt_14N" and not r["veto"] and "15N sibling" in r["note"]
    r = _get(_measure(ts, pairs, prof=NO3), "C", x, a)           # no labelled cluster: tested
    assert r["verdict"] == "contradict" and r["veto"]
    r = _get(_measure(ts, pairs, prof=None), "C", x, a)
    assert r["verdict"] == "contradict"


def test_rule_c_skips_an_ion_below_the_scan_start_plus_one():
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H)]
    # no peak below the lowest reference: the scan starts at C8H12O4 [M-H]- (m/z 171.07)
    ts = _series(lambda i: _refs(i) + _c13(i, x, H, c=6.0), low=False)
    t = _measure(ts, pairs)
    low = _get(t, "C", "C8H12O4")
    assert low["verdict"] == "scan_edge" and not low["veto"]
    assert low["note"].endswith("below the scan start + 1 Da, not tested")
    assert _get(t, "C", "C7H10O5")["verdict"] == "agree"             # m/z 173.05: above the edge
    assert _get(t, "C", x)["verdict"] == "contradict"


def test_rule_c_never_tests_an_ion_only_pair():
    x = "C10H16O5"
    frame = ledger([m0(f"p{k}", n, adduct=H, ion=_ion(n, H), mz=C.ion_mz(n, H)) for k, n in enumerate(REFS)]
                   + [m0("io", x, adduct="[M]-.", ion="C10H16O5-", mz=C.ion_mz(x, H) + 1.00728,
                         method="ion_only:ea")])
    ts = _series(lambda i: _refs(i) + [_row(i, C.ion_mz(x, H) + 1.00728, 1e5, role="M0", nf=x, ad="[M]-.",
                                            ion="C10H16O5-")])
    t = IC.measure(ts, {"f": EV.trim(frame)}, NO3, resolution=ORBI, mass_scale=SCALE_O, log=quiet)
    assert not _has(t, "C", x, "[M]-.") and _has(t, "C", REFS[0])


# --------------------------------------------------------------------------- REQ
Y = "C6H9BrO3"


def _br(i, *, h=1e4, present=lambda i: True, ppm=0.0, n=Y, stamped=True):
    """A bromine ion's M0 stamp (`stamped` False: an unstamped peak), its 13C line
    and, where `present(i)`, its 81Br line `ppm` off its exact position."""
    rows = _c13(i, n, H, h=h)
    if not stamped:
        rows[0].update(role="", neutral_formula="", adduct="", ion_formula="")
    mz, h0 = rows[0]["mz"], rows[0]["height"]
    if present(i):
        rows.append(_row(i, (mz + D81) * (1 + ppm * 1e-6), 0.9728 * h0, role="iso_child", label="81Br"))
    return rows


def test_req_refutes_a_bromine_formula_without_its_81br_line():
    t = _measure(_series(lambda i: _br(i)), [(Y, H)])
    r = _get(t, "REQ", Y)
    assert r["verdict"] == "present" and not r["veto"] and r["line"] == "M+2 (81Br)"
    assert r["expected"] == pytest.approx(0.9728, abs=0.01) and r["n_used"] == N and r["n_present"] == N
    t = _measure(_series(lambda i: _br(i, present=lambda i: False)), [(Y, H)])
    r = _get(t, "REQ", Y)
    assert r["verdict"] == "absent" and r["veto"] and r["det_frac"] == 0.0
    assert r["note"] == ("the M+2 (81Br) line (0.97x the stamped line) absent in 40 of 40 detectable spectra "
                         "(within 1 ppm)")
    assert IC.veto(t) == {(Y, H): "REQ: " + r["note"]}


def test_req_absent_is_at_most_a_fifth_of_three_or_more_detectable_spectra():
    def build(bright, seen):
        # a dim M0 (12 cps: its 81Br line would be below 3 x the noise edge) is not detectable
        return lambda i: _br(i, h=1e4 if i < bright else 12.0 / _wave(i), present=lambda i: i < seen)
    for bright, seen, verdict in ((10, 2, "absent"), (10, 3, "present"), (3, 0, "absent"), (2, 0, "untestable")):
        r = _get(_measure(_series(build(bright, seen)), [(Y, H)]), "REQ", Y)
        assert r["verdict"] == verdict and r["n_used"] == bright, (bright, seen)
        assert r["veto"] == (verdict == "absent")


def test_req_floor_is_the_height_gate_of_the_batch():
    # a 60-cps M0: its 58-cps 81Br line is detectable over 3 x 10 cps, not over 3 x 5 x 10 cps
    build = _series(lambda i: _br(i, h=60.0 / _wave(i), present=lambda i: False))
    assert _get(_measure(build, [(Y, H)], x_edge=1.0), "REQ", Y)["verdict"] == "absent"
    assert _get(_measure(build, [(Y, H)], x_edge=5.0), "REQ", Y)["verdict"] == "untestable"
    assert _get(_measure(build, [(Y, H)], x_edge="auto"), "REQ", Y)["verdict"] == "absent"


def test_req_window_is_one_ppm_on_an_orbitrap_and_two_windows_on_a_tof():
    near, off = _series(lambda i: _br(i, ppm=0.8)), _series(lambda i: _br(i, ppm=1.5))
    assert _get(_measure(near, [(Y, H)]), "REQ", Y)["verdict"] == "present"
    r = _get(_measure(off, [(Y, H)]), "REQ", Y)
    assert r["verdict"] == "absent" and r["window_ppm"] == 1.0
    # 4 sigma widens it
    assert _get(_measure(off, [(Y, H)], scale={"sigma_ppm": 0.4, "stamp_ppm": 6.0}), "REQ", Y)["verdict"] == "present"
    # the TOF vetoes only when the line is absent at the stamp window AND at 20 ppm
    for ppm, verdict in ((5.0, "present"), (15.0, "present"), (25.0, "absent")):
        r = _get(_measure(_series(lambda i: _br(i, ppm=ppm)), [(Y, H)], resolution=TOF, scale=SCALE_T), "REQ", Y)
        assert r["verdict"] == verdict, ppm
    r = _get(_measure(_series(lambda i: _br(i, ppm=15.0)), [(Y, H)], resolution=TOF, scale=SCALE_T), "REQ", Y)
    assert r["n_present"] == 0 and r["n_present_wide"] == N and r["window_ppm"] == 9.074
    r = _get(_measure(_series(lambda i: _br(i, ppm=25.0)), [(Y, H)], resolution=TOF, scale=SCALE_T), "REQ", Y)
    assert "within 9.07 and 20 ppm" in r["note"]


def test_req_places_the_lines_on_the_stamped_isotopologue():
    """A Br2 ion is committed on its 79Br81Br line: the all-light and the 81Br2
    lines are both required, each at ~0.5x the stamped line."""
    z = "C2H2Br2O2"
    mz = C.ion_mz(z, H)

    def build(light=True, heavy=True):
        def f(i):
            h = 1e4 * _wave(i)
            rows = [_row(i, mz + D81, h, role="M0", nf=z, ad=H, ion=_ion(z, H))]
            if light:
                rows.append(_row(i, mz, 0.514 * h))
            if heavy:
                rows.append(_row(i, mz + 2 * D81, 0.486 * h))
            return rows
        return f
    t = _measure(_series(build()), [(z, H)])
    assert _get(t, "REQ", z)["verdict"] == "present"
    r = _get(_measure(_series(build(light=False)), [(z, H)]), "REQ", z)
    assert r["verdict"] == "absent" and r["line"] == "M0 (all-light)" and r["expected"] == pytest.approx(0.514, abs=0.01)
    r = _get(_measure(_series(build(heavy=False)), [(z, H)]), "REQ", z)
    assert r["verdict"] == "absent" and r["line"] == "M+4 (2x81Br)"
    both = _get(_measure(_series(build(light=False, heavy=False)), [(z, H)]), "REQ", z)
    assert "M0 (all-light)" in both["note"] and "M+4 (2x81Br)" in both["note"]   # every absent line named


def test_req_tests_sulfur_and_silicon_on_an_orbitrap_only():
    s = "C6H10O4S"
    ts = _series(lambda i: [_m0(i, s, H, 1e5 * _wave(i))])
    r = _get(_measure(ts, [(s, H)]), "REQ", s)
    assert r["verdict"] == "absent" and r["line"] == "34S" and r["expected"] == pytest.approx(0.0447, abs=1e-3)
    assert not _has(_measure(ts, [(s, H)], resolution=TOF, scale=SCALE_T), "REQ", s)
    # a sulfur-free, halogen-free ion has nothing to require
    assert not _has(_measure(_series(lambda i: _refs(i)), [(n, H) for n in REFS]), "REQ", REFS[0])


def test_req_needs_the_width_model():
    assert not _has(_measure(_series(lambda i: _br(i, present=lambda i: False)), [(Y, H)], resolution=None),
                    "REQ", Y)


def test_req_reads_an_unstamped_pair_at_its_pooled_mass():
    t = _measure(_series(lambda i: _br(i, stamped=False, present=lambda i: False)), [(Y, H)])
    r = _get(t, "REQ", Y)
    assert not r["stamped"] and r["verdict"] == "absent" and r["n_spectra"] == N


def test_req_reads_the_ion_from_neutral_and_adduct_when_the_ledger_holds_the_neutral():
    """A ledger row can store the NEUTRAL as its ion formula: the reagent's Br is
    still the ion's (the ion is neutral + adduct when the stored one has no charge)."""
    x = "C6H10O3"
    a = "[M+Br]-"
    mz = C.ion_mz(x, a)
    ts = _series(lambda i: [_row(i, mz, 1e4 * _wave(i), role="M0", nf=x, ad=a, ion=x)])
    frames = {"f": EV.trim(ledger([m0("p", x, adduct=a, ion=x, mz=mz)]))}
    r = _get(_measure(ts, [], frames=frames), "REQ", x, a)
    assert r["verdict"] == "absent" and r["line"] == "M+2 (81Br)"
    assert IC.ion_counts(x, a, x) == {"C": 6, "H": 10, "O": 3, "Br": 1}
    assert IC.ion_counts(x, a, "C6H10O3Br-") == {"C": 6, "H": 10, "O": 3, "Br": 1}


# --------------------------------------------------------------------------- HIGH
# a Br-free formula whose [M-H]- (m/z 241.051) an ion carrying one Br can also have (a CHNOS rest fits
# within 1 ppm): HIGH names an element only where an ion carrying it fits the mass (2026-09-27)
Z = "C14H10O4"


def _hi(i, *, n=Z, ratio=1.0, ratio_h=None, offset=1.9979535, seen=None, stamp=None, h=1e4, anti=False, n_m0=N):
    """An M0 stamp and a line `offset` Da above it at `ratio` x its area (and
    `ratio_h` x its height, default the same; `anti`: running against it);
    `stamp(i)` -> (role, neutral, iso_label) of the line."""
    if i >= n_m0:
        return []
    mz = C.ion_mz(n, H)
    h0 = h * _wave(i)
    rows = [_row(i, mz, h0, role="M0", nf=n, ad=H, ion=_ion(n, H))]
    if seen is None or seen(i):
        role, nf, label = stamp(i) if stamp else ("", "", "")
        hl = h * h * 2.25 / h0 if anti else h0
        rows.append(_row(i, mz + offset, (ratio if ratio_h is None else ratio_h) * hl, ratio * hl, role=role, nf=nf,
                         ad=H if nf else "", label=label))
    return rows


def test_high_refutes_a_line_the_formula_cannot_make():
    t = _measure(_series(lambda i: _hi(i)), [(Z, H)])
    r = _get(t, "HIGH", Z)
    assert r["verdict"] == "too_high" and r["veto"] and r["offset"] == "81Br"
    assert r["ratio_area"] == pytest.approx(1.0) and r["ratio_height"] == pytest.approx(1.0)
    assert r["r"] == pytest.approx(1.0) and r["presence"] == 1.0
    exp = IC.expected_m2(C.parse_formula(_ion(Z, H)))
    assert r["expected"] == pytest.approx(exp)
    assert r["note"] == (f"a 1.00x line at the 81Br offset ({1 / exp:.0f}x the formula's M+2 {exp:.3f}), r 1.00, "
                         f"in 100% of 40 spectra: the ion carries Br the formula lacks")


def test_high_each_clause_can_fail():
    assert _get(_measure(_series(lambda i: _hi(i, anti=True)), [(Z, H)]), "HIGH", Z)["verdict"] == "consistent"
    for k, verdict in ((24, "too_high"), (23, "consistent")):             # present in >= 60 % of the spectra
        r = _get(_measure(_series(lambda i: _hi(i, seen=lambda i: i < k)), [(Z, H)]), "HIGH", Z)
        assert r["verdict"] == verdict, k
    for ratio, verdict in ((0.21, "too_high"), (0.19, "consistent")):     # the 0.2 floor
        assert _get(_measure(_series(lambda i: _hi(i, ratio=ratio)), [(Z, H)]), "HIGH", Z)["verdict"] == verdict
    # >= 3x the ion's count-aware M+2: a bromide ion's own 81Br line is expected
    exp = IC.expected_m2(C.parse_formula(_ion(Y, H)))
    for x, verdict in ((2.9, "consistent"), (3.05, "too_high")):
        r = _get(_measure(_series(lambda i: _hi(i, n=Y, ratio=x * exp)), [(Y, H)]), "HIGH", Y)
        assert r["verdict"] == verdict, x
    assert _get(_measure(_series(lambda i: _hi(i, n=Y, ratio=3.05 * exp)), [(Y, H)]), "HIGH", Y)["note"].endswith(
        "more than the formula can make")
    # ... on the area AND the height
    for ra, rh in ((3.05, 2.0), (2.0, 3.05)):
        r = _get(_measure(_series(lambda i: _hi(i, n=Y, ratio=ra * exp, ratio_h=rh * exp)), [(Y, H)]), "HIGH", Y)
        assert r["verdict"] == "consistent" and r["ratio_area"] == pytest.approx(ra * exp), (ra, rh)
    # >= 8 spectra of the M0, or no row at all
    assert _has(_measure(_series(lambda i: _hi(i, n_m0=8)), [(Z, H)]), "HIGH", Z)
    assert not _has(_measure(_series(lambda i: _hi(i, n_m0=7)), [(Z, H)]), "HIGH", Z)


def test_high_slot_guards_an_other_m0_and_a_plain_13c_line():
    for k, verdict in ((12, "guarded"), (8, "too_high")):                  # > 20 % of the line's spectra
        r = _get(_measure(_series(lambda i: _hi(i, stamp=lambda i: ("M0", "C9H9NO", "") if i < k else ("", "", ""))),
                          [(Z, H)]), "HIGH", Z)
        assert r["verdict"] == verdict and r["other_m0"] == pytest.approx(k / N)
        r = _get(_measure(_series(lambda i: _hi(i, stamp=lambda i: ("iso_child", "", "13C") if i < k else ("", "", ""))),
                          [(Z, H)]), "HIGH", Z)
        assert r["verdict"] == verdict and r["other_13c"] == pytest.approx(k / N)
    r = _get(_measure(_series(lambda i: _hi(i, stamp=lambda i: ("M0", "C9H9NO", ""))), [(Z, H)]), "HIGH", Z)
    assert not r["veto"] and "another committed M0 in 100%" in r["note"]
    # the line stamped M0 of the SAME pair is no other M0; a '13C2' line is the parent's own, never a slot
    r = _get(_measure(_series(lambda i: _hi(i, stamp=lambda i: ("M0", Z, ""))), [(Z, H)]), "HIGH", Z)
    assert r["other_m0"] == 0.0 and r["verdict"] == "too_high"
    r = _get(_measure(_series(lambda i: _hi(i, stamp=lambda i: ("iso_child", "", "13C2"))), [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "too_high"


def test_high_caps_what_no_isotope_envelope_reaches():
    # at the 18O spacing (no element to fit): above expected + 3 no isotope envelope reaches
    o18 = IC.HIGH_OFFSETS["18O"]
    r = _get(_measure(_series(lambda i: _hi(i, ratio=3.5, offset=o18)), [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "guarded" and "no isotope envelope" in r["note"]
    r = _get(_measure(_series(lambda i: _hi(i, ratio=2.9, offset=o18)), [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "too_high" and r["offset"] == "18O"
    # at the 81Br spacing 2.9x means three Br, which no ion at m/z 241 carries: not this ion's isotope line
    r = _get(_measure(_series(lambda i: _hi(i, ratio=2.9)), [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "guarded" and "no ion carrying the Br this line needs fits its mass" in r["note"]


def test_high_alias_guard_is_orbitrap_only():
    # a line at the F<->OH spacing, 0.14 mDa below 34S: an Orbitrap tells it from the heavy line, a TOF cannot
    ts = _series(lambda i: _hi(i, offset=IC.HIGH_ALIASES["F<->OH"]))
    r = _get(_measure(ts, [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "guarded" and r["offset"] == "34S" and "non-isotopic +2 alias" in r["note"]
    r = _get(_measure(ts, [(Z, H)], resolution=TOF, scale=SCALE_T), "HIGH", Z)
    assert r["verdict"] == "too_high" and r["instrument"] == "tof"
    # the partner window: 1 ppm on the Orbitrap, 10 on the TOF
    ts = _series(lambda i: _hi(i, offset=1.9979535 + 5e-6 * C.ion_mz(Z, H)))
    assert _get(_measure(ts, [(Z, H)]), "HIGH", Z)["verdict"] == "consistent"
    assert _get(_measure(ts, [(Z, H)], resolution=TOF, scale=SCALE_T), "HIGH", Z)["verdict"] == "too_high"


# --------------------------------------------------------------------------- the table, the facts
def test_the_table_and_its_facts():
    assert IC.measure(None, _frames([(Z, H)]), NO3, resolution=ORBI, log=quiet).empty
    assert list(IC.measure(None, {}, NO3, log=quiet).columns) == list(IC.TABLE_COLUMNS)
    assert list(_measure(pd.DataFrame(columns=["sample_item_id", "mz", "height"]), [(Z, H)]).columns) == \
        list(IC.TABLE_COLUMNS)
    assert IC.facts(None) is None and IC.facts(IC._empty()) is None
    x = "C12H20O4"
    pairs = [(n, H) for n in REFS] + [(x, H), (Y, H), (Z, H)]
    # X reads half its carbons; Y has no 81Br line; Z no 13C line and a 1:1 line at the 81Br offset
    ts = _series(lambda i: _refs(i) + _c13(i, x, H, c=6.0) + _br(i, present=lambda i: False) + _hi(i))
    t = _measure(ts, pairs)
    assert list(t.columns) == list(IC.TABLE_COLUMNS)
    f = IC.facts(t)
    assert set(f) == {"veto", "lock"} and set(f["veto"]) == {(x, H), (Y, H), (Z, H)} and f["lock"] == {}
    assert f["veto"][(x, H)] == "rule C: " + _get(t, "C", x)["note"]
    assert f["veto"][(Y, H)] == "REQ: " + _get(t, "REQ", Y)["note"]
    # a pair refuted by several checks carries one joined note, in check order
    assert f["veto"][(Z, H)] == "rule C: " + _get(t, "C", Z)["note"] + "; HIGH: " + _get(t, "HIGH", Z)["note"]
    assert _get(t, "C", Z)["note"].startswith("13C reads -0.1 C for 14")
    assert IC.veto(t.iloc[::-1]) == f["veto"]
    s = IC.summary(t, ORBI)
    assert s["instrument"] == "orbitrap" and s["vetoed_pairs"] == 3 and s["tested"] == len(t)
    assert s["C"]["vetoed"] == 2 and s["C"]["contradict"] == 2 and s["C"]["agree"] == len(REFS) + 1
    assert s["REQ"]["tested"] == 1 and s["REQ"]["absent"] == 1 and s["HIGH"]["too_high"] == 1
    assert IC.summary(None, None) == {"instrument": "tof", "tested": 0, "vetoed_pairs": 0, "locked_pairs": 0}


# --------------------------------------------------------------------------- the level
def _branch_rows(x="C10H18O4", p=""):
    """A neutral seen deprotonated (13C-backed) and clustered: the acid branch
    (`p` prefixes the peak ids, so two neutrals can share a file)."""
    nc = C.parse_formula(x)["C"]
    return ledger([m0(p + "h", x, adduct=H, ion=_ion(x, H), mz=C.ion_mz(x, H)),
                   child(p + "hc", p + "h", "13C+1", 1000.0 * EV.C13_PER_CARBON * nc),
                   m0(p + "n", x, adduct="[M+NO3]-", ion=_ion(x, "[M+NO3]-"), mz=C.ion_mz(x, "[M+NO3]-")),
                   child(p + "nc", p + "n", "13C+1", 1000.0 * EV.C13_PER_CARBON * nc)])


HARD = ("tied", "below", "lead", "lowconf", "label_veto", "iso_veto")


def _levels(frame, **kw) -> dict:
    """{adduct: (hard inputs, facts string, iso_note)} of the pooled fact layer over one file."""
    out = EV._series_pooled({"f": frame}, **kw)
    return {r.adduct: (tuple(h for h in HARD if bool(getattr(r, h))), r.evidence_axes, r.iso_note)
            for r in out.itertuples(index=False)}


def test_a_vetoed_pair_is_hard_with_the_checks_note():
    x = "C10H18O4"
    base = _levels(_branch_rows(x))
    assert base[H][:2] == ((), "iso|chan2|carbon|branch|files:1") and base["[M+NO3]-"][0] == ()
    note = "rule C: 13C reads 4.1 C for 10"
    lv = _levels(_branch_rows(x), iso={"veto": {(x, "[M+NO3]-"): note}})
    hard, axes, why = lv["[M+NO3]-"]
    assert hard == ("iso_veto",) and "iso_veto" in axes and why == note
    assert _levels(_branch_rows(x), iso={"veto": {(x, "[M+NO3]-"): ""}})["[M+NO3]-"][::2] == (("iso_veto",), "")
    # it stands beside a curated identity, as every hard input does
    rows = _branch_rows(x)
    rows.loc[rows.peak_id == "n", "method"] = "known:atmospheric"
    assert _levels(rows, iso={"veto": {(x, "[M+NO3]-"): note}})["[M+NO3]-"][0] == ("iso_veto",)
    # with another hard input both hold
    rows = _branch_rows(x)
    rows.loc[rows.peak_id == "n", "tied"] = True
    assert _levels(rows, iso={"veto": {(x, "[M+NO3]-"): note}})["[M+NO3]-"][0] == ("tied", "iso_veto")


def test_a_vetoed_pair_leaves_its_neutrals_pools_both_ways():
    x = "C10H18O4"
    lv = _levels(_branch_rows(x), iso={"veto": {(x, "[M+NO3]-"): "n"}})
    assert lv[H][0] == () and "branch" not in lv[H][1] and "chan2" not in lv[H][1]     # gives the acid nothing
    assert "branch" not in lv["[M+NO3]-"][1] and "chan2" not in lv["[M+NO3]-"][1]       # takes nothing
    lv = _levels(_branch_rows(x), iso={"veto": {(x, H): "n"}})
    assert lv["[M+NO3]-"][:2] == ((), "iso|carbon|files:1") and lv[H][0] == ("iso_veto",)
    # a veto keyed on another neutral leaves this one alone
    assert _levels(_branch_rows(x), iso={"veto": {("C9H9NO", H): "n"}}) == _levels(_branch_rows(x))


def test_iso_none_and_an_empty_veto_change_nothing():
    rows = _branch_rows()
    base = EV._series_pooled({"f": rows})
    assert base.equals(EV._series_pooled({"f": rows}, iso=None))
    assert base.equals(EV._series_pooled({"f": rows}, iso={"veto": {}}))
    assert not base["iso_veto"].any() and (base["iso_note"] == "").all()


def test_the_veto_reaches_an_ion_only_pair():
    x = "C10H16O5"
    rows = ledger([m0("h", x, adduct=H, ion=_ion(x, H), mz=C.ion_mz(x, H)),
                   m0("io", x, adduct="[M]-.", ion="C10H16O5-", mz=C.ion_mz(x, H) + 1.00728, method="ion_only:ea"),
                   child("ic", "io", "13C+1", 1000.0 * EV.C13_PER_CARBON * 10)])
    assert _levels(rows)["[M]-."][:2] == ((), "iso|carbon|ion_only|files:1")
    lv = _levels(rows, iso={"veto": {(x, "[M]-."): "HIGH: h"}})
    assert lv["[M]-."][0] == ("iso_veto",) and "iso_veto" in lv["[M]-."][1]


def test_the_facts_are_never_axes_nor_in_cross_nor_per_file():
    import inspect
    x = "C10H18O4"
    base = EV._series_pooled({"f": _branch_rows(x)}, cross={x})
    out = EV._series_pooled({"f": _branch_rows(x)}, cross={x}, iso={"veto": {(x, H): "n"}})
    assert (out["corroborated"] == base["corroborated"]).all() and (out["cross"] == base["cross"]).all()
    assert (out["iso"] == base["iso"]).all() and (out["anchor"] == base["anchor"]).all()
    assert "iso" not in inspect.signature(EV._source_neutrals).parameters
    assert "iso" not in inspect.signature(EV.vote_classes).parameters
    assert "iso_veto" not in EV._AXES


# --------------------------------------------------------------------------- the leak guard
def test_a_leaked_veto_would_move_the_goldens():
    """Leak mutant: the isotope facts on a golden set move its fact vector -- the
    facts come only from a batch time series, which no golden source carries."""
    tof, orbi = _pooled("tof"), _pooled("orbi")
    cross = EV._source_neutrals(tof)
    base = EV._series_pooled(orbi, cross=cross)
    assert (len(base), _vector(base)) == ORBI_NO_LOCK
    assert _vector(EV._series_pooled(orbi, cross=cross, iso={"veto": {}})) == ORBI_NO_LOCK[1]
    acids = base[base["adduct"].eq(H)]
    moved = EV._series_pooled(orbi, cross=cross, iso={"veto": {(n, H): "" for n in acids["neutral_formula"]}})
    assert _vector(moved) != ORBI_NO_LOCK[1]
    # one veto on a branch acid takes the branch from its cluster sibling through the pool alone
    sib = set(base.loc[base["adduct"].ne(H) & base["branch"], "neutral_formula"])
    br = base[base["branch"] & base["adduct"].eq(H) & base["neutral_formula"].isin(sib)]
    n = br["neutral_formula"].iloc[0]
    moved = EV._series_pooled(orbi, cross=cross, iso={"veto": {(n, H): ""}})
    after = moved[(moved.neutral_formula == n) & moved.adduct.ne(H)]
    assert len(after) and not after["branch"].any() and not after["iso_veto"].any()


# --------------------------------------------------------------------------- the reference script
def _ll():
    """The reference script with the pre-0.10.0 decision, or skip."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    return LL


def test_the_reference_script_levels_the_vetoes_like_the_engine():
    LL = _ll()
    x, y = "C10H18O4", "C9H14O4"
    rows = pd.concat([_branch_rows(x), _branch_rows(y, "y")], ignore_index=True)
    for iso in ({"veto": {(x, "[M+NO3]-"): "rule C: c"}}, {"veto": {(x, H): "n", (y, H): "m"}}, {"veto": {}}, None):
        core = EV._series_pooled({"s1": rows}, iso=iso)
        frame = LL.measure_source("s1", rows.assign(__file="s1"), None)
        ref = LL.assign_levels(frame, set(), None, None, iso)
        m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
        assert len(m) == len(core) == 4
        assert (m["evidence_level"] == m["level"]).all(), iso
        assert (m["chan2_x"] == m["chan2_y"]).all() and (m["branch_x"] == m["branch_y"]).all()
        assert (m["iso_veto_x"] == m["iso_veto_y"]).all() and (m["iso_note_x"] == m["iso_note_y"]).all()
    # without rule K's facts the two nitrate clusters stay two channels
    w = "C8H12O4"
    two = pd.concat([rows, ledger([m0("w14", w, adduct="[M+NO3]-", ion=_ion(w, "[M+NO3]-"),
                                      mz=C.ion_mz(w, "[M+NO3]-")),
                                   m0("w15", w, adduct="[M+^NO3]-", ion="C8H12^NO7-",
                                      mz=C.ion_mz(w, "[M+^NO3]-"))])], ignore_index=True)
    iso = {"veto": {(x, H): "n"}}
    core = EV._series_pooled({"s1": two}, iso=iso)
    ref = LL.assign_levels(LL.measure_source("s1", two.assign(__file="s1"), None), set(), None, None, iso)
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert (m["evidence_level"] == m["level"]).all() and (m["chan2_x"] == m["chan2_y"]).all()
    assert m.loc[m["neutral_formula"] == w, "chan2_x"].all()
    # with rule K's facts the fold still applies, and the two alien sets add up
    lab = {"untie": set(), "veto": {}, "alien": {(y, "[M+NO3]-")}}
    iso = {"veto": {(x, "[M+NO3]-"): "c"}}
    core = EV._series_pooled({"s1": rows}, label=lab, iso=iso)
    ref = LL.assign_levels(LL.measure_source("s1", rows.assign(__file="s1"), None), set(), None, lab, iso)
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert (m["evidence_level"] == m["level"]).all()                   # the private decision, alike in both
    assert not m.loc[m["adduct"] == H, "branch_x"].any()                # x vetoed, y's 14N line alien


def _iso_table(rows) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    for c in IC.TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    return t[list(IC.TABLE_COLUMNS)]


def test_the_reference_script_reads_the_table_like_the_engine(tmp_path, capsys):
    LL = _ll()
    x = "C10H18O4"
    run = tmp_path / "out" / "RUN_1"
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    rows = _branch_rows(x)
    rows.to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    table = _iso_table([dict(neutral_formula=x, adduct="[M+NO3]-", check="HIGH", veto=True, note="h"),
                        dict(neutral_formula=x, adduct="[M+NO3]-", check="C", veto=True, note="c"),
                        dict(neutral_formula=x, adduct=H, check="REQ", veto=False, note="")])
    table.to_csv(run / "tables" / "iso_checks.csv", index=False)
    assert LL.iso_check_facts(str(run)) == {"veto": {(x, "[M+NO3]-"): "rule C: c; HIGH: h"}, "lock": {}} \
        == IC.facts(table)
    off = LL.series_run([str(run)], []).set_index(["neutral", "adduct"])
    on = LL.series_run([str(run)], [], None, None, "auto").set_index(["neutral", "adduct"])
    assert not off.at[(x, "[M+NO3]-"), "iso_veto"] and off.at[(x, H), "branch"]
    assert on.at[(x, "[M+NO3]-"), "iso_veto"] and not on.at[(x, H), "iso_veto"] and not on.at[(x, H), "branch"]
    explicit = LL.series_run([str(run)], [], None, None, str(run / "tables" / "iso_checks.csv"))
    assert list(explicit["level"]) == list(on["level"])
    assert list(LL.series_run([str(tmp_path / "out")], [], None, None, "auto")["level"]) == list(on["level"])
    core = EV._series_pooled({"s1": rows}, iso=IC.facts(table))
    m = core.merge(on.reset_index(), left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == len(core) and (m["evidence_level"] == m["level"]).all()
    out = tmp_path / "levels.csv"
    assert LL.series_main([str(run), "--iso-checks", "--out", str(out)]) == 0
    got = pd.read_csv(out).set_index(["neutral", "adduct"])
    assert got.loc[(x, "[M+NO3]-"), "iso_veto"] and got.loc[(x, "[M+NO3]-"), "iso_note"] == "rule C: c; HIGH: h"
    assert LL.series_main([str(run), "--out", str(out)]) == 0
    assert not pd.read_csv(out).set_index(["neutral", "adduct"])["iso_veto"][(x, "[M+NO3]-")]
    # no table: nothing fires, stderr says so; an empty table is None too
    bare = tmp_path / "bare"
    (bare / "per_file").mkdir(parents=True)
    rows.to_csv(bare / "per_file" / "s1_ledger.csv", index=False)
    assert LL.iso_check_facts(str(bare)) is None
    assert "no iso_checks.csv" in capsys.readouterr().err
    (bare / "tables").mkdir()
    IC._empty().to_csv(bare / "tables" / "iso_checks.csv", index=False)
    assert LL.iso_check_facts(str(bare)) is None
    # a named table applies only to a source that holds every pair it vetoes: another batch is untouched
    other = tmp_path / "other"
    (other / "per_file").mkdir(parents=True)
    _branch_rows("C9H14O4").to_csv(other / "per_file" / "s1_ledger.csv", index=False)
    named = tmp_path / "t.csv"
    _iso_table([dict(neutral_formula=x, adduct="[M+NO3]-", check="C", veto=True, note="c"),
                dict(neutral_formula="C9H14O4", adduct=H, check="C", veto=True, note="c")]).to_csv(named, index=False)
    got = LL.series_run([str(other)], [], None, None, str(named))
    assert not got["iso_veto"].any() and got["branch"].all()
    assert "does not hold 1 pair(s) the table vetoes" in capsys.readouterr().err
    got = LL.series_run([str(run), str(other)], [], None, None, str(run / "tables" / "iso_checks.csv"))
    assert got.loc[got.source == "RUN_1"].set_index("adduct")["iso_veto"]["[M+NO3]-"]
    assert not got.loc[got.source == "other", "iso_veto"].any()


# --------------------------------------------------------------------------- the batch, end to end
X = "C12H20O4"


def _batch(ts_rows=True):
    rows = []
    for i in range(N):
        rows += _filler(i)
        rows += _refs(i) + _c13(i, X, H, c=6.0) + _c13(i, Y, H, h=1e4) + _c13(i, Z, H, h=1e4, phase=1.0)
        rows.append(_row(i, C.ion_mz(Z, H) + 1.9979535, 1e4 * _wave(i, 1.0)))   # Z's 1:1 line at +81Br
    pk = pd.DataFrame(rows)[["sample_item_id", "datetime_utc", "mz", "height", "area"]]
    pk["sample_item_name"] = "n_" + pk["sample_item_id"]
    return pk


def _run_batch(tmp_path, monkeypatch, *, ts=True, resolving_power=100_000):
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as TT
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    commits = [(n, H) for n in REFS + [X, Y, Z]]

    def fake_assign(sid, context="ambient-air", **kw):
        ids = [f"P{j}" for j in range(len(commits))]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": [C.ion_mz(n, a) for n, a in commits],
                                         "height": [5000.0] * len(ids)}))
        for pid, (n, a) in zip(ids, commits):
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=_ion(n, a), ion_score=0.95,
                                compound_score=0.95, ppm_error=0.1, pass_no=1, method="cheminfo", confidence="High",
                                commentary=f"Pass 1: {n} {a}")
        TT.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        return {"ledger": led, "stats": {"noise_edge_cps": 10.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    pk = _batch()
    AB.run(peaks=pk, ts_peaks=pk if ts else None, reagent="NO3", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=resolving_power, log=lambda *a: None)


def test_a_batch_writes_the_table_and_levels_the_vetoes(tmp_path, monkeypatch):
    """assign_batch.run on an Orbitrap-class batch: the table is written from the
    stamped series, the summary carries the funnel, the pooled fact table carries
    each veto with the check's note, and the scale rejects each refuted pair on
    the merged ledger (5b, "refuted: iso_veto")."""
    _run_batch(tmp_path, monkeypatch)
    table = pd.read_csv(tmp_path / "tables" / "iso_checks.csv")
    assert list(table.columns) == list(IC.TABLE_COLUMNS)
    assert set(table["instrument"]) == {"orbitrap"}
    veto = IC.veto(table)
    assert set(veto) == {(X, H), (Y, H), (Z, H)}
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["iso_checks"]
    assert summ["instrument"] == "orbitrap" and summ["vetoed_pairs"] == 3
    assert summ["C"]["contradict"] == 1 and summ["REQ"]["absent"] == 1 and summ["HIGH"]["too_high"] == 1
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False) \
        .set_index(["neutral_formula", "adduct"])
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert {"iso_veto", "iso_note"} <= set(ev.columns)
    ev = ev.set_index(["neutral_formula", "adduct"])
    for k in veto:
        assert merged.loc[k, "evidence_level"] == "5b" and merged.loc[k, "would_lift"].startswith("refuted: iso_veto")
        assert EV.truthy(ev.loc[k, "iso_veto"]) and ev.loc[k, "iso_note"] == veto[k]
    assert (merged.loc[[(n, H) for n in REFS], "evidence_level"] != "5b").all()


def test_a_batch_without_a_time_series_writes_an_empty_table(tmp_path, monkeypatch):
    _run_batch(tmp_path, monkeypatch, ts=False)
    table = pd.read_csv(tmp_path / "tables" / "iso_checks.csv")
    assert table.empty and list(table.columns) == list(IC.TABLE_COLUMNS)
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["iso_checks"]
    assert summ == {"instrument": "orbitrap", "tested": 0, "vetoed_pairs": 0, "locked_pairs": 0}
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False)
    assert not merged["would_lift"].astype(str).str.contains("iso_veto").any()
