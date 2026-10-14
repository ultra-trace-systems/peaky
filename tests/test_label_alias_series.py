"""A series step does not corroborate a count its 15N alias could be standing in
for (assignment/fluorine.py `series_alias_holds`, read by the tier engine and the
evidence scale's `anchor` axis).

On a 15N-nitrate run two element swaps are sub-mDa: C8S == H ^N O7 (0.27 mDa)
and C2F3 == H2 ^N O4 (0.21 mDa). Every organosulfur reading with C >= 8 and every
fluorinated one with C >= 2, F >= 3 has a same-ion reading with one more ^N, and
the swap keeps every CH2 / O / H2O / CF2 step: a homologous series reads both
alike. The fixtures are synthetic: an organosulfur [M-H]- reading committed by a
series step (`series_unit`) from an anchor, beside a committed labelled cluster
(the ledger is a 15N-nitrate one) or an unlabelled one.

Run: pytest tests/test_label_alias_series.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import fluorine as FL
from peaky.assignment import ledger as L
from peaky.assignment import passes as P
from peaky.assignment import tiers as T
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

ADD = "[M-H]-"
NO3_15 = "[M+^NO3]-"
NO3_14 = "[M+NO3]-"
#: the series-step reading, its anchor (one H2O heavier) and the alias of the reading
S_READ, S_ANCHOR = "C20H18O3S", "C20H20O4S"
S_ALIAS = "C12H18O7"                       # on [M+^NO3]-: C20H18O3S - C8S + O4
F14 = (1.0 - ISO.label_purity()) / ISO.label_purity()


def _ion(nf: str, adduct: str = ADD) -> str:
    return C.format_formula(T._ion_counts(nf, adduct, labelled=True)) + "-"


def _mz(nf: str, adduct: str = ADD) -> float:
    return C.ion_mz(nf, adduct)


def _cfg(floor: float = 50.0) -> P.PassConfig:
    cfg = P.PassConfig(height_cutoff_cps=floor, instrument_class="orbitrap")
    cfg.cal_mu, cfg.cal_sigma = 0.0, 0.2
    return cfg


def _ledger(*, read: str = S_READ, adduct: str = ADD, h: float = 300.0, cluster: str = NO3_15,
            unit: str | None = "H2O", anchor: str | None = "anc", extra: list[tuple] = ()) -> pd.DataFrame:
    """The reading `read` on `adduct` at height `h`, committed by a series step
    (`unit`, `anchor`) from its anchor (pass 3, no series credit of its own),
    plus a bright committed cluster on `cluster` and `extra` picked lines
    (peak_id, mz, height)."""
    rows = [dict(peak_id="cl", mz=_mz("C10H16O4", cluster), height=5e4),
            dict(peak_id="anc", mz=_mz(S_ANCHOR), height=h),
            dict(peak_id="s", mz=_mz(read, adduct), height=h)]
    rows += [dict(peak_id=p, mz=m, height=hh) for p, m, hh in extra]
    led = L.new_ledger(pd.DataFrame(rows))
    L.commit_assignment(led, "cl", neutral_formula="C10H16O4", adduct=cluster, ion_formula=_ion("C10H16O4", cluster),
                        ion_score=0.97, compound_score=0.97, ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                        confidence="High", commentary="Pass 1")
    L.commit_assignment(led, "anc", neutral_formula=S_ANCHOR, adduct=ADD, ion_formula=_ion(S_ANCHOR),
                        ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=3,
                        method="contaminant:organosulfate", confidence="High", commentary="Pass 3")
    L.commit_assignment(led, "s", neutral_formula=read, adduct=adduct, ion_formula=_ion(read, adduct),
                        ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=4, method="residual:series",
                        confidence="High", commentary="Pass 4 (deep series)", anchor_peak_id=anchor,
                        series_unit=unit)
    return led


def _tier(led, pid="s", cfg=None):
    t = T.compute_tiers(led, cfg=cfg or _cfg()).set_index("peak_id")
    return t.at[pid, "tier"], t.at[pid, "tier_reason"]


# =========================================================================== the alias table
def test_the_c8s_alias_is_sub_mda_and_the_c2f3_one_unchanged():
    assert FL.ALIAS_C8S.mda < 0.3 and FL.ALIAS_C2F3.mda < 0.25
    assert FL.ALIAS_OUT == FL.ALIAS_C2F3.out and FL.ALIAS_SWAP == FL.ALIAS_C2F3.swap
    assert {a.element for a in FL.LABEL_ALIASES} == {"F", "S"}
    # the swap applied to the [M-H]- reading is the alias's [M+^NO3]- ion
    swapped = C.parse_formula(_ion(S_READ))
    for el, d in FL.ALIAS_C8S.swap.items():
        swapped[el] = swapped.get(el, 0) + d
    assert C.format_formula({k: v for k, v in swapped.items() if v}) + "-" == _ion(S_ALIAS, NO3_15)
    assert abs(_mz(S_READ) - _mz(S_ALIAS, NO3_15)) * 1e3 == pytest.approx(FL.ALIAS_C8S.mda, abs=1e-3)


def test_an_alias_whose_neutral_cannot_exist_is_not_one():
    assert FL.alias_admissible(C.parse_formula(S_READ), FL.ALIAS_C8S)
    # C12H18O2S - C8S + H ^N O7: DBE -4
    assert not FL.alias_admissible(C.parse_formula("C12H18O2S"), FL.ALIAS_C8S)
    # a reading short of C8S has nothing to swap
    assert FL.alias_admissible(C.parse_formula("C6H10O4S"), FL.ALIAS_C8S)
    # the default alias is still the fluorine one
    assert FL.alias_admissible(C.parse_formula("C6H5F7O2"))
    assert not FL.alias_admissible(C.parse_formula("C6H7F7O"))


@pytest.mark.parametrize("floor, ratio, expect", [
    (50.0, None, "open"),          # the alias's twin would read ~3 cps: under the floor
    (50.0, 0.02, "veto"),          # a line at the alias's 14N twin, in the band
    (0.5, None, "tested"),         # testable and absent: the alias is ruled out
])
def test_the_c8s_alias_takes_the_same_14n_twin_test(floor, ratio, expect):
    h0 = 300.0
    extra = [("tw", _mz(S_READ) - FL.D_15N, h0 * ratio)] if ratio else []
    led = _ledger(extra=extra)
    ion = C.parse_formula(_ion(S_READ))
    veto, open_ = FL._alias_twin(ion, h0, FL._Lines(led), _mz(S_READ), 1.0, F14, floor,
                                 C.parse_formula(S_READ), alias=FL.ALIAS_C8S)
    assert (expect == "veto") == (veto is not None)
    assert (expect == "open") == (open_ is not None)
    if veto:
        assert "C8S -> H^NO7" in veto


# =========================================================================== the tier engine
def test_a_series_step_does_not_lift_an_untested_s_whose_alias_is_open():
    """The defect: an organosulfur reading whose 34S line is under the floor is
    Candidate, unless a series step 'corroborates' it -- and on a 15N-nitrate run
    the step is its C8S -> H^NO7 alias's step too. Its credit is withheld."""
    led = _ledger()
    holds = FL.series_alias_holds(led, cfg=_cfg())
    assert set(holds) == {"s"} and "C8S -> H^NO7" in holds["s"] and "untested" in holds["s"]
    tier, reason = _tier(led)
    assert tier == "Candidate", reason
    assert reason.startswith("S1 untestable at this intensity")
    assert FL.SERIES_ALIAS_MARK in reason and "series-anchor support" not in reason


def test_an_unlabelled_ledger_keeps_the_series_credit():
    led = _ledger(cluster=NO3_14)
    assert FL.series_alias_holds(led, cfg=_cfg()) == {}
    tier, reason = _tier(led)
    assert tier == "Assigned" and "series-anchor support" in reason and "S count untested" in reason
    assert FL.SERIES_ALIAS_MARK not in reason


def test_a_row_without_series_credit_is_not_touched():
    led = _ledger(unit=None, anchor=None)
    assert FL.series_alias_holds(led, cfg=_cfg()) == {}
    tier, reason = _tier(led)
    assert tier == "Candidate" and FL.SERIES_ALIAS_MARK not in reason


def test_a_seen_34s_line_keeps_the_series_credit():
    """The S count tested by its own line (present, though predicted under the
    refutation multiple): the alias has no S, the series step stands."""
    h0 = 1000.0
    led = _ledger(h=h0, extra=[("s34", _mz(S_READ) + ISO.D_34S, 0.045 * h0)])
    assert FL.series_alias_holds(led, cfg=_cfg()) == {}
    tier, reason = _tier(led)
    assert tier == "Assigned" and "series-anchor support" in reason and "S envelope line present" in reason


def test_the_alias_twin_seen_withholds_the_credit():
    led = _ledger(extra=[("tw", _mz(S_READ) - FL.D_15N, 300.0 * 0.02)])
    holds = FL.series_alias_holds(led, cfg=_cfg())
    assert "14N twin of its 15N alias (C8S -> H^NO7, ^N1)" in holds["s"]
    tier, reason = _tier(led)
    assert tier == "Candidate" and FL.SERIES_ALIAS_MARK in reason


def test_an_inadmissible_alias_keeps_the_credit():
    led = _ledger(read="C12H18O2S")
    assert FL.series_alias_holds(led, cfg=_cfg()) == {}
    assert _tier(led)[0] == "Assigned"


def test_a_reading_that_carries_15n_has_its_alias_left_open():
    """[M+^NO3]- of an organosulfur neutral carries one ^N: its own 14N twin
    cannot be told from its alias's, so nothing tests the alias."""
    led = _ledger(read="C20H18O3S", adduct=NO3_15)
    holds = FL.series_alias_holds(led, cfg=_cfg())
    assert "carries ^N1" in holds["s"]
    # its 34S line testable (a low floor) and absent: the count is refuted, not untested -- no hold
    assert FL.series_alias_holds(led, cfg=_cfg(floor=0.5)) == {}


def test_the_fluorine_alias_is_held_the_same_way():
    """An F-rich reading (F >= 2H: no partial-F cap) on a degenerate mass, whose
    only corroboration is a CF2 series step: its C2F3 -> H2^NO4 alias keeps the
    step, so the step does not break the degeneracy -- unless the alias is
    tested (bright enough for its twin to show, and absent)."""
    nf = "C8H3F13O2"
    for h, floor, held in ((300.0, 50.0, True), (2e4, 50.0, False)):
        led = _ledger(read=nf, h=h, unit="CF2")
        led.loc[led["peak_id"] == "s", "degeneracy_density"] = 4
        holds = FL.series_alias_holds(led, cfg=_cfg(floor))
        assert ("s" in holds) == held, holds
        tier, reason = _tier(led, cfg=_cfg(floor))
        if held:
            assert "C2F3 -> H2^NO4" in holds["s"]
            assert tier == "Candidate" and reason.startswith("mass-degenerate") and FL.SERIES_ALIAS_MARK in reason
        else:
            assert tier == "Assigned" and "series-anchor support" in reason


def test_the_hold_is_one_switch(monkeypatch):
    led = _ledger()
    monkeypatch.setattr(FL, "ALIAS_SERIES_HOLDS", False)
    assert FL.series_alias_holds(led, cfg=_cfg()) == {}
    tier, reason = _tier(led)
    assert tier == "Assigned" and "series-anchor support" in reason


# =========================================================================== the evidence scale and the vote
def _classes(led):
    T.apply_tiers(led, cfg=_cfg())
    vc = EV.vote_classes(EV.trim(led))
    pid = led.loc[vc.index, "peak_id"]
    return dict(zip(pid, vc))


def test_the_anchor_axis_reads_the_withheld_credit():
    unl = _ledger(cluster=NO3_14)
    assert _classes(unl)["s"] == 1                             # 4b: one corroboration (anchor)
    assert EV.SERIES_HELD not in EV.trim(unl).columns          # no mark, no flag: the frame as before
    led = _ledger()
    assert _classes(led)["s"] == 0
    tr = EV.trim(led)
    assert tr.loc[tr["peak_id"] == "s", EV.SERIES_HELD].all() and not tr.loc[tr["peak_id"] != "s", EV.SERIES_HELD].any()
    # the full ledger (the per-file evidence stage) and its trim read alike
    for frame in (led, tr):
        lv = EV._series_levels(frame).set_index("peak_id")
        assert "anchor" not in str(lv.at["s", "evidence_axes"]).split("|"), lv.loc["s"].to_dict()
    assert EV.series_held(EV.trim(tr)).equals(EV.series_held(tr))   # a trim of a trim keeps the flag


def test_a_one_file_series_alias_no_longer_outvotes_the_count():
    """One file reads the line as the series-step organosulfur reading, three as
    its alias, the labelled cluster, at Candidate. The step had made the one file
    'formula confirmed' (class 1) and the evidence outranked the count; with the
    credit withheld the count decides."""
    led = _ledger()
    T.apply_tiers(led, cfg=_cfg())
    m0 = AB._m0(led)
    m0[AB.VOTE_CLASS] = EV.vote_classes(EV.trim(led)).reindex(m0.index).fillna(0).astype(int)
    per_file = {"f0": m0}
    for k in range(1, 4):
        per_file[f"f{k}"] = pd.DataFrame([dict(mz=_mz(S_ALIAS, NO3_15) + 1e-5 * k, neutral_formula=S_ALIAS,
                                               adduct=NO3_15, tier="Candidate", ion_score=0.9,
                                               **{AB.VOTE_CLASS: 0})])
    merged, _j = AB.align(per_file, tol_ppm=6.0)
    row = merged.iloc[(merged["mz"] - _mz(S_READ)).abs().argsort().iloc[0]]
    assert (row["neutral_formula"], row["adduct"]) == (S_ALIAS, NO3_15), row.to_dict()
