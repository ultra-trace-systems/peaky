"""C19(c): the `tentative_lead` split of `below_assignability`, a refactor that
moves no level, tier or claim.

`below_assignability` used to say two things. Either the assignment argues with
itself (O >= 11 on a saturated window, an O-monster, a carbon cluster, a
carbon-rich skeleton, an impossible ionization, an off-calibration residual fit,
unconfirmed F >= 4), or the proposal is only unsupported: a reference-list match
too dim to test, a commit outside the element budget, a residual fit with
nothing behind it, an uncorroborated radical anion, a reagent-N re-read. The
second kind is now `tentative_lead`.

Pinned here:
- each lead setter writes the lead and not below, and each hard setter still
  writes below and not the lead;
- a row that both mark keeps below;
- the pooled `lead` fact holds when ANY row is a lead. A lead-only pair is 5b
  with the reason a below row gives ("5b: below assignability"), in the engine
  and in the reference script;
- nothing that decides a tier, a report row or an ion-only parent can tell the
  two columns apart;
- commit, clear and displace reset both flags and never create a column;
- a ledger written before the split (no column) levels exactly as it did.

Run: pytest tests/test_tentative_lead.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import cleanup as CL
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import passes as P
from peaky.assignment import plausibility as PL
from peaky.assignment import reflists as RL
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C
from peaky.io import publish as PUB
from peaky.reporting import report as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import level_ledger as LL  # noqa: E402

BELOW, LEAD = "below_assignability", "tentative_lead"
FLAG_COLUMNS = [BELOW, LEAD]


def _flags(led, i) -> tuple[bool, bool]:
    """(below, lead) of row `i`, reading a missing column as False."""
    return tuple(bool(led.at[i, c]) if c in led.columns and pd.notna(led.at[i, c]) else False
                 for c in FLAG_COLUMNS)


def _tiered(rows: list[dict]) -> pd.DataFrame:
    """A committed ledger as the tier stage leaves it: the rows given, then the
    O >= 11 flag pass, which is where the flag columns are born."""
    led = pd.DataFrame(rows)
    for col, default in (("commentary", ""), ("tier", "Assigned"), ("tier_reason", ""),
                         ("isotopologues", "[]"), ("degeneracy_note", ""), ("method", "cheminfo+grid"),
                         ("confidence", "High")):
        if col not in led.columns:
            led[col] = default
    T.flag_below_assignability(led)
    return led


# =========================================================================== the setters
def test_the_tier_stage_creates_both_flags_and_the_o11_flag_is_hard():
    led = _tiered([dict(peak_id="mon", role="M0", neutral_formula="C20H20O29", adduct="[M-H]-",
                        degeneracy_note="MASS-SATURATED: 52 plausible formulas", tier="Candidate"),
                   dict(peak_id="cho", role="M0", neutral_formula="C10H16O4", adduct="[M-H]-",
                        degeneracy_note="unique")])
    assert set(FLAG_COLUMNS) <= set(led.columns)
    assert _flags(led, 0) == (True, False)
    assert _flags(led, 1) == (False, False)


# one ledger row per hard setter: the demote writes below_assignability, never the lead
HARD = {
    "oxygen monster": (lambda led: PL.demote_oxygen_monsters(led, log=lambda *a: None),
                       dict(neutral_formula="C5H4O8", degeneracy_note="MASS-SATURATED: 27 plausible formulas")),
    "carbon cluster": (lambda led: PL.demote_carbon_clusters(led, log=lambda *a: None),
                       dict(neutral_formula="C24H2")),
    "unconfirmed fluorine": (lambda led: CL.demote_unconfirmed_fluorine(led, log=lambda *a: None),
                             dict(neutral_formula="C11H6F16")),
    "carbon-rich": (lambda led: CL.demote_implausible_carbon(led, log=lambda *a: None),
                    dict(neutral_formula="C36H6O")),
    "impossible ionization": (lambda led: CL.demote_implausible_ionization(led, log=lambda *a: None),
                              dict(neutral_formula="C7H10")),
}


@pytest.mark.parametrize("name", sorted(HARD))
def test_each_hard_setter_writes_below_and_not_the_lead(name):
    demote, row = HARD[name]
    led = _tiered([dict(dict(peak_id="p", role="M0", mz=300.0, adduct="[M-H]-"), **row)])
    assert _flags(led, 0) == (False, False)
    demote(led)
    assert led.at[0, "tier"] == "Candidate"
    assert _flags(led, 0) == (True, False), name
    assert not led[LEAD].any()


class _ResidualCfg:
    minor_channels = ("[M+CO3]-", "[M+O2]-", "[M]-.")
    cal_mu, cal_sigma, cal_z_accept = 0.0, 0.4, 2.0


def _residual(neutral, adduct, *, ppm=-0.3, comm="", iso="[]"):
    return dict(peak_id=neutral + adduct, role="M0", neutral_formula=neutral, adduct=adduct,
                method="residual:series", ppm_error=ppm, isotopologues=iso, commentary=comm)


@pytest.mark.parametrize("row, kind", [
    (_residual("C6H5N3", "[M+Br]-"), "lead"),                                    # N3, no isotope
    (_residual("C12H6O2", "[M-H]-", comm="-2xCH2 (0 supporting anchors)"), "lead"),  # gap-fill, no anchors
    (_residual("C9H14O4", "[M+CO3]-"), "lead"),                                  # the sole minor channel
    (_residual("C9H12O5", "[M-H]-", ppm=5.0, iso='[{"label": "13C"}]'), "below"),  # off-calibration
])
def test_the_speculative_residual_demote_splits_by_reason(row, kind):
    led = _tiered([row])
    out = CL.demote_speculative_residual(led, _ResidualCfg(), log=lambda *a: None)
    assert out == {"residual_demoted": 1}
    assert led.at[0, "tier"] == "Candidate"
    assert _flags(led, 0) == ((False, True) if kind == "lead" else (True, False))
    assert list(led[LEAD]) == [kind == "lead"] and list(led[BELOW]) == [kind == "below"]
    assert "speculative residual fit -- " in led.at[0, "commentary"]      # the note is unchanged


def test_the_off_budget_demote_writes_the_lead_and_keeps_its_note_and_audit():
    led = _tiered([dict(peak_id="p", role="M0", mz=200.0, neutral_formula="C5H6ClN2OP", adduct="[M-H]-",
                        method="certified:multi-channel"),
                   dict(peak_id="q", role="M0", mz=201.0, neutral_formula="C10H16O5", adduct="[M-H]-")])
    audit = []
    out = PL.demote_implausible(led, audit=audit, log=lambda *a: None, context="ambient-air")
    assert out["budget_demoted"] == 1
    assert _flags(led, 0) == (False, True) and _flags(led, 1) == (False, False)
    assert "on no curated list" in led.at[0, "commentary"]
    assert len(audit) == 1 and audit[0]["after_tier_or_role"] == "Candidate"


def test_the_demote_row_takes_a_lead_argument():
    led = _tiered([dict(peak_id="p", role="M0", neutral_formula="C10H16O5", adduct="[M-H]-")])
    PL._demote_row(led, 0, reason="r", audit=None, evidence="e", degeneracy_note=None, n_iso=0, lead=True)
    assert _flags(led, 0) == (False, True)
    PL._demote_row(led, 0, reason="r", audit=None, evidence="e", degeneracy_note=None, n_iso=0)
    assert _flags(led, 0) == (True, True)                  # a hard demote on a lead: both, still hard


def test_an_uncorroborated_radical_anion_is_a_lead_and_corroboration_clears_both():
    led = _tiered([dict(peak_id="c", role="M0", neutral_formula="C3H4", adduct="[M+CO3]-", ion_formula="",
                        dbe=2.0),
                   dict(peak_id="k", role="M0", neutral_formula="C4H4O3", adduct="[M-H]-", ion_formula="",
                        dbe=3.0),
                   dict(peak_id="u", role="M0", neutral_formula="C6H6", adduct="[M+CO3]-", ion_formula="",
                        dbe=4.0)])
    led.loc[0, [BELOW, LEAD]] = True                       # whatever an earlier stage said
    out = CL.relabel_radical_anions(led, log=lambda *a: None)
    assert out == {"radical_relabeled": 2, "radical_corroborated": 1}
    assert led.at[0, "adduct"] == "[M]-." and _flags(led, 0) == (False, False)
    assert led.at[2, "neutral_formula"] == "C7H6O3" and _flags(led, 2) == (False, True)
    assert led.at[2, "confidence"] == "Low (radical anion)"


def test_the_reagent_n_reread_is_a_lead_and_a_frame_without_flags_gets_none():
    rows = [dict(peak_id="u", role="M0", neutral_formula="C5H6", adduct="[M+(CH4N2O)H]+", ion_formula="",
                 dbe=3.0)]
    led = _tiered(rows)
    assert CL.relabel_reagent_n_adducts(led, log=lambda *a: None) == {"reagent_n_relabeled": 1}
    assert led.at[0, "neutral_formula"] == "C6H10N2O" and _flags(led, 0) == (False, True)
    merged = pd.DataFrame(rows).assign(tier="Assigned")    # the merged ledger carries no flag
    CL.relabel_reagent_n_adducts(merged, log=lambda *a: None)
    assert merged.at[0, "neutral_formula"] == "C6H10N2O"
    assert not (set(FLAG_COLUMNS) & set(merged.columns))


F_CONF, F_DIM = "C10H16O5", "C9H14O5"
RLIST = RL.ReferenceList(
    id="leadlist", system="t", label="Lead test list", data_version="1",
    polarity="negative", native_detection="[M-H]-", applies_to_contexts=("x",),
    references=({"authors": "X", "title": "T"},),
    formulas=frozenset({F_CONF, F_DIM}), radicals=frozenset(),
    conditions_of={f: () for f in (F_CONF, F_DIM)},
    source_file="t.json", always_active=False, meta_of={})


def _oracle(client, sample_id, formulas, *, allow_partial=True, mechanism_ids=None):
    """Server stand-in: both formulas match; only F_CONF has a scored 13C line."""
    rows = []
    for f in formulas:
        cnt = C.parse_formula(f)
        cnt["H"] = cnt.get("H", 0) - 1
        base = dict(compound_formula=f, compound_score=0.9, ion_formula=C.format_formula(cnt) + "-",
                    ion_score=0.9, iso_label="M0", is_base=True, iso_score=0.9, sample_peak_id=f,
                    sample_peak_mz=C.ion_mz(f, "[M-H]-"), sample_peak_intensity=1e4, ppm_error=0.2)
        rows.append(base)
        if f == F_CONF:
            rows.append(dict(base, is_base=False, iso_label="13C",
                             sample_peak_mz=C.ion_mz(f, "[M-H]-") + 1.00336, iso_score=0.95))
    return pd.DataFrame(rows)


def test_the_reflist_dim_rescue_writes_the_lead_and_the_confirmed_one_neither():
    led = L.new_ledger(pd.DataFrame([("conf", C.ion_mz(F_CONF, "[M-H]-"), 1.0e5),
                                     ("dim", C.ion_mz(F_DIM, "[M-H]-"), 500.0)],
                                    columns=["peak_id", "mz", "height"]))
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    out = RL.rescue_unexplained_by_reflist(None, "S", led, None, cfg, [RLIST], ["[M-H]-"],
                                           score_fn=_oracle, log=lambda *a: None)
    assert out == {"rescued": 1, "tentative": 1}
    by = led.set_index("peak_id")
    assert "dim" in str(by.at["dim", "confidence"])
    assert (bool(by.at["dim", BELOW]), bool(by.at["dim", LEAD])) == (False, True)
    assert (bool(by.at["conf", BELOW]), bool(by.at["conf", LEAD])) == (False, False)


def test_a_row_both_mark_keeps_below():
    """An F-monster outside the element budget: the fluorine demote (hard) and the
    budget demote (a lead) both fire; the row stays hard and levels as before."""
    led = _tiered([dict(peak_id="p", role="M0", mz=300.0, neutral_formula="C11H6F16", adduct="[M-H]-",
                        ion_formula="C11H5F16-", height=1000.0, confidence="High", tied=False,
                        degeneracy_density=1, resolvability="resolved")])
    CL.demote_unconfirmed_fluorine(led, log=lambda *a: None)
    PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None)
    assert _flags(led, 0) == (True, True)
    lv = EV.compute_levels(led)
    assert lv.at[0, "evidence_level"] == "5b" and lv.at[0, "level_reason"] == "5b: below assignability"


# =========================================================================== the level
LEDGER_COLUMNS = list(EV.PREDICATE_COLUMNS)


def _m0(peak_id, neutral, adduct="[M-H]-", *, below=False, lead=False, tied=False, anchor=None,
        confidence="High", mz=200.0):
    return {"role": "M0", "peak_id": peak_id, "parent_peak_id": None, "iso_label": None,
            "neutral_formula": neutral, "adduct": adduct, "ion_formula": neutral, "mz": mz,
            "height": 1000.0, "tier": "Candidate", "method": "pass2", "confidence": confidence,
            "tied": tied, "below_assignability": below, "tentative_lead": lead,
            "degeneracy_density": 0.5, "degeneracy_note": "", "resolvability": "resolved",
            "series_unit": None, "anchor_peak_id": anchor, "isotopologues": "",
            "ppm_error_cal": 0.4, "occurrence": 0.9}


def _child(peak_id, parent, label, height):
    row = _m0(peak_id, None, adduct=None)
    row.update(role="iso_child", parent_peak_id=parent, iso_label=label, height=height,
               degeneracy_density=None)
    return row


def _frame(rows):
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def test_the_lead_is_a_predicate_column_trim_keeps():
    assert LEAD in EV.PREDICATE_COLUMNS and BELOW in EV.PREDICATE_COLUMNS
    assert LEAD in EV.trim(_frame([_m0("p", "C7H12O4", lead=True)])).columns


def test_a_lead_only_pair_is_5b_with_the_reason_a_below_row_gives():
    for extra, reason in (({}, "5b: below assignability"),
                          ({"tied": True}, "5b: near-tie broken by the arbiter; below assignability"),
                          ({"confidence": "Low (0.3)"},
                           "5b: below assignability; engine confidence Low/Suspect")):
        as_below = EV.compute_levels(_frame([_m0("p", "C7H12O4", below=True, anchor="a", **extra)]))
        as_lead = EV.compute_levels(_frame([_m0("p", "C7H12O4", lead=True, anchor="a", **extra)]))
        both = EV.compute_levels(_frame([_m0("p", "C7H12O4", below=True, lead=True, anchor="a", **extra)]))
        for out in (as_below, as_lead, both):
            assert out.at[0, "evidence_level"] == "5b"
            assert out.at[0, "level_reason"] == reason
        pd.testing.assert_frame_equal(as_below, as_lead)
    # the mutant: no flag at all is not hard (the anchor gives 4b)
    assert EV.compute_levels(_frame([_m0("p", "C7H12O4", anchor="a")])).at[0, "evidence_level"] == "4b"


def test_the_pooled_lead_is_any_row():
    one = _frame([_m0("a", "C8H12O4", lead=True, anchor="x"), _m0("b", "C9H14O4", anchor="x", mz=210.0)])
    two = _frame([_m0("a", "C8H12O4", anchor="x"), _m0("b", "C9H14O4", anchor="x", mz=210.0)])
    pairs = EV.level_pooled({"f1": one, "f2": two}).set_index("neutral_formula")
    assert bool(pairs.at["C8H12O4", "lead"]) and not bool(pairs.at["C8H12O4", "below"])
    assert pairs.at["C8H12O4", "evidence_level"] == "5b"
    assert pairs.at["C8H12O4", "level_reason"] == "5b: below assignability"
    assert not bool(pairs.at["C9H14O4", "lead"]) and pairs.at["C9H14O4", "evidence_level"] == "4b"


def _mixed(flag: str) -> pd.DataFrame:
    """Every kind of pair the table reaches (branch, chan2, iso, curated, tie),
    with the flagged rows carrying the flag in the `flag` column."""
    lead_col = {"below": dict(below=True), "lead": dict(lead=True)}[flag]
    rows = [
        _m0("br1", "C10H16O5", **lead_col), _m0("br2", "C10H16O5", "[M+NO3]-", mz=262.0),
        _m0("ch1", "C6H8O5", anchor="z", **lead_col), _m0("ch2", "C6H8O5", "[M+Br]-", mz=240.0),
        _m0("iso", "C9H14O4", **lead_col), _child("iso13", "iso", "13C", 100.0),
        _m0("tie", "C5H8O4", tied=True, **lead_col),
        _m0("kn", "HNO3", **lead_col), _m0("clean", "C7H10O4", anchor="z"),
    ]
    rows[7]["method"] = "known:atmospheric"
    return _frame(rows)


def test_the_flag_in_either_column_levels_and_tiers_identically():
    by_below, by_lead = _mixed("below"), _mixed("lead")
    lv_b, lv_l = EV.compute_levels(by_below), EV.compute_levels(by_lead)
    pd.testing.assert_frame_equal(lv_b, lv_l)
    assert set(lv_l.loc[by_lead[LEAD].astype(bool), "evidence_level"]) == {"5b"}
    pb = EV.level_pooled({"f": by_below}).drop(columns=["below", "lead"])
    pl = EV.level_pooled({"f": by_lead}).drop(columns=["below", "lead"])
    pd.testing.assert_frame_equal(pb, pl)
    # the tier engine reads neither flag: the same tiers whichever column carries it
    tb, tl = T.compute_tiers(by_below), T.compute_tiers(by_lead)
    pd.testing.assert_frame_equal(tb, tl)


def test_a_ledger_without_the_column_levels_exactly_as_before():
    """A ledger written before the split carries its leads in below_assignability
    and no tentative_lead column: the column reads False, nothing moves."""
    old = _mixed("below")
    no_col = old.drop(columns=[LEAD])
    pd.testing.assert_frame_equal(EV.compute_levels(no_col), EV.compute_levels(old))
    a = EV.level_pooled({"f": no_col})
    b = EV.level_pooled({"f": old})
    pd.testing.assert_frame_equal(a, b)
    assert not a["lead"].any()


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename(columns={"occurrence": "occurrence_y"}).to_csv(path, index=False)
    return path


def test_the_reference_script_reads_the_lead_in_lockstep(tmp_path):
    for flag in ("below", "lead"):
        frame = _mixed(flag)
        path = _write(tmp_path / flag / "s_ledger.csv", frame)
        got = LL.run([str(path)], [])
        eng = EV.compute_levels(frame)
        m0 = frame[frame.role == "M0"]
        engine = dict(zip(zip(m0.neutral_formula, m0.adduct), eng.loc[m0.index, "evidence_level"]))
        script = dict(zip(zip(got.neutral, got.adduct), got.level))
        assert script == engine, flag
        assert script[("C6H8O5", "[M-H]-")] == "5b"
        if flag == "lead":
            assert got.set_index(["neutral", "adduct"]).at[("C6H8O5", "[M-H]-"), "lead"]
    # a missing column reads False: the old ledger's levels, row for row
    old = _mixed("below")
    with_col = LL.run([str(_write(tmp_path / "w" / "s_ledger.csv", old))], [])
    no_col = LL.run([str(_write(tmp_path / "n" / "s_ledger.csv", old.drop(columns=[LEAD])))], [])
    assert list(with_col.level) == list(no_col.level)


# =========================================================================== the readers
def test_the_ion_only_stage_skips_a_lead_parent():
    from tests.test_ion_only import _cfg, _ledger, _row  # the ion-only fixture, as it is
    led = _ledger()
    led[LEAD] = False
    led.loc[led["peak_id"] == "A", LEAD] = True
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    assert out["ion_only_committed"] == 0 and out["ion_only_parents"] == 1
    assert _row(led, "H")["role"] == L.ROLE_UNEXPLAINED
    # the mutant: the same parent unflagged lends its row
    led = _ledger()
    led[LEAD] = False
    assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_committed"] == 1
    assert _flags(led, led.index[led["peak_id"] == "H"][0]) == (False, False)


def _report_ledger(flag: str) -> pd.DataFrame:
    peaks = pd.DataFrame({"peak_id": ["A", "D"], "mz": [200.1, 145.05], "height": [1.0e5, 5.0e4]})
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-", ion_formula="C10H15O4-",
                        ion_score=0.97, compound_score=0.96, ppm_error=-0.3, pass_no=1, method="cheminfo",
                        confidence="High", commentary="Pass 1")
    L.commit_assignment(led, "D", neutral_formula="C5H6ClN2OP", adduct="[M-H]-", ion_formula="C5H5ClN2OP-",
                        ion_score=0.85, compound_score=0.85, ppm_error=0.3, pass_no=4,
                        method="certified:multi-channel", confidence="Good", commentary="Pass 4")
    T.apply_tiers(led)
    led.loc[led["peak_id"] == "D", flag] = True
    return led


def test_the_below_assignability_sheet_lists_leads_too():
    by_below = R.build_sheets(_report_ledger(BELOW))["Below assignability"]
    by_lead = R.build_sheets(_report_ledger(LEAD))["Below assignability"]
    assert list(by_lead["neutral_formula"]) == list(by_below["neutral_formula"]) == ["C5H6ClN2OP"]
    assert LEAD in by_lead.columns and bool(by_lead[LEAD].iloc[0])
    assert not bool(by_below[LEAD].iloc[0])


def test_publish_carries_the_lead_beside_below():
    cols = PUB._ENGINE_PROVENANCE_COLUMNS
    assert LEAD in cols and cols.index(LEAD) == cols.index(BELOW) + 1
    row = pd.Series({"tier": "Candidate", BELOW: False, LEAD: True})
    prov = PUB._engine_provenance(row, None)
    assert prov[LEAD] is True and prov[BELOW] is False


# =========================================================================== the reset
def _flagged_ledger():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["A", "B", "C"], "mz": [200.0, 201.0, 202.0],
                                     "height": [1e5, 1e4, 5e4]}))
    for pid, f in (("A", "C10H16O4"), ("C", "C9H14O4")):
        L.commit_assignment(led, pid, neutral_formula=f, adduct="[M-H]-", ion_score=0.9, pass_no=1,
                            method="cheminfo", confidence="High", commentary="Pass 1")
    led[BELOW] = True
    led[LEAD] = True
    return led


def test_commit_clear_and_displace_reset_both_flags():
    led = _flagged_ledger()
    a = led.index[led["peak_id"] == "A"][0]
    L.commit_assignment(led, "A", neutral_formula="C10H14O4", adduct="[M-H]-", ion_score=0.9, pass_no=2,
                        method="cheminfo", confidence="High", commentary="re-won", overwrite=True)
    assert _flags(led, a) == (False, False)
    assert _flags(led, led.index[led["peak_id"] == "C"][0]) == (True, True)   # only the row committed
    led = _flagged_ledger()
    L.clear_assignment(led, "A", reason="audit")
    assert _flags(led, a) == (False, False)
    led = _flagged_ledger()
    L.displace_to_isotopologue(led, "C", "A", iso_label="13C")
    c = led.index[led["peak_id"] == "C"][0]
    assert led.at[c, "role"] == L.ROLE_ISO and _flags(led, c) == (False, False)
    assert _flags(led, led.index[led["peak_id"] == "A"][0]) == (True, True)   # the parent keeps its own


def test_the_reset_never_creates_a_column():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["A", "C"], "mz": [200.0, 202.0], "height": [1e5, 5e4]}))
    for pid in ("A", "C"):
        L.commit_assignment(led, pid, neutral_formula="C10H16O4", adduct="[M-H]-", ion_score=0.9, pass_no=1,
                            method="cheminfo", confidence="High", commentary="Pass 1")
    L.displace_to_isotopologue(led, "C", "A", iso_label="13C")
    L.clear_assignment(led, "A", reason="audit")
    assert not (set(FLAG_COLUMNS) & set(led.columns))
    led[BELOW] = True                                       # an older ledger: below only
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-", ion_score=0.9, pass_no=1,
                        method="cheminfo", confidence="High", commentary="Pass 1")
    assert not bool(led.loc[led["peak_id"] == "A", BELOW].iloc[0])     # reset...
    assert LEAD not in led.columns                                      # ...and the lead column not made


def test_the_flag_helpers():
    frame = pd.DataFrame({BELOW: [True, False, None, "False", float("nan")],
                          LEAD: [False, True, None, "True", True]})
    assert list(L.flagged(frame)) == [True, True, False, True, True]
    assert list(L.flagged(frame.drop(columns=[LEAD]))) == [True, False, False, False, False]
    assert not L.flagged(pd.DataFrame({"x": [1]})).any() and not L.has_flags(pd.DataFrame({"x": [1]}))
    bare = pd.DataFrame({"x": [1, 2]})
    assert L.mark_lead(bare, 0) is False and LEAD not in bare.columns
    old = pd.DataFrame({BELOW: [False, False]})
    assert L.mark_lead(old, 1) is True and list(old[LEAD]) == [False, True]
    assert L.ASSIGNABILITY_FLAGS == (BELOW, LEAD)
