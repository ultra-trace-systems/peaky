"""The isotope-contradiction guard (peaky/assignment/isoguard.py): a reading the
evidence scale's own isotope facts contradict is never Assigned -- per file (the
`isotope_guard` stage, on the facts the `evidence` stage computed) and on the
merged ledger (the pooled facts)."""
from __future__ import annotations

import types

import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import isoguard as IG
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.assignment.levels import lines as LN
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

ORBI = 120_000.0
CAL = (0.0, 0.3)
GATE = 100.0


def _ledger(n="C10H16O4", a="[M-H]-", *, c13_count=10.0, snr13=40.0, s34=None, h0=2.0e5):
    """One committed M0 (Assigned) with its 13C line at the height of ``c13_count``
    carbons and, optionally, a 34S line at ``s34`` x the M0; the M0's server S/N
    makes the PREDICTED 13C line S/N (nC x R13C x S/N) equal ``snr13``. A floor of
    faint unexplained peaks sets the file's edge."""
    mz0 = C.ion_mz(n, a)
    nc = EV.count_element(n, "C")
    sn0 = snr13 / (nc * IC.R13C)
    h1 = h0 * c13_count * IC.R13C * ((mz0 + 1.0) / mz0) ** 1.5
    rows = [dict(peak_id="A", mz=mz0, height=h0, area=h0, signal_to_noise=sn0),
            dict(peak_id="B", mz=mz0 + ISO.D_13C_EXACT, height=h1, area=h1, signal_to_noise=sn0 * h1 / h0)]
    if s34 is not None:
        rows.append(dict(peak_id="S", mz=mz0 + ISO.D_34S, height=h0 * s34, area=h0 * s34,
                         signal_to_noise=sn0 * s34))
    rows += [dict(peak_id=f"n{k}", mz=150.0 + 7.1 * k, height=60.0, area=60.0, signal_to_noise=1.0)
             for k in range(12)]
    led = L.new_ledger(pd.DataFrame(rows))
    L.commit_assignment(led, "A", neutral_formula=n, adduct=a,
                        ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "-",
                        ion_score=0.97, compound_score=0.96, ppm_error=0.0, pass_no=1, method="cheminfo",
                        confidence="High", commentary="Pass 1")
    L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.9)
    T.apply_tiers(led)
    led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
    return led


def _inputs():
    return EV.file_run_inputs(sample_id="s1", reagent="NO3", context="ambient-air", resolution=ORBI,
                              height_gate_cps=GATE, degeneracy_cal=CAL)


def _guarded(led):
    """The per-file path: the `evidence` stage's levels (their pair facts kept), then the guard."""
    out = []
    EV.apply_levels(led, run_inputs=_inputs(), pairs_out=out)
    s = IG.guard_ledger(led, out[-1], log=lambda *a: None)
    return led[led["role"] == L.ROLE_M0].iloc[0], s, out[-1]


# --------------------------------------------------------------------------- the 13C count
def test_a_c10_reading_whose_own_13c_line_reads_c6_at_snr_40_is_candidate_with_the_note():
    row, s, pairs = _guarded(_ledger(c13_count=6.0, snr13=40.0))
    assert row["tier"] == "Candidate" and s["demoted"] == 1 and s["by_fact"]["13C"] == 1
    note = [x for x in str(row["tier_reason"]).split(" | ") if x.startswith(IG.NOTE_MARK)]
    assert len(note) == 1 and note[0].startswith("isotope contradiction: 13C reads ") and "C for 10" in note[0]
    # the fact is the levels' own (committed_reasons), not a second computation
    assert note[0] == IG.note_text([pairs["committed_reasons"].iloc[0]])
    # the reading and its level are untouched: only the tier moves
    assert row["neutral_formula"] == "C10H16O4" and row["evidence_level"] == "5b"


def test_the_same_reading_at_a_predicted_13c_snr_of_5_is_untestable_and_untouched():
    row, s, pairs = _guarded(_ledger(c13_count=6.0, snr13=5.0))
    assert row["tier"] == "Assigned" and s["demoted"] == 0
    assert IG.NOTE_MARK not in str(row["tier_reason"]) and not pairs["committed_reasons"].iloc[0]


def test_a_13c_count_inside_the_levels_band_stays_assigned():
    # C10 read as C8: |8 - 10| = 2 <= max(1.5, 0.25 x 10) -- the levels' band (levels.lines.C_TOL_*)
    row, s, _p = _guarded(_ledger(c13_count=8.0, snr13=40.0))
    assert row["tier"] == "Assigned" and s["demoted"] == 0
    row2, s2, _p = _guarded(_ledger(c13_count=7.0, snr13=40.0))
    assert row2["tier"] == "Candidate" and s2["demoted"] == 1


# --------------------------------------------------------------------------- the 34S line
def test_a_sulfur_reading_whose_34s_line_is_predicted_at_snr_15_and_absent_is_candidate():
    # 34S at 0.0447 x the M0 with the M0's S/N = 15 / 0.0447: a line the file shows at S/N 15
    n = "C9H14O4S"
    nc = EV.count_element(n, "C")
    snr13 = 15.0 / 0.0447 * nc * IC.R13C
    row, s, _p = _guarded(_ledger(n, c13_count=9.0, snr13=snr13))
    assert row["tier"] == "Candidate" and s["by_fact"]["absent"] == 1
    assert "isotope contradiction: 34S (" in str(row["tier_reason"]) and " absent in 1/1 files" in str(row["tier_reason"])
    # ... and with the line present at its ratio the reading stays Assigned
    row2, s2, _p = _guarded(_ledger(n, c13_count=9.0, snr13=snr13, s34=0.0447))
    assert row2["tier"] == "Assigned" and s2["demoted"] == 0


def test_a_34s_line_under_snr_10_cannot_refute():
    n = "C9H14O4S"
    nc = EV.count_element(n, "C")
    snr13 = 5.0 / 0.0447 * nc * IC.R13C          # the 34S line would show at S/N 5
    row, s, _p = _guarded(_ledger(n, c13_count=9.0, snr13=snr13, h0=2.0e6))
    assert row["tier"] == "Assigned" and s["demoted"] == 0


# --------------------------------------------------------------------------- the facts' texts
def test_the_guard_reads_the_texts_the_levels_write():
    """One pattern per producer, pinned to the producer's own output: contradiction_a and
    carbon_test are called; test (b)'s text is q1_isotopes' f-string with values."""
    L1 = {"label": "34S", "ratio": 0.0447, "elements": ["S"], "mode": "full"}
    recs = [dict(L=L1, n_test=2, n_bad=2, n_abs=2, n_hi=0, n_lo=0)]
    a = LN.contradiction_a(recs, 1)
    assert a == ["34S (0.0447x) absent in 2/2 files"]
    ctx = types.SimpleNamespace(c13={("C10H16O4", "[M-H]-"): dict(c=6.0, se=float("nan"), n=1, src="per-file 13C, 1 files")})
    cv, ctext = LN.carbon_test(ctx, ("C10H16O4", "[M-H]-"), {"C": 10, "H": 15, "O": 4})
    assert cv == "contradicts"
    b = "37Cl line at 0.33x seen in 3/3 files (in band for C9H15ClO4 [M-H]-), predicts 0x"
    kinds = [f["kind"] for f in IG.facts_of(" | ".join(a + [b, ctext]))]
    assert kinds == ["a", "b", "c"]
    # the three bind; another line's absence (18O) or a too-low 34S line does not
    assert IG.contradiction(" | ".join(a + [b, ctext]), {"C": 10, "H": 15, "O": 4}) == a + [b, ctext]
    assert IG.contradiction("18O (0.00821x) absent in 2/2 files", {"C": 10, "O": 4}) == []
    assert IG.contradiction("34S (0.0447x) too low in 2/2 files", {"C": 9, "S": 1}) == []
    assert IG.contradiction("14N (reagent impurity) (0.0204x) absent in 2/2 files", {"C": 9}) == []


def test_an_unexplained_37cl_line_binds_only_a_reading_without_cl():
    b = "37Cl line at 0.33x seen in 3/3 files (in band for C9H15ClO4 [M-H]-), predicts 0x"
    assert IG.contradiction(b, {"C": 10, "H": 15, "O": 4}) == [b]
    assert IG.contradiction(b, {"C": 9, "H": 15, "O": 4, "Cl": 1}) == []
    s = "34S line at 0.05x seen in 3/3 files (in band for C9H14O4S [M-H]-), predicts 0x"
    assert IG.contradiction(s, {"C": 10, "O": 4}) == [s] and IG.contradiction(s, {"C": 9, "O": 4, "S": 1}) == []
    # another isotope's unexplained line is a level fact only
    assert IG.contradiction("81Br line at 0.97x seen in 3/3 files (in band for X [M-H]-), predicts 0x",
                            {"C": 10}) == []


# --------------------------------------------------------------------------- the stage and the merge
def test_the_stage_sits_right_after_evidence_and_reads_its_facts():
    names = [s.name for s in A._STAGES]
    assert names.index("isotope_guard") == names.index("evidence") + 1
    assert names.index("timeseries") == names.index("isotope_guard") + 1
    led = _ledger(c13_count=6.0, snr13=40.0)
    st = types.SimpleNamespace(led=led, cfg=None, log=lambda *a: None, profile=None, sample_id="s1",
                               level_pairs=[])
    EV.apply_levels(led, run_inputs=_inputs(), pairs_out=st.level_pairs)
    s = A._stage_isotope_guard(st)
    assert s["demoted"] == 1 and led.loc[led["peak_id"] == "A", "tier"].iloc[0] == "Candidate"
    # a file the evidence stage did not level has no facts: nothing moves
    led2 = _ledger(c13_count=6.0, snr13=40.0)
    st2 = types.SimpleNamespace(led=led2, log=lambda *a: None, level_pairs=[])
    assert A._stage_isotope_guard(st2)["demoted"] == 0
    assert led2.loc[led2["peak_id"] == "A", "tier"].iloc[0] == "Assigned"


def test_the_merged_guard_reads_the_pooled_facts_and_only_demotes():
    merged = pd.DataFrame([dict(mz=199.0976, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned",
                                tier_reason=pd.NA),
                           dict(mz=213.0768, neutral_formula="C10H14O5", adduct="[M-H]-", tier="Assigned",
                                tier_reason="vote note"),
                           dict(mz=229.0717, neutral_formula="C10H14O6", adduct="[M-H]-", tier="Candidate",
                                tier_reason=pd.NA)])
    levels = pd.DataFrame([dict(neutral_formula="C10H16O4", adduct="[M-H]-",
                                committed_reasons="13C reads 5.1 C for 10 (tol 2.5, rule C, 40 spectra)"),
                           dict(neutral_formula="C10H14O5", adduct="[M-H]-",
                                committed_reasons="18O (0.0103x) absent in 4/4 files"),
                           dict(neutral_formula="C10H14O6", adduct="[M-H]-",
                                committed_reasons="13C reads 2.0 C for 10 (tol 2.5, rule C, 40 spectra)")])
    s = IG.guard_merged(merged, levels, log=lambda *a: None)
    assert s["demoted"] == 1 and s["tested"] == 2
    assert merged["tier"].tolist() == ["Candidate", "Assigned", "Candidate"]
    assert str(merged.at[0, "tier_reason"]) == "isotope contradiction: 13C reads 5.1 C for 10 (tol 2.5, rule C, 40 spectra)"
    assert merged.at[1, "tier_reason"] == "vote note" and pd.isna(merged.at[2, "tier_reason"])


@pytest.mark.parametrize("frame", [None, pd.DataFrame(columns=["neutral_formula", "adduct"])])
def test_no_facts_change_nothing(frame):
    led = _ledger(c13_count=6.0, snr13=40.0)
    before = led.copy()
    s = IG.guard_ledger(led, frame, log=lambda *a: None)
    assert s["demoted"] == 0
    pd.testing.assert_frame_equal(led, before)


def test_the_merged_row_says_what_the_per_file_guard_held():
    led1 = _ledger(c13_count=6.0, snr13=40.0)
    _guarded(led1)
    led2 = _ledger(c13_count=10.0, snr13=40.0)
    _guarded(led2)
    merged = pd.DataFrame([dict(mz=199.0976, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned",
                                tier_reason=pd.NA, srcs="f1,f2"),
                           dict(mz=213.0768, neutral_formula="C10H14O5", adduct="[M-H]-", tier="Assigned",
                                tier_reason=pd.NA, srcs="f1")])
    assert IG.merged_notes(merged, {"f1": led1, "f2": led2}, log=lambda *a: None) == 1
    t = str(merged.at[0, "tier_reason"])
    assert t.startswith("isotope contradiction in 1 of 2 file(s): 13C reads ") and merged.at[0, "tier"] == "Assigned"
    assert pd.isna(merged.at[1, "tier_reason"])
    # the vote's key is the ION: a batch-level re-read of the same ion (C13H14O4 [M+NH4]+ in the file,
    # C13H17NO4 [M+H]+ on the merged row) carries the file's note; another ion does not
    fact = "13C reads 6.0 C for 13 (tol 3.2, per-file 13C, 1 files)"
    pfl = pd.DataFrame([dict(mz=268.1180, role="M0", neutral_formula="C13H14O4", adduct="[M+NH4]+", tier="Candidate",
                             tier_reason=f"isotope contradiction: {fact}")])
    m4 = pd.DataFrame([dict(mz=268.1180, neutral_formula="C13H17NO4", adduct="[M+H]+", tier="Candidate",
                            tier_reason=pd.NA, srcs="g1"),
                       dict(mz=268.1180, neutral_formula="C14H17NO3", adduct="[M+H]+", tier="Candidate",
                            tier_reason=pd.NA, srcs="g1")])
    assert IG.merged_notes(m4, {"g1": pfl}, log=lambda *a: None) == 1
    assert m4.at[0, "tier_reason"] == f"isotope contradiction in 1 of 1 file(s): {fact}" and pd.isna(m4.at[1, "tier_reason"])


def test_the_batch_merge_holds_a_reading_its_pooled_facts_contradict(tmp_path, monkeypatch):
    """Every file reads the C10 reading's own 13C line at ~C6 (the per-file stage is faked here, so
    the per-file guard does not run): the pooled facts make the merged row Candidate with the note,
    and batch_summary counts it."""
    import json
    from tests.test_evidence_outputs import _run_batch
    _run_batch(tmp_path, monkeypatch, ledger_of=lambda n: _ledger(c13_count=6.0, snr13=40.0))
    summ = json.load(open(tmp_path / "batch_summary.json"))
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    r = merged[(merged["neutral_formula"] == "C10H16O4") & (merged["adduct"] == "[M-H]-")]
    assert len(r) == 1 and r["tier"].iloc[0] == "Candidate"
    assert "isotope contradiction: 13C reads " in str(r["tier_reason"].iloc[0])
    g = summ["merge_gates"]["isotope_guard"]
    assert g["demoted"] == 1 and g["by_fact"]["13C"] == 1 and g["rows"][0]["neutral_formula"] == "C10H16O4"


# =========================================================================== repair after refute (2026-11-05)
def test_the_13c_count_binds_from_the_levels_snr_10():
    # predicted 13C line S/N 15 (>= the levels' 10): tested, contradicted -> Candidate
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=15.0))
    assert row["tier"] == "Candidate" and s["by_fact"]["13C"] == 1
    # S/N 9: untestable -> untouched
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=9.0))
    assert row["tier"] == "Assigned" and s["demoted"] == 0


def test_a_chlorine_reading_whose_37cl_line_is_absent_is_candidate():
    n = "C8H13ClO4"
    nc = EV.count_element(n, "C")
    row, s, pairs = _guarded(_ledger(n, c13_count=8.0, snr13=40.0))
    assert row["tier"] == "Candidate" and s["by_fact"]["absent"] == 1
    assert "isotope contradiction: 37Cl (" in str(row["tier_reason"]) and " absent in 1/1 files" in str(row["tier_reason"])
    assert IG.contradiction("37Cl (0.32x) absent in 2/2 files", {"C": 8, "Cl": 1}) == ["37Cl (0.32x) absent in 2/2 files"]
    assert nc == 8


def test_two_facts_of_one_reading_both_bind():
    """A cell of two facts (the levels join them with ' | '): an S reading whose 13C line reads ~C5 for C9 AND
    whose 34S line is absent at S/N 15 carries both in the note."""
    n = "C9H14O4S"
    nc = EV.count_element(n, "C")
    snr13 = 15.0 / 0.0447 * nc * IC.R13C
    row, s, pairs = _guarded(_ledger(n, c13_count=5.0, snr13=snr13))
    reasons = pairs["committed_reasons"].iloc[0]
    assert " | " in reasons
    t = str(row["tier_reason"])
    assert row["tier"] == "Candidate" and "34S (" in t and "13C reads " in t, t


def test_the_guard_reads_m0_rows_only():
    led = _ledger(c13_count=6.0, snr13=40.0)
    out = []
    EV.apply_levels(led, run_inputs=_inputs(), pairs_out=out)
    i = led.index[led["peak_id"] == "B"][0]           # the 13C child, given the reading and an Assigned tier
    led.at[i, "neutral_formula"], led.at[i, "adduct"], led.at[i, "tier"] = "C10H16O4", "[M-H]-", "Assigned"
    s = IG.guard_ledger(led, out[-1], log=lambda *a: None)
    assert s["demoted"] == 1 and led.at[i, "tier"] == "Assigned"


def test_the_evidence_stage_hands_the_guard_its_facts():
    """The real stage functions in order on one file: `_stage_evidence` keeps the file's pair facts
    (st.level_pairs) and `_stage_isotope_guard` demotes on them."""
    led = _ledger(c13_count=6.0, snr13=40.0)
    st = types.SimpleNamespace(led=led, cfg=types.SimpleNamespace(height_cutoff=GATE, noise_edge_cps=None),
                               reagent_profile="NO3", adducts=["[M-H]-"], sample_id="s1", profile=None,
                               resolving_power=ORBI, reflists_active=None, degeneracy_cal=CAL,
                               reflists_context=None, reagent_halogen=EV.DETECT_HALOGEN, log=lambda *a: None)
    A._stage_evidence(st)
    assert len(st.level_pairs) == 1
    s = A._stage_isotope_guard(st)
    assert s["demoted"] == 1 and led.loc[led["peak_id"] == "A", "tier"].iloc[0] == "Candidate"


def test_an_offline_run_records_the_guard_in_its_stats():
    from tests.test_blocked import _run
    res = _run("isoguard-stats")
    g = res["stats"]["isotope_guard"]
    assert isinstance(g, dict) and set(g) >= {"tested", "demoted", "by_fact", "rows"}
    assert g == res["summaries"]["isotope_guard"] and g["tested"] >= 1
