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


#: the reference acids every test file carries (High confidence, Assigned, their own 13C line at their own
#: count, predicted 13C S/N 40): the reliability gate's proof that the file's 13C count is trustworthy
REF_ACIDS = [f"C{k}H{2 * k - 2}O4" for k in range(12, 24)]


def _ledger(n="C10H16O4", a="[M-H]-", *, c13_count=10.0, snr13=40.0, s34=None, h0=2.0e5, refs=None,
            ref_c13_x=1.0, c13_line=True, side13=()):
    """One committed M0 (Assigned) with its 13C line at the height of ``c13_count``
    carbons and, optionally, a 34S line at ``s34`` x the M0; the M0's server S/N
    makes the PREDICTED 13C line S/N (nC x R13C x S/N) equal ``snr13``. A floor of
    faint unexplained peaks sets the file's edge. ``refs``: the reference acids (default REF_ACIDS), each
    with its 13C line at ``ref_c13_x`` x its own count (1.0: the file's 13C count is trustworthy).
    ``c13_line`` False: no pick at the 13C position; ``side13``: extra unexplained picks (dppm off the 13C
    position, x the predicted 13C area of the formula's own count)."""
    refs = REF_ACIDS if refs is None else refs
    mz0 = C.ion_mz(n, a)
    nc = EV.count_element(n, "C")
    sn0 = snr13 / (nc * IC.R13C)
    h1 = h0 * c13_count * IC.R13C * ((mz0 + 1.0) / mz0) ** 1.5
    rows = [dict(peak_id="A", mz=mz0, height=h0, area=h0, signal_to_noise=sn0)]
    if c13_line:
        rows.append(dict(peak_id="B", mz=mz0 + ISO.D_13C_EXACT, height=h1, area=h1, signal_to_noise=sn0 * h1 / h0))
    hp = h0 * nc * IC.R13C * ((mz0 + 1.0) / mz0) ** 1.5
    t13 = mz0 + ISO.D_13C_EXACT
    rows += [dict(peak_id=f"X{k}", mz=t13 * (1 + d * 1e-6), height=hp * x, area=hp * x, signal_to_noise=sn0 * hp * x / h0)
             for k, (d, x) in enumerate(side13)]
    if s34 is not None:
        rows.append(dict(peak_id="S", mz=mz0 + ISO.D_34S, height=h0 * s34, area=h0 * s34,
                         signal_to_noise=sn0 * s34))
    rows += [dict(peak_id=f"n{k}", mz=150.0 + 7.1 * k, height=60.0, area=60.0, signal_to_noise=1.0)
             for k in range(12)]
    for k, rn in enumerate(refs):
        rm = C.ion_mz(rn, "[M-H]-")
        rc = EV.count_element(rn, "C")
        rs = 40.0 / (rc * IC.R13C)
        r1 = 1.0e5 * rc * ref_c13_x * IC.R13C * ((rm + 1.0) / rm) ** 1.5
        rows += [dict(peak_id=f"R{k}", mz=rm, height=1.0e5, area=1.0e5, signal_to_noise=rs),
                 dict(peak_id=f"R{k}c", mz=rm + ISO.D_13C_EXACT, height=r1, area=r1, signal_to_noise=rs * r1 / 1.0e5)]
    led = L.new_ledger(pd.DataFrame(rows))
    for k, rn in enumerate(refs):
        L.commit_assignment(led, f"R{k}", neutral_formula=rn, adduct="[M-H]-",
                            ion_formula=C.format_formula(EV.ion_composition(rn, "[M-H]-", None)) + "-",
                            ion_score=0.97, compound_score=0.96, ppm_error=0.0, pass_no=1, method="cheminfo",
                            confidence="High", commentary="Pass 1")
        L.attach_isotopologue(led, f"R{k}c", f"R{k}", iso_label="13C", iso_match_score=0.9)
    L.commit_assignment(led, "A", neutral_formula=n, adduct=a,
                        ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "-",
                        ion_score=0.97, compound_score=0.96, ppm_error=0.0, pass_no=1, method="cheminfo",
                        confidence="High", commentary="Pass 1")
    if c13_line:
        L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.9)
    T.apply_tiers(led)
    led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
    return led


def _inputs():
    return EV.file_run_inputs(sample_id="s1", reagent="NO3", context="ambient-air", resolution=ORBI,
                              height_gate_cps=GATE, degeneracy_cal=CAL)


def _pair(pairs, row):
    """The levels' pair record of a ledger row's reading."""
    return pairs[(pairs["neutral_formula"] == row["neutral_formula"]) & (pairs["adduct"] == row["adduct"])].iloc[0]


def _guarded(led, mode="demote"):
    """The per-file path: the `evidence` stage's levels (their pair facts kept), then the guard (the demote
    mode unless asked: the tests of what binds read the tier)."""
    out = []
    EV.apply_levels(led, run_inputs=_inputs(), pairs_out=out)
    s = IG.guard_ledger(led, out[-1], mode=mode, log=lambda *a: None)
    return led[led["role"] == L.ROLE_M0].iloc[0], s, out[-1]


# --------------------------------------------------------------------------- the 13C count
def test_a_c10_reading_whose_own_13c_line_reads_c6_at_snr_40_is_candidate_with_the_note():
    row, s, pairs = _guarded(_ledger(c13_count=6.0, snr13=40.0))
    assert row["tier"] == "Candidate" and s["demoted"] == 1 and s["by_fact"]["13C"] == 1
    note = [x for x in str(row["tier_reason"]).split(" | ") if x.startswith(IG.NOTE_MARK)]
    assert len(note) == 1 and note[0].startswith("isotope contradiction: 13C reads ") and "C for 10" in note[0]
    # the fact is the levels' own (committed_reasons), not a second computation
    assert note[0] == IG.note_text([_pair(pairs, row)["committed_reasons"]])
    # the reading and its level are untouched: only the tier moves
    assert row["neutral_formula"] == "C10H16O4" and row["evidence_level"] == "5b"


def test_the_same_reading_at_a_predicted_13c_snr_of_5_is_untestable_and_untouched():
    row, s, pairs = _guarded(_ledger(c13_count=6.0, snr13=5.0))
    assert row["tier"] == "Assigned" and s["demoted"] == 0
    assert IG.NOTE_MARK not in str(row["tier_reason"]) and not _pair(pairs, row)["committed_reasons"]


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
    st = types.SimpleNamespace(led=led, cfg=types.SimpleNamespace(isotope_guard="demote"), log=lambda *a: None,
                               profile=None, sample_id="s1", level_pairs=[])
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
    # the batch's proof that its pooled 13C counts are trustworthy: 12 Assigned rows whose pooled count agrees
    refs = [f"C{k}H{2 * k - 2}O4" for k in range(12, 24)]
    merged = pd.concat([merged, pd.DataFrame([dict(mz=200.0 + k, neutral_formula=r, adduct="[M-H]-", tier="Assigned",
                                                   tier_reason=pd.NA) for k, r in enumerate(refs)])], ignore_index=True)
    levels = pd.concat([levels.assign(committed_c13=levels["committed_reasons"]),
                        pd.DataFrame([dict(neutral_formula=r, adduct="[M-H]-", committed_reasons="",
                                           committed_c13=f"13C reads {k:.1f} C for {k}")
                                      for k, r in zip(range(12, 24), refs)])], ignore_index=True)
    s = IG.guard_merged(merged, levels, mode="demote", log=lambda *a: None)
    assert s["demoted"] == 1 and s["tested"] == 14 and s["facts_unreliable"] is False and s["n_ref"] == 13
    merged = merged.iloc[:3]
    assert merged["tier"].tolist() == ["Candidate", "Assigned", "Candidate"]
    assert str(merged.at[0, "tier_reason"]) == "isotope contradiction: 13C reads 5.1 C for 10 (tol 2.5, rule C, 40 spectra)"
    assert merged.at[1, "tier_reason"] == "vote note" and pd.isna(merged.at[2, "tier_reason"])


@pytest.mark.parametrize("frame", [None, pd.DataFrame(columns=["neutral_formula", "adduct"])])
def test_no_facts_change_nothing(frame):
    led = _ledger(c13_count=6.0, snr13=40.0)
    before = led.copy()
    s = IG.guard_ledger(led, frame, mode="demote", log=lambda *a: None)
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
    from peaky.assignment import passes as PA
    _run_batch(tmp_path, monkeypatch, ledger_of=lambda n: _ledger(c13_count=6.0, snr13=40.0),
               cfg=PA.PassConfig(isotope_guard="demote"))
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
    reasons = _pair(pairs, row)["committed_reasons"]
    assert " | " in reasons
    t = str(row["tier_reason"])
    assert row["tier"] == "Candidate" and "34S (" in t and "13C reads " in t, t


def test_the_guard_reads_m0_rows_only():
    led = _ledger(c13_count=6.0, snr13=40.0)
    out = []
    EV.apply_levels(led, run_inputs=_inputs(), pairs_out=out)
    i = led.index[led["peak_id"] == "B"][0]           # the 13C child, given the reading and an Assigned tier
    led.at[i, "neutral_formula"], led.at[i, "adduct"], led.at[i, "tier"] = "C10H16O4", "[M-H]-", "Assigned"
    s = IG.guard_ledger(led, out[-1], mode="demote", log=lambda *a: None)
    assert s["demoted"] == 1 and led.at[i, "tier"] == "Assigned"


def test_the_evidence_stage_hands_the_guard_its_facts():
    """The real stage functions in order on one file: `_stage_evidence` keeps the file's pair facts
    (st.level_pairs) and `_stage_isotope_guard` demotes on them."""
    led = _ledger(c13_count=6.0, snr13=40.0)
    st = types.SimpleNamespace(led=led, cfg=types.SimpleNamespace(height_cutoff=GATE, noise_edge_cps=None,
                                                                  isotope_guard="demote"),
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


# =========================================================================== the reliability gate (M1 §12)
def test_a_file_whose_13c_counts_fail_its_own_reference_rows_demotes_nothing():
    """Every reference acid's 13C line reads 0.3x its count (a file whose isotope lines under-respond): the
    file's 13C facts are not trusted -- the C6-read C10 reading stays Assigned and the record says why."""
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=40.0, ref_c13_x=0.3))
    assert row["tier"] == "Assigned" and s["demoted"] == 0
    assert s["facts_unreliable"] is True and s["n_ref"] == 13 and s["agreement"] == 0.0
    assert IG.NOTE_MARK not in str(row["tier_reason"])
    # the same file with trustworthy reference lines: the guard binds, and says how much it trusted
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=40.0))
    assert row["tier"] == "Candidate" and s["facts_unreliable"] is False
    assert s["n_ref"] == 13 and s["n_agree"] == 12 and s["agreement"] == pytest.approx(12 / 13, abs=1e-4)


def test_the_gate_needs_the_agreement_fraction_over_enough_reference_rows():
    def frame(n_agree, n_contra, conf="High"):
        names = [f"C{k}H{2 * k - 2}O4" for k in range(5, 5 + n_agree + n_contra)]
        rows = pd.DataFrame(dict(role="M0", tier="Assigned", confidence=conf, neutral_formula=names, adduct="[M-H]-"))
        c = [f"13C reads {k}.0 C for {k}" for k in range(5, 5 + n_agree)] + \
            [f"13C reads 1.0 C for {k} (tol 2.0, per-file 13C, 1 files)" for k in range(5 + n_agree, 5 + n_agree + n_contra)]
        return rows, pd.DataFrame(dict(neutral_formula=names, adduct="[M-H]-", committed_c13=c))
    assert IG.GATE_MIN_AGREE == 0.9 and IG.GATE_MIN_REF == 10
    assert IG.reliability(*frame(9, 1))["reliable"] is True                 # 0.90 of 10
    assert IG.reliability(*frame(17, 2))["reliable"] is False               # 0.895: just under 0.9
    assert IG.reliability(*frame(8, 1))["reliable"] is False                # 9 rows: too few to prove anything
    assert IG.reliability(*frame(18, 3))["reliable"] is False               # 0.857
    # only the file's High / Good rows prove it (per file); the merged rows carry no confidence
    rows, pairs = frame(9, 1, conf="Low")
    assert IG.reliability(rows, pairs)["n_ref"] == 0 and IG.reliability(rows, pairs, confidence=False)["reliable"]
    # a Candidate row or a carbon-free count is no reference
    rows, pairs = frame(10, 0)
    rows.loc[0, "tier"] = "Candidate"
    pairs.loc[1, "committed_c13"] = "13C reads 0.1 C for 0"
    assert IG.reliability(rows, pairs)["n_ref"] == 8


def test_the_merged_guard_needs_trustworthy_pooled_counts():
    refs = [f"C{k}H{2 * k - 2}O4" for k in range(12, 24)]
    merged = pd.DataFrame([dict(mz=199.0976, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned",
                                tier_reason=pd.NA)] +
                          [dict(mz=200.0 + k, neutral_formula=r, adduct="[M-H]-", tier="Assigned", tier_reason=pd.NA)
                           for k, r in enumerate(refs)])
    bad = [f"13C reads {0.3 * k:.1f} C for {k} (tol {max(1.5, 0.25 * k):.1f}, rule C, 40 spectra)" for k in range(12, 24)]
    levels = pd.DataFrame([dict(neutral_formula="C10H16O4", adduct="[M-H]-",
                                committed_reasons="13C reads 5.1 C for 10 (tol 2.5, rule C, 40 spectra)",
                                committed_c13="13C reads 5.1 C for 10 (tol 2.5, rule C, 40 spectra)")] +
                          [dict(neutral_formula=r, adduct="[M-H]-", committed_reasons=b, committed_c13=b)
                           for r, b in zip(refs, bad)])
    s = IG.guard_merged(merged, levels, mode="demote", log=lambda *a: None)
    assert s["demoted"] == 0 and s["facts_unreliable"] is True and s["agreement"] == 0.0
    assert (merged["tier"] == "Assigned").all()


def test_the_pooled_re_level_obeys_the_gate():
    from peaky.assignment import calpool as CP
    from tests.test_calpool import _batch, _inputs_for
    pool = CP.pool_calibration(_batch())
    led = _ledger(c13_count=6.0, snr13=40.0, ref_c13_x=0.3)
    at = led.copy(deep=True)
    at["tier"] = pd.NA
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1", guard="demote"), audit=[], log=lambda *a: None)
    g = out["isotope_guard"]
    assert g["facts_unreliable"] is True and g["demoted"] == 0 and g["source"] == "pooled"


# =========================================================================== note-only default + per-row conditions (M1 §13)
REC_KEYS = {"mode", "tested", "demoted", "would_demote", "by_fact", "held", "rows", "held_rows", "facts_unreliable",
            "agreement", "n_ref", "n_agree"}


def test_the_default_mode_notes_the_contradiction_and_moves_no_tier():
    assert IG.DEFAULT_MODE == "note" and IG.MODES == ("note", "demote")
    from peaky.assignment import passes as PA
    assert PA.PassConfig().isotope_guard == "note" and IG.mode_of(PA.PassConfig()) == "note" and IG.mode_of(None) == "note"
    row, s, pairs = _guarded(_ledger(c13_count=6.0, snr13=40.0), mode=None)
    assert row["tier"] == "Assigned" and s["mode"] == "note" and s["demoted"] == 0 and s["would_demote"] == 1
    assert s["rows"][0]["neutral_formula"] == "C10H16O4" and s["by_fact"]["13C"] == 1
    assert "isotope contradiction: 13C reads " in str(row["tier_reason"])
    # control: the demote mode moves the same row
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=40.0), mode="demote")
    assert row["tier"] == "Candidate" and s["mode"] == "demote" and s["demoted"] == s["would_demote"] == 1
    with pytest.raises(ValueError):
        IG.resolve_mode("strict")


def test_the_stage_runs_in_the_runs_mode():
    from peaky.assignment import passes as PA
    for cfg, tier in ((PA.PassConfig(), "Assigned"), (None, "Assigned"), (PA.PassConfig(isotope_guard="demote"), "Candidate")):
        led = _ledger(c13_count=6.0, snr13=40.0)
        st = types.SimpleNamespace(led=led, cfg=cfg, log=lambda *a: None, level_pairs=[])
        EV.apply_levels(led, run_inputs=_inputs(), pairs_out=st.level_pairs)
        s = A._stage_isotope_guard(st)
        assert led.loc[led["peak_id"] == "A", "tier"].iloc[0] == tier and s["would_demote"] == 1
        assert IG.NOTE_MARK in str(led.loc[led["peak_id"] == "A", "tier_reason"].iloc[0])


def test_the_record_has_one_shape_at_every_call_site():
    led = _ledger(c13_count=6.0, snr13=40.0)
    assert set(IG.guard_ledger(led, None, log=lambda *a: None)) == REC_KEYS          # no levels
    assert set(_guarded(_ledger(c13_count=6.0))[1]) == REC_KEYS                      # binding
    assert set(_guarded(_ledger(c13_count=6.0, ref_c13_x=0.3))[1]) == REC_KEYS       # gate closed
    assert set(_guarded(_ledger(c13_count=10.0))[1]) == REC_KEYS                     # nothing contradicted
    assert set(IG.guard_merged(pd.DataFrame(columns=["tier"]), None, log=lambda *a: None)) == REC_KEYS
    s = IG.guard_ledger(led, None, log=lambda *a: None)
    assert s["facts_unreliable"] is True and s["n_ref"] == 0 and s["agreement"] is None and s["mode"] == "note"
    s = _guarded(_ledger(c13_count=6.0, ref_c13_x=0.3))[1]
    # the reading and the 12 references (their own counts read 0.3x) are all held by the gate
    assert s["held"]["gate"] == 13 and s["held_rows"][0]["condition"] == "gate" and s["would_demote"] == 0


# --------------------------------------------------------------------------- the per-row conditions
def test_a_13c_line_shadowed_by_a_pick_6_ppm_off_is_a_blend_not_a_contradiction():
    """The C10 reading's 13C line is not within the levels' 3-ppm window (its count reads ~0), but a pick 6 ppm off
    carries 1.0x the predicted 13C area: an unresolved neighbour, so the count does not bind."""
    row, s, pairs = _guarded(_ledger(c13_line=False, side13=[(6.0, 1.0)], snr13=40.0))
    assert "(tol" in str(_pair(pairs, row)["committed_c13"])                # the levels do read a contradiction
    assert row["tier"] == "Assigned" and s["demoted"] == 0 and s["held"]["blend"] == 1
    assert s["held_rows"][0]["condition"] == "blend" and IG.NOTE_MARK not in str(row["tier_reason"])
    # the note mode applies the same conditions: a blend is neither noted nor counted as a would-be demotion
    row, s, _p = _guarded(_ledger(c13_line=False, side13=[(6.0, 1.0)], snr13=40.0), mode="note")
    assert s["would_demote"] == 0 and s["held"]["blend"] == 1 and IG.NOTE_MARK not in str(row["tier_reason"])
    row, s, _p = _guarded(_ledger(c13_line=False, side13=[(6.0, 0.3)], snr13=40.0), mode="note")
    assert s["would_demote"] == 1 and row["tier"] == "Assigned" and IG.NOTE_MARK in str(row["tier_reason"])
    # controls: the same pick at 0.3x the predicted area, or 12 ppm off, is no blend: the count binds
    for side in ([(6.0, 0.3)], [(12.0, 1.0)]):
        row, s, _p = _guarded(_ledger(c13_line=False, side13=side, snr13=40.0))
        assert row["tier"] == "Candidate" and s["held"]["blend"] == 0, side
    # the edges: just inside 3..10 ppm at exactly 0.5x is a blend
    row, s, _p = _guarded(_ledger(c13_line=False, side13=[(9.9, 0.5)], snr13=40.0))
    assert row["tier"] == "Assigned" and s["held"]["blend"] == 1
    assert IG.BLEND_PPM_MIN == IC.C_TOL_PPM and IG.BLEND_PPM_MAX == 10.0 and IG.BLEND_MIN_AREA == 0.5


def test_a_missing_13c_line_under_predicted_snr_30_is_untested():
    row, s, _p = _guarded(_ledger(c13_line=False, snr13=20.0))
    assert row["tier"] == "Assigned" and s["held"]["untested"] == 1 and s["demoted"] == 0
    # controls: missing at predicted S/N 40 binds; present (read C6) at S/N 20 binds -- a measured low count
    row, s, _p = _guarded(_ledger(c13_line=False, snr13=40.0))
    assert row["tier"] == "Candidate" and s["held"]["untested"] == 0
    row, s, _p = _guarded(_ledger(c13_count=6.0, snr13=20.0))
    assert row["tier"] == "Candidate" and s["held"]["untested"] == 0
    assert IG.UNTESTED_MAX_SNR13 == 30.0


def test_the_per_row_conditions_lift_only_the_13c_fact():
    """An S reading whose 13C line is missing at predicted S/N 26 (untested) AND whose 34S line is absent at S/N 12:
    the 34S fact still binds, and the note carries it alone."""
    n = "C9H14O4S"
    nc = EV.count_element(n, "C")
    snr13 = 12.0 / 0.0447 * nc * IC.R13C           # the 34S line would show at S/N 12 (>= the levels' 10)
    assert snr13 < IG.UNTESTED_MAX_SNR13
    row, s, pairs = _guarded(_ledger(n, c13_line=False, snr13=snr13))
    reasons = _pair(pairs, row)["committed_reasons"]
    assert "13C reads" in reasons and "34S (" in reasons
    t = str(row["tier_reason"])
    assert row["tier"] == "Candidate" and "34S (" in t and "13C reads" not in t and s["by_fact"]["absent"] == 1


def test_the_line_view_reads_any_pick_but_the_row_itself():
    led = _ledger(c13_line=False, side13=[(6.0, 1.0)], snr13=40.0)
    i = led.index[led["peak_id"] == "A"][0]
    v = IG.LineView(led)
    assert v.condition(i, "C10H16O4", "[M-H]-") == "blend"
    assert v.condition(i, "C0H2O", "[M-H]-") is None                     # no carbon: no 13C condition
    led2 = led.drop(columns=["area"])
    assert IG.LineView(led2).condition(i, "C10H16O4", "[M-H]-") is None   # no areas: no blend can be shown


def _merged_case(per_file):
    """A merged C10 row from the files of ``per_file`` ({sid: ledger}) whose POOLED count contradicts it, and 12
    agreeing pooled references (the batch gate passes)."""
    refs = [f"C{k}H{2 * k - 2}O4" for k in range(12, 24)]
    led0 = next(iter(per_file.values()))
    mz0 = float(led0.loc[led0["peak_id"] == "A", "mz"].iloc[0])
    merged = pd.DataFrame([dict(mz=mz0, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned",
                                tier_reason=pd.NA, srcs=",".join(per_file))] +
                          [dict(mz=200.0 + k, neutral_formula=r, adduct="[M-H]-", tier="Assigned", tier_reason=pd.NA,
                                srcs="") for k, r in enumerate(refs)])
    c = "13C reads 0.1 C for 10 (tol 2.5, per-file 13C, 3 files)"
    levels = pd.DataFrame([dict(neutral_formula="C10H16O4", adduct="[M-H]-", committed_reasons=c, committed_c13=c)] +
                          [dict(neutral_formula=r, adduct="[M-H]-", committed_reasons="",
                                committed_c13=f"13C reads {k:.1f} C for {k}") for k, r in zip(range(12, 24), refs)])
    return merged, levels


@pytest.mark.parametrize("kinds,held", [(("blend", "blend"), True), (("blend", "clean"), True),
                                         (("blend", "clean", "clean"), False), (("clean", "clean"), False),
                                         (("untested", "untested", "clean"), True)])
def test_the_merged_guard_reads_the_condition_in_its_source_files(kinds, held):
    make = {"blend": lambda: _ledger(c13_line=False, side13=[(6.0, 1.0)], snr13=40.0),
            "untested": lambda: _ledger(c13_line=False, snr13=20.0),
            "clean": lambda: _ledger(c13_line=False, snr13=40.0)}
    per_file = {f"f{k}": make[x]() for k, x in enumerate(kinds)}
    merged, levels = _merged_case(per_file)
    s = IG.guard_merged(merged, levels, ledgers=per_file, mode="demote", log=lambda *a: None)
    assert s["facts_unreliable"] is False
    assert (merged.at[0, "tier"] == "Assigned") is held and s["demoted"] == (0 if held else 1)
    if held:
        assert sum(s["held"].values()) == 1 and s["held_rows"][0]["condition"] in ("blend", "untested")
    # without the files the merged row binds (no geometry to read)
    merged, levels = _merged_case(per_file)
    assert IG.guard_merged(merged, levels, mode="demote", log=lambda *a: None)["demoted"] == 1


# --------------------------------------------------------------------------- the gate, pinned at its use site
def _gate_frame(n_agree, n_contra, conf="High", adducts=("[M-H]-",)):
    names = [f"C{k}H{2 * k - 2}O4" for k in range(5, 5 + n_agree + n_contra)]
    rows, pairs = [], []
    for j, nm in enumerate(names):
        k = 5 + j
        for ad in adducts:
            rows.append(dict(role="M0", tier="Assigned", confidence=conf, neutral_formula=nm, adduct=ad, mz=100.0 + j,
                             tier_reason=pd.NA))
    for ad in adducts:
        for j, nm in enumerate(names):
            k = 5 + j
            c = f"13C reads {k}.0 C for {k}" if j < n_agree else f"13C reads 1.0 C for {k} (tol 2.0, per-file 13C, 1 files)"
            pairs.append(dict(neutral_formula=nm, adduct=ad, committed_c13=c, committed_reasons=""))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def test_the_gate_needs_ten_reference_rows_even_when_all_agree():
    assert IG.reliability(*_gate_frame(10, 0))["reliable"] is True
    assert IG.reliability(*_gate_frame(9, 0))["reliable"] is False      # 9 rows at 1.0: too few
    assert IG.reliability(*_gate_frame(5, 0))["reliable"] is False


def test_good_confidence_rows_are_references_too():
    g = IG.reliability(*_gate_frame(9, 1, conf="Good"))
    assert g["n_ref"] == 10 and g["reliable"] is True
    rows, pairs = _gate_frame(10, 0)
    rows.loc[:4, "confidence"] = "Good"
    rows.loc[5:, "confidence"] = "High"
    assert IG.reliability(rows, pairs)["n_ref"] == 10


def test_the_per_file_gate_counts_only_high_and_good_references():
    """12 High references agree; 4 Low-confidence Assigned rows are contradicted. The file's gate reads its High /
    Good rows only (12/12): it opens, and the demote mode takes the 4 rows. Counting the Low rows (12/16 = 0.75)
    would close it."""
    rows, pairs = _gate_frame(12, 0)
    low, lp = _gate_frame(0, 4, conf="Low")
    low["neutral_formula"] = [f"C{k}H{2 * k}O5" for k in range(30, 34)]
    lp["neutral_formula"] = low["neutral_formula"].tolist()
    lp["committed_reasons"] = lp["committed_c13"]
    led = pd.concat([rows, low], ignore_index=True)
    pf = pd.concat([pairs, lp], ignore_index=True)
    assert IG.reliability(led, pf, confidence=False)["reliable"] is False
    s = IG.guard_ledger(led, pf, mode="demote", log=lambda *a: None)
    assert s["facts_unreliable"] is False and s["n_ref"] == 12 and s["demoted"] == 4
    assert (led.iloc[12:]["tier"] == "Candidate").all()


def test_an_unreliable_file_applies_none_of_its_facts():
    """The file's references contradict their 13C counts (gate closed); a sulfur reading's 34S line is absent --
    a fact the gate does not measure -- and still nothing binds in that file, in either mode."""
    rows, pairs = _gate_frame(2, 10)
    rows = pd.concat([rows, pd.DataFrame([dict(role="M0", tier="Assigned", confidence="High",
                                               neutral_formula="C9H14O4S", adduct="[M-H]-", mz=209.05,
                                               tier_reason=pd.NA)])], ignore_index=True)
    pairs = pd.concat([pairs, pd.DataFrame([dict(neutral_formula="C9H14O4S", adduct="[M-H]-", committed_c13="",
                                                 committed_reasons="34S (0.0447x) absent in 1/1 files")])],
                      ignore_index=True)
    for mode in ("demote", "note"):
        led = rows.copy()
        s = IG.guard_ledger(led, pairs, mode=mode, log=lambda *a: None)
        assert s["facts_unreliable"] is True and s["demoted"] == 0 and s["would_demote"] == 0
        assert (led["tier"] == "Assigned").all() and led["tier_reason"].isna().all()
        assert s["held"]["gate"] == 1 and s["held_rows"][0]["neutral_formula"] == "C9H14O4S"
    # control: the same 34S row in a file whose gate opens is Candidate
    rows2, pairs2 = _gate_frame(12, 0)
    led = pd.concat([rows2, rows.iloc[[-1]]], ignore_index=True)
    s = IG.guard_ledger(led, pd.concat([pairs2, pairs.iloc[[-1]]], ignore_index=True), mode="demote", log=lambda *a: None)
    assert s["demoted"] == 1 and led.iloc[-1]["tier"] == "Candidate"


def test_the_reference_count_is_keyed_by_the_reading_not_the_formula():
    """Twelve neutrals, each Assigned twice: as [M-H]- its count agrees, as [M+NO3]- it contradicts (the pairs
    frame lists the [M-H]- texts last). The gate reads each reading's own count: 12 of 24 agree, closed."""
    rows, _p = _gate_frame(12, 0, adducts=("[M-H]-", "[M+NO3]-"))
    _a, agree = _gate_frame(12, 0)
    _b, contra = _gate_frame(0, 12)
    contra["adduct"] = "[M+NO3]-"
    contra["neutral_formula"] = agree["neutral_formula"].tolist()
    g = IG.reliability(rows, pd.concat([contra, agree], ignore_index=True))
    assert g["n_ref"] == 24 and g["n_agree"] == 12 and g["reliable"] is False


def test_a_failing_file_does_not_switch_off_the_merged_guard(tmp_path, monkeypatch):
    """The batch's pooled facts contradict the C10 reading and its pooled counts pass the batch gate; one file's own
    gate failed (its record says so). The merged guard reads only the pooled counts: it still binds."""
    import json
    from peaky.assignment import passes as PA
    from tests.test_evidence_outputs import _run_batch
    _run_batch(tmp_path, monkeypatch, ledger_of=lambda n: _ledger(c13_count=6.0, snr13=40.0),
               cfg=PA.PassConfig(isotope_guard="demote"),
               stats_extra={"isotope_guard": {"facts_unreliable": True, "demoted": 0}})
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert any((p.get("isotope_guard") or {}).get("facts_unreliable") is True for p in summ["per_file"])
    g = summ["merge_gates"]["isotope_guard"]
    assert g["mode"] == "demote" and g["demoted"] == 1 and g["facts_unreliable"] is False


def test_the_batch_defaults_to_the_note_mode_and_hands_the_mode_to_every_file(tmp_path, monkeypatch):
    import json
    from peaky.assignment import passes as PA
    from tests.test_evidence_outputs import _run_batch
    _run_batch(tmp_path, monkeypatch, ledger_of=lambda n: _ledger(c13_count=6.0, snr13=40.0))
    summ = json.load(open(tmp_path / "batch_summary.json"))
    g = summ["merge_gates"]["isotope_guard"]
    assert g["mode"] == "note" and g["demoted"] == 0 and g["would_demote"] == 1 and set(g) >= REC_KEYS
    # the mode sits beside the episode evidence's in batch_summary
    assert summ["isotope_guard_mode"] == "note" and "episode_evidence_mode" in summ
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    r = merged[(merged["neutral_formula"] == "C10H16O4") & (merged["adduct"] == "[M-H]-")]
    assert r["tier"].iloc[0] == "Assigned" and "isotope contradiction: 13C reads " in str(r["tier_reason"].iloc[0])
    seen = []
    _run_batch(tmp_path / "d", monkeypatch, seen=seen, cfg=PA.PassConfig(isotope_guard="demote"))
    assert seen and all(kw["cfg"].isotope_guard == "demote" for kw in seen)
    assert json.load(open(tmp_path / "d" / "batch_summary.json"))["isotope_guard_mode"] == "demote"


def test_the_cli_flag_reaches_the_config_and_the_manifest():
    from peaky import cli
    from peaky.reporting import provenance as PV
    p = cli.build_parser()
    a = p.parse_args(["batch", "--batch", "b", "--reagent", "NO3", "--isotope-guard", "demote"])
    assert a.isotope_guard == "demote" and cli._sidelobe_cfg(a)["cfg"].isotope_guard == "demote"
    a = p.parse_args(["batch", "--batch", "b", "--reagent", "NO3"])
    assert a.isotope_guard is None and cli._sidelobe_cfg(a) == {}
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--batch", "b", "--reagent", "NO3", "--isotope-guard", "strict"])
    from peaky.assignment import passes as PA
    man = PV.build_manifest(run_dir="/tmp", batch_name="b", dataset=None, sample_ids=[], reagent="NO3",
                            cfg=PA.PassConfig(isotope_guard="demote"))
    assert man["config"]["isotope_guard"] == "demote"


def test_the_pooled_re_level_takes_the_cfgs_mode():
    from peaky.assignment import calpool as CP
    from tests.test_calpool import _batch, _inputs_for
    pool = CP.pool_calibration(_batch())
    led = _ledger(c13_count=6.0, snr13=40.0)
    at = led.copy(deep=True)
    at["tier"] = pd.NA
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1"), audit=[], log=lambda *a: None)     # cfg None: note
    g = out["isotope_guard"]
    assert g["mode"] == "note" and g["demoted"] == 0 and g["would_demote"] == 1
    assert led[led["role"] == "M0"].iloc[0]["tier"] == "Assigned"


def test_the_batch_merge_reads_the_per_row_condition_in_its_files(tmp_path, monkeypatch):
    """Every file's 13C line of the C10 reading is shadowed: no pick within 3 ppm (the pooled count reads ~0) but a
    pick 6 ppm off at 1.0x the predicted line. The merged guard reads that in the files: a blend, the row stays
    Assigned in the demote mode (control: the files' line simply missing at S/N 40 binds)."""
    import json
    from peaky.assignment import passes as PA
    from tests.test_evidence_outputs import _run_batch
    for side, held in (([(6.0, 1.0)], True), ((), False)):
        out = tmp_path / ("b" if held else "c")
        _run_batch(out, monkeypatch, ledger_of=lambda n: _ledger(c13_line=False, side13=side, snr13=40.0),
                   cfg=PA.PassConfig(isotope_guard="demote"))
        g = json.load(open(out / "batch_summary.json"))["merge_gates"]["isotope_guard"]
        merged = pd.read_csv(out / "merged_ledger.csv")
        r = merged[(merged["neutral_formula"] == "C10H16O4") & (merged["adduct"] == "[M-H]-")]
        assert (r["tier"].iloc[0] == "Assigned") is held, (side, g)
        assert (g["held"]["blend"] == 1) is held and g["demoted"] == (0 if held else 1)
