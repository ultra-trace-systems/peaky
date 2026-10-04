"""The evidence scale's decision (peaky.assignment.levels.decide / split /
routes / lists / source): one passing case and one mutant per level (3c, 4a,
4b, 5a, 5b, the reagent bucket, NA) on small synthetic sources, plus the step
rules the levels read (the split, the below setters, the series exclusion, the
lead and context texts, the claims). A mutant is the minimal change that must
move the level. Synthetic, offline."""
import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment.levels import competitors as CP
from peaky.assignment.levels import decide as DC
from peaky.assignment.levels import lists as LS
from peaky.assignment.levels import routes as RT
from peaky.assignment.levels import scale as SC
from peaky.assignment.levels import source as SRC
from peaky.assignment.levels import space as SP
from peaky.assignment.levels import split as SPL
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution

ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()     # R(200) = 100 000
TOF = Resolution(coef=200 / 9000 / 200 ** 1.0, exponent=1.0, offset=0.0).as_dict()  # R(200) = 9 000
NO3 = ["[M+NO3]-", "[M-H]-"]


# --------------------------------------------------------------------------- a hand-built source
class _Ctx:
    """The slice of a RunContext the decision reads (nitrate profile, unlabelled)."""

    def __init__(self, decomp=NO3):
        self.space = SP.Space("NO3", "ambient-air", [], ())
        self.decomp_adducts = list(decomp)
        self.engine_channels = list(NO3)
        self.labelled = False
        self.alien = frozenset()

    def decompositions(self, counts, n, a, adducts=None):
        return SP.decompositions(self.space, counts, n, a, self.decomp_adducts if adducts is None else adducts)


class _NoLists:
    """Context lists with no entry."""

    def hits(self, neutral):
        return []

    def mode_flag(self, *a):
        return ""

    def context(self, hits):
        return ""

    def context_source(self, hits):
        return "registry: pass-0 registry (negative)"


ROW = dict(ion="", height=1e5, n_files_obs=3, iso_veto=False, label_veto=False, lowconf=False, below=False,
           ion_only=False, tied=False, committed_contradicted=False, committed_reasons="", no_comp_info=False,
           ion_only_reading=False, lead_fact=False, committed_matched="", committed_matched_lines="",
           committed_plausible=True, n_excl_iso=0, n_competitors=0)


def _prepared(rows, *, comps=(), lists=None, below=None, lead_by=None, decomp=NO3):
    P = pd.DataFrame([{**ROW, "mz": C.ion_mz(r["neutral_formula"], r["adduct"]), **r} for r in rows])
    P["n_competitors"] = [sum(1 for c in comps if (c[0], c[1]) == (n, a)) for n, a in
                          zip(P["neutral_formula"], P["adduct"])]
    cm = pd.DataFrame([dict(neutral_formula=c[0], adduct=c[1], competitor=f"{c[2]} {c[3]}", kind="mass", ppm=0.1,
                            status=c[4], how="isotopes" if c[4] == "excluded" else "", why="", window="run",
                            comp_neutral=c[2], comp_adduct=c[3]) for c in comps], columns=list(CP.COMP_COLUMNS))
    P["n_excl_iso"] = [int(((cm["neutral_formula"] == n) & (cm["adduct"] == a) & (cm["status"] == "excluded")).sum())
                       for n, a in zip(P["neutral_formula"], P["adduct"])]
    led = pd.DataFrame(dict(role="M0", neutral_formula=P["neutral_formula"], adduct=P["adduct"], method="grid",
                            height=1e5, mz=P["mz"]))
    return DC.Prepared(name="t", P=P, comps=cm, pf={"f1": led}, ctx=_Ctx(decomp), lists=lists or _NoLists(), ts=None,
                       tol_ppm=1.0, pol="negative", run_classes=DC.run_classes_of(NO3), below=below or {},
                       alien=frozenset(), arm=False, skip_m0=frozenset(), gate=None, lead_by=lead_by)


def _level(S, partners=None):
    res = DC.inpass(S, partners)
    texts = DC.inpass_texts(S, res)
    lv = DC.relevel(S, res)
    rec = DC.records(S, res, texts, lv)
    rec["claim"] = rec["evidence_level"].map(SC.claim_class)
    return rec


def _one(row, **kw):
    return _level(_prepared([row], **kw)).iloc[0]


NITROPHENOL = dict(neutral_formula="C6H5NO3", adduct="[M-H]-", committed_matched="C", committed_matched_lines="13C")
ACID = dict(neutral_formula="C10H16O4", adduct="[M-H]-", committed_matched="C", committed_matched_lines="13C")


# --------------------------------------------------------------------------- 3c
def test_3c_pinned_and_a_named_entry():
    lists = LS.ContextLists("negative", "ambient-air", [])        # the pass-0 registry: nitrophenol is named
    r = _one(NITROPHENOL, lists=lists)
    assert r.evidence_level == "3c" and r.claim == "identified"
    assert r.split_pinned and r.split_how.startswith("only decomposition [pinned only by the context window: C6H4")
    assert r.named_list == "registry:nitroaromatic = nitrophenol"
    assert r.context == "registry:nitroaromatic = nitrophenol"
    assert r.context_source == "registry:nitroaromatic: pass-0 registry (negative)"
    assert r.would_lift == "level 2 (MS2 / standards) is not automatic"
    assert r.evidence.startswith("3c (split pinned + named context-list entry) · ion: unique in the calibrated window")
    assert " · context source: registry:nitroaromatic: pass-0 registry (negative) · tags: " in r.evidence
    assert "pinned only by the context window" in r.tag_kinds.split("|")


def test_3c_mutant_no_named_entry_is_4a():
    r = _one(NITROPHENOL)
    assert r.evidence_level == "4a" and r.claim == "neutral"
    assert r.would_lift == "3c needs a NAMED context-list entry naming the neutral"


def test_3c_needs_no_positive_fact_and_a_class_entry_is_a_tag():
    lists = LS.ContextLists("negative", "ambient-air", [])
    assert _one(dict(NITROPHENOL, committed_matched="", committed_matched_lines=""), lists=lists).evidence_level == "3c"
    pfba = dict(neutral_formula="C4HF7O2", adduct="[M-H]-")                 # perfluoroacid: a class-scope family
    r = _one(pfba, lists=lists)
    assert r.evidence_level != "3c" and "class list" in r.tag_kinds.split("|")
    assert r.context == "registry:perfluoroacid"


# --------------------------------------------------------------------------- 4a / 4b
def test_4a_pinned_and_an_own_isotope_line():
    r = _one(ACID)
    assert r.evidence_level == "4a"
    # pinned with the side channels locked; formate / acetate would open it (the note names them)
    assert r.split_how == ("only decomposition; side channels locked (acetate, formate would open it: formate: "
                           "C9H14O2 [M+CHO2]-; acetate: C8H12O2 [M+C2H3O2]-)")
    assert "side channels locked (split)" in r.tag_kinds.split("|")
    assert r.positive_fact == "own in-band isotope line(s) 13C (C of the neutral)"
    assert r.evidence.startswith("4a (split pinned + positive fact) · ion: unique in the calibrated window; own lines "
                                 "in band: 13C · split: PINNED -- only decomposition; side channels locked (")
    assert " · positive fact: own in-band isotope line(s) 13C (C of the neutral) · named list: none · " in r.evidence


def test_4a_mutant_no_positive_fact_is_4b():
    r = _one(dict(ACID, committed_matched="", committed_matched_lines=""))
    assert r.evidence_level == "4b" and r.claim == "ion"
    assert r.would_lift == ("4a needs a positive fact (an own in-band isotope line of the neutral's elements, the 15N "
                            "label, or NH4 tracking); 3c needs a named context-list entry")


def test_4a_mutant_an_element_the_neutral_lacks_is_no_positive_fact():
    r = _one(dict(ACID, committed_matched="N", committed_matched_lines="15N"))   # N rides on the adduct only
    assert r.evidence_level == "4b"


def test_4b_split_open():
    r = _one(dict(ACID, adduct="[M+NO3]-"))
    assert r.evidence_level == "4b" and not r.split_pinned
    assert r.split_how == "2 decompositions: C10H16O4 [M+NO3]-; C10H17NO7 [M-H]-"
    assert r.would_lift == "split not pinned: 2 decompositions: C10H16O4 [M+NO3]-; C10H17NO7 [M-H]-"
    assert " · split: open -- 2 decompositions" in r.evidence


def test_4b_mutant_a_competitor_left_is_5a():
    r = _one(dict(ACID, adduct="[M+NO3]-"), comps=[("C10H16O4", "[M+NO3]-", "C9H12O5", "[M+NO3]-", "left")])
    assert r.evidence_level == "5a" and r.claim == "tentative"
    assert r.competitors_left == "C9H12O5 [M+NO3]-"
    assert r.would_lift == "competitors left: C9H12O5 [M+NO3]-"
    assert r.evidence.startswith("5a · competitors left (1) · ion: 1 competitor(s); 1 left")


def test_ion_only_reads_4b_whatever_the_split():
    r = _one(dict(ACID, adduct="[M]-.", ion_only=True, ion_only_reading=True))
    assert r.evidence_level == "4b" and r.split_how == "ion-only channel (process open)"
    assert r.would_lift == "ion-only channel: the neutral and the process stay open"


# --------------------------------------------------------------------------- 5a
def test_5a_mutant_the_competitor_excluded_establishes_the_ion():
    comps = [("C10H16O4", "[M-H]-", "C9H12O5", "[M-H]-", "excluded")]
    r = _one(ACID, comps=comps)
    assert r.evidence_level == "4a" and r.competitors_left == ""
    assert "1 competitor(s); isotopes exclude 1; none left" in r.evidence
    r = _one(ACID, comps=[c[:4] + ("left",) for c in comps])
    assert r.evidence_level == "5a"


def test_5a_lists_every_competitor_left_and_cuts_would_lift_at_six():
    comps = [("C10H16O4", "[M-H]-", f"C{k}H12O5", "[M-H]-", "left") for k in range(5, 13)]
    r = _one(ACID, comps=comps)
    assert r.competitors_left.count("; ") == 7
    assert r.would_lift.endswith(" (+2 more)")


# --------------------------------------------------------------------------- 5b
def test_5b_rejected_and_its_mutant():
    r = _one(dict(ACID, lowconf=True, iso_veto=True))
    assert r.evidence_level == "5b" and r.would_lift == "refuted: iso_veto; lowconf"
    assert r.evidence.startswith("5b · rejected: iso_veto; lowconf · ion: ")
    assert _one(ACID).evidence_level == "4a"                                  # the mutant: no rejection


def test_5b_own_isotopes_and_untestable():
    r = _one(dict(ACID, committed_contradicted=True, committed_reasons="13C (0.11x) absent in 3/3 files"))
    assert r.evidence_level == "5b" and r.would_lift == "refuted: own isotopes: 13C (0.11x) absent in 3/3 files"
    assert "[committed contradicted]" in r.evidence
    r = _one(dict(ACID, no_comp_info=True))
    assert r.evidence_level == "5b" and r.would_lift == "nothing could be enumerated or tested"


def test_below_setters_o11_alone_is_a_tag_any_other_or_none_rejects():
    k = ("C10H16O4", "[M-H]-")
    r = _one(dict(ACID, below=True), below={k: dict(setters=[DC.O11_SETTER])})
    assert r.evidence_level == "4a" and DC.O11_NOTE in r.tags.split(" | ")
    r = _one(dict(ACID, below=True), below={k: dict(setters=[DC.O11_SETTER, "O-monster"])})
    assert r.evidence_level == "5b" and r.would_lift == f"refuted: below: {DC.O11_SETTER} & O-monster"
    r = _one(dict(ACID, below=True), below={})
    assert r.would_lift == "refuted: below: setter not found"


def test_classify_below_row_reads_the_setter_texts():
    assert DC.classify_below_row("below-assignability (O>=11, mass-saturated)", "") == \
        ("implausible", [DC.O11_SETTER], [])
    assert DC.classify_below_row(float("nan"), "unconfirmed fluorine")[1] == ["unconfirmed fluorine"]
    assert DC.classify_below_row("near-tie: best alternative X", None)[0] == "degeneracy/tie"
    assert DC.classify_below_row("", "")[0] == "unclassified"


# --------------------------------------------------------------------------- the reagent bucket
def test_reagent_bucket_beats_a_veto_and_its_mutant():
    r = _one(dict(neutral_formula="HNO3", adduct="[M+NO3]-", lowconf=True))
    assert r.evidence_level == "reagent" and r.claim == "reagent"
    assert r.would_lift == "reagent ion / reagent cluster (not levelled)"
    assert r.evidence.startswith("reagent bucket · reagent identity (HNO3) · ion: ")
    r = _one(dict(neutral_formula="HNO3", adduct="[M+NO3]-", lowconf=True), decomp=NO3)
    assert DC.reagent_identity("HNO3", "[M+Cl]-", "negative") == ""          # the mutant: not a reagent adduct


@pytest.mark.parametrize("n,a,pol,want", [("H2O", "[M+NO3]-", "negative", "H2O"), ("H2O", "[M-H]-", "negative", ""),
                                          ("H2N2O6", "[M-H]-", "negative", "2HNO3"),
                                          ("CH6N2O2", "[M+H]+", "positive", "urea + H2O"),
                                          ("C2H6O", "[M+H]+", "positive", "")])
def test_reagent_identity(n, a, pol, want):
    assert DC.reagent_identity(n, a, pol) == want


# --------------------------------------------------------------------------- NA
def _file(rows):
    led = L.new_ledger(pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(rows))],
                                     "mz": [C.ion_mz(n, a) for n, a in rows], "height": [1e5] * len(rows)}))
    for i, (n, a) in enumerate(rows):
        L.commit_assignment(led, f"p{i}", neutral_formula=n, adduct=a, ion_formula=f"{n}{a[-1]}", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="x")
    return led


def _summary(resolution):
    return dict(reagent="NO3", context="ambient-air", reflists_active=[], resolution=resolution,
                per_file=[dict(sample_id="f1", height_gate_cps=1e3, degeneracy_cal={"mu": 0.0, "sigma": 0.3})])


def test_na_on_a_tof_class_source_before_any_scale_work(monkeypatch):
    """NA before the run context, the enumeration, the gate and pass A; the
    cheap pair facts ride along (D17: the batch table keeps them)."""
    src = EV.source_from_frames({"f1": _file([("C10H16O4", "[M-H]-")])},
                                run_inputs=SRC.RunInputs(summary=_summary(TOF)), mode="adapted")
    monkeypatch.setattr(SRC, "context_of", lambda s: pytest.fail("run context on a TOF source"))
    monkeypatch.setattr(CP, "q1_pass", lambda *a, **k: pytest.fail("pass A on a TOF source"))
    out = EV.level_source(src)
    assert {"iso_veto", "lowconf", "ion_only"} <= set(out.columns) and "n_competitors" not in out.columns
    r = out.iloc[0]
    assert r.evidence_level == "NA" and r.claim == "not assessed"
    assert r.evidence == "NA · not assessed on this instrument class (width model R(200) = 9 000 < 50 000)"
    assert r.would_lift == "" and r.tags == "" and r.competitors_left == ""


def test_na_without_a_width_model():
    src = EV.source_from_frames({"f1": _file([("C10H16O4", "[M-H]-")])},
                                run_inputs=SRC.RunInputs(summary=_summary(None)), mode="adapted")
    r = EV.level_source(src).iloc[0]
    assert r.evidence_level == "NA"
    assert r.evidence.endswith("(no width model: the instrument class is unknown)")


def test_na_mutant_an_orbitrap_class_source_is_levelled():
    led = _file([("C10H16O4", "[M-H]-"), ("C9H14O4", "[M-H]-")])
    src = EV.source_from_frames({"f1": led}, run_inputs=SRC.RunInputs(summary=_summary(ORBI)), mode="adapted")
    out = EV.level_source(src)
    assert len(out) == 2 and "NA" not in set(out["evidence_level"])
    assert set(out["evidence_level"]) <= set(SC.LEVELS) | {"reagent"}
    assert list(out.columns[:10]) == ["neutral_formula", "adduct", *SC.COLUMNS]
    assert (out["context_source"] != "").all()


# --------------------------------------------------------------------------- step rules the levels read
def test_lead_tag_prints_on_every_level_with_its_setter():
    r = _one(dict(ACID, lead_fact=True), lead_by={("C10H16O4", "[M-H]-"): "reflist_dim"})
    assert r.evidence_level == "4a"
    assert "lead: reflist_dim (a tentative lead: not answered by an own isotope line)" in r.tags.split(" | ")
    assert "lead (a tentative lead" in _one(dict(ACID, lead_fact=True)).tags


def test_side_channels_locked_and_the_switch(monkeypatch):
    acid = dict(neutral_formula="C10H16O4", adduct="[M+NO3]-")
    S = _prepared([acid], decomp=["[M+NO3]-"])
    r = S.P.iloc[0]
    s = SPL.split(S, r)
    assert "[M+CHO2]-" not in {d["adduct"] for d in s["plaus"]}
    monkeypatch.setattr(EV, "UNLOCKED", frozenset({"formate"}))
    s2 = SPL.split(S, r)
    assert "[M+CHO2]-" in {d["adduct"] for d in s2["plaus"]} or len(s2["plaus"]) >= len(s["plaus"])
    monkeypatch.setattr(EV, "SIDE_CHANNELS_LOCKED", False)
    assert SPL.locked_adducts() == set()


def test_series_exclusion_needs_an_anchored_series():
    D = pd.DataFrame(dict(neutral_formula=["C8H12O4", "C9H14O4", "C10H16O4"], adduct=["[M-H]-"] * 3))
    D["mz"] = [C.ion_mz(n, a) for n, a in zip(D["neutral_formula"], D["adduct"])]
    member = np.array([True, True, True])
    anchor = np.array([True, False, True])
    homo = RT.homologue_facts(D, member, anchor, tol_ppm=1.0, units=RT.LADDER_UNITS)
    assert homo[1]["anchored"] and homo[1]["unit"] == "CH2" and homo[1]["n_ok"] == 2
    ions = {RT.ion_key(n, a) for n, a in zip(D["neutral_formula"], D["adduct"])}
    left = {1: [dict(name="C7H18O5 [M-H]-", kind="mass", neutral="C7H18O5", adduct="[M-H]-")]}
    ex = RT.series_exclusions(D, left, homo, ions)
    assert list(ex[1]) == ["C7H18O5 [M-H]-"]
    homo0 = RT.homologue_facts(D, member, np.array([True, False, False]), tol_ppm=1.0, units=RT.LADDER_UNITS)
    assert not homo0[1]["anchored"] and RT.series_exclusions(D, left, homo0, ions) == {}


def test_partners_from_names_how_not_the_internal_level():
    lev = pd.DataFrame(dict(neutral_formula=["X1", "X2", "X3", "X4"], adduct=["[M+H]+", "[M+NH4]+", "[M-H]-", "[M-H]-"],
                            inpass_level=["3b", "3c", "3d", "4a"], inpass_why=["routes", "split pinned + listed",
                                                                                "ladder", "split pinned"],
                            ion_only_reading=[False, False, False, False]))
    p = EV.partners_from(lev, "src")
    assert dict(p["X1"]) == {"protonation": ["src X1 [M+H]+ two routes"]}
    assert "X2" not in p                                     # ammonium: a locked side-channel class
    assert dict(p["X3"]) == {"deprotonation": ["src X3 [M-H]- ladder"]}
    assert "X4" not in p
    assert dict(EV.partners_from(lev, "src", locked=False)["X2"]) == {"ammonium": ["src X2 [M+NH4]+ listed"]}


def test_context_source_names_the_keyword_and_the_field():
    act = LS.activation_record(batch="run 7", dataset="AP oxidation demo-set", label="NO3- CIMS")
    assert act["matched"]["ap_ox"] == [["dataset", "ap oxidation"]]
    lists = LS.ContextLists("negative", "ambient-air", [["monoterpene_hom_kang2024", "2024.2"],
                                                        ["contaminants_keller2008", "2008.2"]], activation=act)
    hits = lists.hits("C10H16O4")
    assert hits and not any(h["named"] for h in hits)
    assert lists.context(hits) == "reflist:monoterpene_hom_kang2024"
    assert lists.context_source(hits) == \
        "reflist:monoterpene_hom_kang2024: keyword 'ap oxidation' in the dataset name"
    assert lists.context_source([]).startswith("reflist:monoterpene_hom_kang2024: keyword 'ap oxidation' in the "
                                               "dataset name; reflist:contaminants_keller2008: always active")
    assert LS.ContextLists("negative", "ambient-air", [["monoterpene_hom_kang2024", "x"]]).context_source(hits) == \
        "reflist:monoterpene_hom_kang2024: activation not recorded"


def test_amine_gate_admissibility_rule():
    merged = pd.DataFrame(dict(neutral_formula=["C9H20N2O"], adduct=["[M+H]+"]))
    g = SPL.AmineGate(merged, None, set(), scan=(100.0, 400.0))
    # Y = C9H17NO: its [M+H]+ ion is NOT the merged C9H20N2O [M+H]+ ion -> absent in scan -> inadmissible
    a = g.nh4_admissibility("C9H17NO")
    assert not a["ok"] and a["why"].startswith("no uronium adduct ion of Y present (merged + per-file ledgers): "
                                               "[M+H]+ m/z 156.14 in scan, absent")
    # the urea cluster of C8H16 is the C9H20N2O [M+H]+ ion: present under another reading
    b = g.nh4_admissibility("C8H16")
    assert b["ok"] and b["present"] == ["[M+(CH4N2O)H]+"] and not b["own_name"]
    assert g("C8H16") ["how"] == SPL.NO_TS_WHY and not g("C8H16")["kept"]
    assert g("C2H8O2Si")["kept"]                             # Si: kept


def test_claims_and_summaries():
    assert [SC.claim_class(x) for x in ("3c", "4a", "4b", "5a", "5b", "reagent", "NA", None)] == \
        ["identified", "neutral", "ion", "tentative", "tentative", "reagent", "not assessed", "tentative"]
    assert SC.summarize(["4b", "NA", "reagent", "3c", None]) == {"3c": 1, "4b": 1, "reagent": 1, "NA": 1}
    assert list(SC.summarize_claims(["ion"])) == ["identified", "neutral", "ion", "tentative", "reagent",
                                                  "not assessed"]
    assert EV.SCALE_RELEASE == "0.10.0" and EV.SIDE_CHANNELS_LOCKED and EV.UNLOCKED == frozenset()
