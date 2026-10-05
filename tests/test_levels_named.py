"""The named cases of the evidence scale, each rebuilt from its DECISIVE facts.

Each case is a small synthetic source in the shape the decision reads after
pass B (levels.decide.Prepared): the pair's facts (ion established, its own
in-band isotope lines, rejections), the real enumeration space of the run's
reagent and context (levels.space.Space), the real context lists (the run's
reference lists and the pass-0 registry) and, on the uronium runs, the real
amine gate over a synthetic merged ledger / per-file ledger / batch time
series. Each test pins the level and the split / tag / flag text that decides
it, and its MUTANT is the minimal change of a decisive fact that must move the
level or the decisive text. The same pairs on the real batches sit in
tests/fixtures/levels/v1_* (tests/test_levels_fixtures.py).

Offline and synthetic; no run directory is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment.levels import competitors as CP
from peaky.assignment.levels import decide as DC
from peaky.assignment.levels import lists as LS
from peaky.assignment.levels import scale as SC
from peaky.assignment.levels import source as SRC
from peaky.assignment.levels import space as SP
from peaky.assignment.levels import split as SPL
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution

ACTIVE = [["contaminants_keller2008", "2008.2"], ["monoterpene_hom_kang2024", "2024.2"]]
#: how the two lists were activated on the batches: one always, one by a keyword in the dataset name
ACTIVATION = LS.activation_record(batch="", dataset="AP oxidation demo-set", label="")
RUNS = {
    # the labelled-nitrate batch: the run's opened families and decomposition grid
    "nitrate": dict(reagent="NO3+NO3_15N", context="ambient-air", pol="negative", labelled=True,
                    fams=("organosulfate", "nitrate", "siloxane", "amine", "fluorinated"),
                    decomp=["[M+NO3]-", "[M-H]-", "[M+^NO3]-", "[M+Br]-"],
                    engine=["[M+NO3]-", "[M-H]-", "[M+^NO3]-"], scan=(130.0, 707.0)),
    # the uronium batch: [M+NH4]+ is the reagent's own channel there (the amine gate decides it)
    "uronium": dict(reagent="Ur", context="uronium", pol="positive", labelled=False,
                    fams=("amine", "siloxane", "pdms", "glycol_peg", "phthalate"),
                    decomp=["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                    engine=["[M+H]+", "[M+(CH4N2O)H]+"], scan=(125.0, 741.0)),
}


class _Ctx:
    """The slice of a RunContext the decision reads, on the run's real enumeration space."""

    def __init__(self, run: dict, *, context=None, labelled=None):
        self.space = SP.Space(run["reagent"], context or run["context"], ACTIVE, run["fams"])
        self.decomp_adducts = list(run["decomp"])
        self.engine_channels = list(run["engine"])
        self.labelled = run["labelled"] if labelled is None else labelled
        self.alien = frozenset()

    def decompositions(self, counts, n, a, adducts=None):
        return SP.decompositions(self.space, counts, n, a, self.decomp_adducts if adducts is None else adducts)


#: a pair whose ion is established (unique in the calibrated window) and that nothing rejects
ROW = dict(ion="", height=1e5, n_files_obs=10, iso_veto=False, label_veto=False, lowconf=False, below=False,
           ion_only=False, tied=False, committed_contradicted=False, committed_reasons="", no_comp_info=False,
           ion_only_reading=False, lead_fact=False, committed_matched="", committed_matched_lines="",
           committed_plausible=True, n_excl_iso=0, n_competitors=0)
C13 = dict(committed_matched="C", committed_matched_lines="13C")      # an own in-band 13C line


def _m0(n, a):
    return dict(role="M0", neutral_formula=n, adduct=a, method="grid", height=1e5, mz=C.ion_mz(n, a))


def _ts(traces: dict, *, n=40, extra=()):
    """A batch time series: one spectrum every 2 h; ``traces`` = {(neutral, adduct): heights}."""
    t0 = pd.Timestamp("2026-01-05 00:00:00")
    rows = []
    for k in range(n):
        for (nn, aa), h in traces.items():
            rows.append(dict(sample_item_id=f"x{k:03d}", datetime_utc=t0 + pd.Timedelta(hours=2 * k),
                             mz=C.ion_mz(nn, aa), height=float(h[k]), role="M0", neutral_formula=nn, adduct=aa))
    for mz in extra:            # unrelated peaks: a series that is there, whatever it holds
        rows += [dict(sample_item_id=f"x{k:03d}", datetime_utc=t0 + pd.Timedelta(hours=2 * k), mz=mz,
                      height=1e4, role="M0", neutral_formula="", adduct="") for k in range(n)]
    return pd.DataFrame(rows)


SHAPE = 1e4 * (1.5 + np.sin(np.arange(40) / 3.0))                      # a shaped trace (cv >> FLAT_CV)
NOISE = 1e4 * (1.5 + np.cos(np.arange(40) * 2.3 + 0.7) * np.sin(np.arange(40) * 0.9))


def _case(rows, run="nitrate", *, merged=(), per_file=(), ts=None, protected=(), context=None, labelled=None,
          alien=(), comps=(), with_gate=None, partners=None, n_files=1):
    """Level ``rows`` (pair facts over ROW) on a synthetic source of ``run`` (``n_files`` copies of the per-file
    ledger; ``partners`` = other-source partners as ``evidence.partners_from`` gives them)."""
    R = RUNS[run]
    ctx = _Ctx(R, context=context, labelled=labelled)
    P = pd.DataFrame([{**ROW, "mz": C.ion_mz(r["neutral_formula"], r["adduct"]), **r} for r in rows])
    P["n_competitors"] = [sum(1 for c in comps if (c[0], c[1]) == (n, a)) or int(r.n_competitors)
                          for n, a, r in zip(P["neutral_formula"], P["adduct"], P.itertuples())]
    cm = pd.DataFrame([dict(neutral_formula=c[0], adduct=c[1], competitor=f"{c[2]} {c[3]}", kind="mass", ppm=0.1,
                            status=c[4], how="isotopes" if c[4] == "excluded" else "", why="", window="run",
                            comp_neutral=c[2], comp_adduct=c[3]) for c in comps], columns=list(CP.COMP_COLUMNS))
    led = pd.DataFrame([_m0(n, a) for n, a in zip(P["neutral_formula"], P["adduct"])]
                       + [_m0(n, a) for n, a in per_file])
    pf = {f"f{k + 1}": led.copy() for k in range(n_files)}
    gate = None
    if with_gate if with_gate is not None else R["pol"] == "positive":
        mf = pd.DataFrame([dict(neutral_formula=n, adduct=a) for n, a in merged], columns=["neutral_formula", "adduct"])
        gate = SPL.AmineGate(mf, ts, set(protected), R["scan"], per_file=pf)
    lists = LS.ContextLists(R["pol"], context or R["context"], ACTIVE, activation=ACTIVATION)
    S = DC.Prepared(name="t", P=P, comps=cm, pf=pf, ctx=ctx, lists=lists, ts=None, tol_ppm=1.0, pol=R["pol"],
                    run_classes=DC.run_classes_of(R["engine"]), below={}, alien=frozenset(alien), arm=False,
                    skip_m0=frozenset(), gate=gate, lead_by={})
    res = DC.inpass(S, partners)
    texts = DC.inpass_texts(S, res)
    lv = DC.relevel(S, res)
    rec = DC.records(S, res, texts, lv)
    rec["claim"] = rec["evidence_level"].map(SC.claim_class)
    return rec


def _one(row, run="nitrate", **kw):
    rec = _case([row], run, **kw)
    return rec.iloc[0]


def _kinds(r) -> set:
    return set(str(r.tag_kinds).split("|")) - {""}


# =========================================================================== uronium: the NH4 admissibility rule
DIPEA = dict(neutral_formula="C8H19N", adduct="[M+H]+")
#: Y = C8H16 (DIPEA - NH3): its urea-cluster ion C9H21N2O+ is in the run, read as C9H20N2O [M+H]+ (merged)
#: and as C8H16 [M+(CH4N2O)H]+ in one file
DIPEA_Y_ION = dict(merged=[("C9H20N2O", "[M+H]+")], per_file=[("C8H16", "[M+(CH4N2O)H]+")])
#: a time series that holds neither Y's parents nor its NH4 ion: the gate reads 'parent channels absent'
NO_PARENT_TS = _ts({}, extra=(300.0,))


def test_dipea_nh4_admissible_through_ys_urea_ion_under_another_reading_is_the_open_amine_default():
    r = _one(DIPEA, "uronium", ts=NO_PARENT_TS, **DIPEA_Y_ION)
    assert r.evidence_level == "4b" and r.claim == "ion" and not r.split_pinned
    assert r.split_how == (
        "amine default (NH4 reading unconfirmed: C8H16 [M+NH4]+ -- parent channels absent) [NH4 admissibility rule: "
        "Y's uronium adduct ion is present under another reading: C8H16 -- urea cluster m/z 173.1648 as C8H16 "
        "[M+(CH4N2O)H]+ (1 per-file) / C9H20N2O [M+H]+ (merged)]")
    assert r.nh4_gate == "amine default"
    assert {"NH4 reading admissible via Y's ion under another reading", "amine default",
            "named list, not 3c"} <= _kinds(r)
    assert r.named_list == "reflist:contaminants_keller2008 = DIPEA"
    assert "named list (not used: split open): reflist:contaminants_keller2008 = DIPEA" in r.tags.split(" | ")
    assert r.would_lift.startswith("split not pinned: amine default (NH4 reading unconfirmed")


def test_dipea_mutant_without_ys_ion_the_nh4_reading_is_inadmissible_and_the_name_lifts_it_to_3c():
    r = _one(DIPEA, "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "3c" and r.split_pinned
    # Y's [M+H]+ lies below the scan: only the urea cluster is testable, and it is absent
    assert r.split_how == ("NH4 reading inadmissible: no uronium adduct of Y (C8H16 [M+NH4]+ -- no uronium adduct ion "
                           "of Y present (merged + per-file ledgers): urea cluster m/z 173.16 in scan, absent, [M+H]+ "
                           "m/z 113.13 outside the scan (not counted)); only decomposition")


TBA = dict(neutral_formula="C12H27N", adduct="[M+H]+")


def test_tba_nh4_inadmissible_pins_the_split_and_a_named_entry_makes_it_3c():
    r = _one(TBA, "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "3c" and r.claim == "identified"
    assert r.split_how == (
        "NH4 reading inadmissible: no uronium adduct of Y (C12H24 [M+NH4]+ -- no uronium adduct ion of Y present "
        "(merged + per-file ledgers): [M+H]+ m/z 169.20 in scan, absent, urea cluster m/z 229.23 in scan, absent); "
        "only decomposition")
    assert r.named_list == "reflist:contaminants_keller2008 = TBA" and r.nh4_gate == "NH4 inadmissible"
    assert r.nh4_inadmissible.endswith("(gate: presence-reread)")
    assert "NH4 reading inadmissible" in _kinds(r)
    assert r.would_lift == "level 2 (MS2 / standards) is not automatic"
    assert r.evidence.startswith("3c (split pinned + named context-list entry) · ion: unique in the calibrated window"
                                 " · split: PINNED -- NH4 reading inadmissible")


def test_tba_mutant_ys_protonated_ion_in_the_run_opens_the_split():
    """Y = C12H24 committed as [M+H]+ under its own name: the NH4 reading is
    admissible (and, Y's ion being its own name, no 'another reading' note);
    with no trace to confirm it the gate does not keep it, and an [M+H]+ with an
    admissible NH4 reading is open -- 4b, the name not used."""
    r = _one(TBA, "uronium", ts=NO_PARENT_TS, merged=[("C12H24", "[M+H]+")])
    assert r.evidence_level == "4b" and not r.split_pinned
    assert r.nh4_gate == "amine default"
    assert r.split_how == "amine default (NH4 reading unconfirmed: C12H24 [M+NH4]+ -- parent flat, adduct unconfirmable)"
    assert r.nh4_admissible == "C12H24: uronium adduct ion present: [M+H]+ m/z 169.1951 as C12H24 [M+H]+ (merged)"
    assert "named list, not 3c" in _kinds(r)
    assert "NH4 admissibility rule: Y's uronium adduct ion is present under another reading" not in r.split_how


C12H23N = dict(neutral_formula="C12H23N", adduct="[M+H]+", **C13)


def test_c12h23n_nh4_inadmissible_with_an_own_13c_line_is_4a():
    r = _one(C12H23N, "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "4a" and r.claim == "neutral" and r.split_pinned
    assert r.split_how.startswith("NH4 reading inadmissible: no uronium adduct of Y (C12H20 [M+NH4]+ -- no uronium "
                                  "adduct ion of Y present (merged + per-file ledgers): [M+H]+ m/z 165.16 in scan, "
                                  "absent, urea cluster m/z 225.20 in scan, absent); only decomposition")
    assert r.positive_fact == "own in-band isotope line(s) 13C (C of the neutral)"
    assert r.would_lift == "3c needs a NAMED context-list entry naming the neutral"


def test_c12h23n_mutant_without_the_13c_line_is_4b():
    r = _one(dict(C12H23N, committed_matched="", committed_matched_lines=""), "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "4b" and r.split_pinned and r.positive_fact == ""
    assert r.would_lift.startswith("4a needs a positive fact")


# --------------------------------------------------------------------------- the amine gate KEEPs the NH4 reading
def _tracking_ts(Y, *, track=True):
    """Y's NH4 ion (the committed [M+H]+ ion) and Y's urea-cluster parent over 40 2-h bins."""
    nh4 = SHAPE * 0.3 if track else NOISE
    return _ts({(Y, "[M+NH4]+"): nh4, (Y, "[M+(CH4N2O)H]+"): SHAPE})


@pytest.mark.parametrize("x,y", [("C8H15N", "C8H12"), ("C9H13N", "C9H10")])
def test_gate_keep_leaves_the_amine_reading_open_and_contested(x, y):
    r = _one(dict(neutral_formula=x, adduct="[M+H]+"), "uronium", ts=_tracking_ts(y),
             per_file=[(y, "[M+(CH4N2O)H]+")])
    assert r.evidence_level == "4b" and not r.split_pinned
    assert r.nh4_gate == "NH4 kept"
    assert r.split_how.startswith(f"NH4 reading kept by the amine gate: {y} [M+NH4]+ (tracks its own [M+H]+/urea "
                                  f"parent, r 1.00 over 40 2-h bins) -- the committed [M+H]+ (amine) reading is "
                                  f"contested")
    assert {"NH4 reading kept by the gate"} <= _kinds(r)
    assert r.nh4_gate_detail.startswith(f"{y}: keep (tracks its own [M+H]+/urea parent")


@pytest.mark.parametrize("x,y", [("C8H15N", "C8H12"), ("C9H13N", "C9H10")])
def test_gate_keep_mutants(x, y):
    """An NH4 trace that does not track its parent: the gate re-reads it as the
    amine (amine default, still open); Y's ion absent from the run: the NH4
    reading is inadmissible and the split pins."""
    r = _one(dict(neutral_formula=x, adduct="[M+H]+"), "uronium", ts=_tracking_ts(y, track=False),
             per_file=[(y, "[M+(CH4N2O)H]+")])
    assert r.nh4_gate == "amine default" and not r.split_pinned and r.evidence_level == "4b"
    r = _one(dict(neutral_formula=x, adduct="[M+H]+"), "uronium", ts=_tracking_ts(y))
    assert r.nh4_gate == "NH4 inadmissible" and r.split_pinned


C8H17N = dict(neutral_formula="C8H17N", adduct="[M+H]+", **C13)


def test_c8h17n_amine_default_is_open_even_with_an_own_13c_line():
    r = _one(C8H17N, "uronium", ts=NO_PARENT_TS, merged=[("C9H18N2O", "[M+H]+")],
             per_file=[("C8H14", "[M+(CH4N2O)H]+")])
    assert r.evidence_level == "4b" and r.nh4_gate == "amine default"
    assert r.split_how.startswith("amine default (NH4 reading unconfirmed: C8H14 [M+NH4]+ -- parent channels absent) "
                                  "[NH4 admissibility rule: Y's uronium adduct ion is present under another reading: "
                                  "C8H14 -- urea cluster m/z 171.1492 as ")


def test_c8h17n_mutant_inadmissible_pins_and_the_line_makes_it_4a():
    r = _one(C8H17N, "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "4a" and r.nh4_gate == "NH4 inadmissible"


# --------------------------------------------------------------------------- siloxanes
#: an amine reading of a siloxane's ion: Y = C16H48O8Si8, the cyclic siloxane D8
SILOX = dict(neutral_formula="C16H51NO8Si8", adduct="[M+H]+")
SILOX_Y_ION = [("C16H48O8Si8", "[M+(CH4N2O)H]+")]


def test_siloxane_nh4_reading_is_kept_by_the_si_exception_when_admissible():
    r = _one(SILOX, "uronium", ts=NO_PARENT_TS, per_file=SILOX_Y_ION)
    assert r.nh4_gate == "NH4 kept" and not r.split_pinned and r.evidence_level == "4b"
    assert "(Si (siloxane NH4 adducts are real))" in r.split_how


def test_siloxane_the_admissibility_rule_overrides_the_si_exception():
    r = _one(SILOX, "uronium", ts=NO_PARENT_TS)
    assert r.nh4_gate == "NH4 inadmissible" and r.split_pinned
    assert r.nh4_inadmissible.endswith("(gate: Si)")
    assert r.evidence_level == "4b"
    assert _one(dict(SILOX, **C13), "uronium", ts=NO_PARENT_TS).evidence_level == "4a"


def test_siloxane_rejected_is_5b_whatever_the_gate():
    r = _one(dict(SILOX, lowconf=True), "uronium", ts=NO_PARENT_TS, per_file=SILOX_Y_ION)
    assert r.evidence_level == "5b" and r.would_lift == "refuted: lowconf"
    assert r.evidence.startswith("5b · rejected: lowconf · ion: unique in the calibrated window · split: ")


# =========================================================================== named entries and their flags
@pytest.mark.parametrize("neutral,name", [("C16H32O2", "Palmitic acid"), ("C18H36O2", "Stearic acid"),
                                          ("C18H34O2", "Oleic acid")])
def test_fatty_acids_protonated_are_3c_with_the_mode_flag(neutral, name):
    r = _one(dict(neutral_formula=neutral, adduct="[M+H]+"), "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "3c" and r.split_how == "only decomposition"
    assert r.evidence.startswith("3c (split pinned + named context-list entry) [FLAG: every named entry's source "
                                 "mode contradicts this run (MODE CONTRADICTS); flag only, not a block] · ")
    assert name.lower() in r.named_list.lower()
    assert "MODE CONTRADICTS: entry recorded in ESI-, this run is ESI+" in r.named_mode_flag
    assert "3c name: mode contradicts" in _kinds(r)
    assert any(t.startswith("FLAG, named entry's source mode contradicts this run: ") and
               t.endswith(" -- flag only, does not block 3c") for t in r.tags.split(" | "))


def test_fatty_acid_mutant_on_its_own_polarity_carries_no_flag():
    r = _one(dict(neutral_formula="C16H32O2", adduct="[M-H]-"), "nitrate")
    assert r.evidence_level == "3c" and r.named_mode_flag == "" and "[FLAG" not in r.evidence
    assert "3c name: mode contradicts" not in _kinds(r)


@pytest.mark.parametrize("neutral,name", [("C10H10O4", "Dimethyl phthalate"), ("C11H16O2", "BHA")])
def test_esi_plus_entries_deprotonated_are_3c_with_the_mode_flag(neutral, name):
    r = _one(dict(neutral_formula=neutral, adduct="[M-H]-"), "nitrate")
    assert r.evidence_level == "3c"
    assert f"reflist:contaminants_keller2008 = {name} [MODE CONTRADICTS: entry recorded in ESI+, this run is " \
           f"ESI-]" in r.named_list
    assert "[FLAG: every named entry's source mode contradicts this run" in r.evidence


@pytest.mark.parametrize("neutral", ["C10H10O4", "C11H16O2"])
def test_esi_plus_entry_mutant_protonated_on_a_positive_run_is_unflagged(neutral):
    r = _one(dict(neutral_formula=neutral, adduct="[M+H]+"), "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "3c" and r.named_mode_flag == ""


TFA_NO3 = dict(neutral_formula="C2HF3O2", adduct="[M+NO3]-")


def test_tfa_nitrate_cluster_is_3c_with_the_ion_form_flag():
    r = _one(TFA_NO3, "nitrate")
    assert r.evidence_level == "3c" and r.split_how == "only decomposition"
    assert r.named_list == ("reflist:contaminants_keller2008 = Trifluoroacetic acid, TFA (anion) [ion form differs: "
                            "entry = the anion [M-H]-, this pair = [M+NO3]-]")
    assert "[FLAG" not in r.evidence                       # an ion-form note is no mode contradiction
    assert "class list" in _kinds(r) and r.context.startswith("reflist:contaminants_keller2008 = Trifluoroacetic")


def test_tfa_mutant_the_anion_itself_carries_no_ion_form_flag():
    r = _one(dict(TFA_NO3, adduct="[M-H]-"), "nitrate")
    assert r.evidence_level == "3c" and r.named_mode_flag == ""


# =========================================================================== the context window
@pytest.mark.parametrize("neutral,name,other", [("C6H5NO3", "nitrophenol", "C6H4"),
                                                ("C6H5NO4", "nitrocatechol", "C6H4O")])
def test_nitroaromatics_are_3c_pinned_only_by_the_context_window(neutral, name, other):
    r = _one(dict(neutral_formula=neutral, adduct="[M-H]-"), "nitrate")
    assert r.evidence_level == "3c" and r.named_list == f"registry:nitroaromatic = {name}"
    assert r.split_how.startswith(f"only decomposition [pinned only by the context window: {other} [M+NO3]- ((H+X)/C=")
    assert r.window_only.startswith(f"{other} [M+NO3]- ((H+X)/C=")
    assert "pinned only by the context window" in _kinds(r)
    assert any(t.endswith("-- a window lifted would open the split") for t in r.tags.split(" | "))


@pytest.mark.parametrize("neutral", ["C6H5NO3", "C6H5NO4"])
def test_nitroaromatic_mutant_a_wider_window_opens_the_split(neutral):
    r = _one(dict(neutral_formula=neutral, adduct="[M-H]-"), "nitrate", context="combustion")
    assert not r.split_pinned and r.evidence_level == "4b"
    assert "named list (not used: split open): registry:nitroaromatic" in r.tags


DMF = dict(neutral_formula="C3H7NO", adduct="[M+(CH4N2O)H]+")


def test_urea_cluster_and_the_protonated_x_plus_urea_isobar_admitted_over_the_window():
    r = _one(DMF, "uronium", ts=NO_PARENT_TS)
    assert r.evidence_level == "4b" and not r.split_pinned
    assert r.window_isobar == "C4H11N3O2 [M+H]+ (X+urea; window (H+X)/C=2.75 out of (0.4, 2.6))"
    assert "[X+reagent isobar admitted over the context window: C4H11N3O2 [M+H]+ (X+urea; window" in r.split_how
    assert "X+reagent isobar admitted over the window" in _kinds(r)
    assert ("-- the context window does not exclude the X+reagent isobar") in r.tags
    assert "decision" not in r.evidence                    # no internal vocabulary reaches the record


def test_urea_cluster_mutant_without_the_isobar_rule_the_window_pins_it(monkeypatch):
    monkeypatch.setattr(SPL, "REAGENT_MOLS", {"positive": {}, "negative": {}})
    r = _one(DMF, "uronium", ts=NO_PARENT_TS)
    assert r.split_pinned and r.window_isobar == ""
    assert "pinned only by the context window: C4H11N3O2 [M+H]+" in r.split_how


# =========================================================================== locked side channels
C11 = dict(neutral_formula="C11H16O5", adduct="[M-H]-", **C13)


def test_c11_deprotonated_pinned_only_by_the_side_channel_lock_is_4a():
    r = _one(C11, "nitrate")
    assert r.evidence_level == "4a" and r.split_pinned
    assert r.split_how == ("only decomposition; side channels locked (acetate, formate would open it: formate: "
                           "C10H14O3 [M+CHO2]-; acetate: C9H12O3 [M+C2H3O2]-)")
    assert "side channels locked (split)" in _kinds(r) and "side channels locked" in r.tags.split(" | ")


def test_c11_mutant_formate_unlocked_opens_the_split(monkeypatch):
    monkeypatch.setattr(EV, "UNLOCKED", frozenset({"formate"}))
    r = _one(C11, "nitrate")
    assert not r.split_pinned and r.evidence_level == "4b"
    assert "C10H14O3 [M+CHO2]-" in r.split_how


CHLORO = dict(neutral_formula="C10H17ClO4", adduct="[M-H]-", committed_matched="Cl", committed_matched_lines="37Cl")


def test_a_chlorinated_acid_names_the_locked_chloride_channel_that_would_open_it():
    r = _one(CHLORO, "nitrate")
    assert r.evidence_level == "4a" and r.chloride_open == "C10H16O4 [M+Cl]-"
    assert r.split_how == ("only decomposition; side channels locked (acetate, chloride, formate would open it: "
                           "formate: C9H15ClO2 [M+CHO2]-; acetate: C8H13ClO2 [M+C2H3O2]-; chloride: C10H16O4 [M+Cl]-)")
    assert "chloride would open the split (locked)" in _kinds(r)
    assert r.positive_fact == "own in-band isotope line(s) 37Cl (Cl of the neutral)"


def test_chloride_mutant_unlocked_opens_the_split(monkeypatch):
    monkeypatch.setattr(EV, "UNLOCKED", frozenset({"chloride"}))
    r = _one(CHLORO, "nitrate")
    assert not r.split_pinned and r.evidence_level == "4b" and r.chloride_open == ""
    assert "C10H16O4 [M+Cl]-" in r.split_how


# =========================================================================== the 15N label
LABEL = dict(neutral_formula="C10H16O4", adduct="[M+^NO3]-")


def test_the_15n_label_pins_the_split_and_is_the_positive_fact():
    r = _one(LABEL, "nitrate")
    assert r.evidence_level == "4a" and r.split_pinned
    assert r.split_how == "label (the ion carries the reagent's ^N)"
    assert r.positive_fact == "15N label"


def test_label_mutants():
    """The 14N cluster of the same neutral carries no label; a label twin that
    vetoes the reading rejects it; an alien pair (rule K) loses the label pin."""
    r = _one(dict(LABEL, adduct="[M+NO3]-"), "nitrate")
    assert r.positive_fact == "" and r.evidence_level == "4b"
    r = _one(dict(LABEL, label_veto=True), "nitrate")
    assert r.evidence_level == "5b" and r.would_lift == "refuted: label_veto"
    r = _one(LABEL, "nitrate", alien=[("C10H16O4", "[M+^NO3]-")])
    assert "label" not in r.split_how and r.positive_fact == ""


# =========================================================================== the reagent bucket
def test_hbr_on_the_labelled_cluster_is_the_reagent_bucket():
    r = _one(dict(neutral_formula="HBr", adduct="[M+^NO3]-", lowconf=True), "nitrate")
    assert r.evidence_level == "reagent" and r.claim == "reagent"
    assert r.evidence.startswith("reagent bucket · reagent identity (HBr) · ion: unique in the calibrated window · "
                                 "split: label (the ion carries the reagent's ^N) · context source: ")
    assert r.would_lift == "reagent ion / reagent cluster (not levelled)"


def test_reagent_mutant_a_non_reagent_combination_is_levelled():
    r = _one(dict(neutral_formula="HBrO", adduct="[M+^NO3]-", lowconf=True), "nitrate")
    assert r.evidence_level == "5b" and r.would_lift == "refuted: lowconf"


# =========================================================================== NA: a TOF-class source
TOF = Resolution(coef=200 / 9000 / 200 ** 1.0, exponent=1.0, offset=0.0).as_dict()      # R(200) = 9 000
ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()          # R(200) = 100 000


def _file(rows):
    led = L.new_ledger(pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(rows))],
                                     "mz": [C.ion_mz(n, a) for n, a in rows], "height": [1e5] * len(rows)}))
    for i, (n, a) in enumerate(rows):
        L.commit_assignment(led, f"p{i}", neutral_formula=n, adduct=a, ion_formula=f"{n}{a[-1]}", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="x")
    return led


def test_a_tof_class_file_reads_na_in_the_per_file_stage():
    led = _file([("C2HF3O2", "[M+NO3]-"), ("C6H5NO3", "[M-H]-")])
    ri = EV.file_run_inputs(sample_id="f1", reagent="NO3+NO3_15N", context="ambient-air", resolution=TOF,
                            reflists_active=ACTIVE, height_gate_cps=1e3, degeneracy_cal=(0.0, 0.3))
    out = EV.apply_levels(led, run_inputs=ri)
    m0 = led[led["role"] == "M0"]
    assert set(m0["evidence_level"]) == {"NA"} and set(m0["claim"]) == {"not assessed"}
    assert set(m0["evidence"]) == {"NA · not assessed on this instrument class (width model R(200) = 9 000 < 50 000)"}
    assert out["instrument"] == {"class": "tof", "r200": 9000.0} and out["claims"]["not assessed"] == 2


def test_na_mutant_the_same_file_on_an_orbitrap_class_width_model_is_levelled():
    led = _file([("C2HF3O2", "[M+NO3]-"), ("C6H5NO3", "[M-H]-")])
    ri = EV.file_run_inputs(sample_id="f1", reagent="NO3+NO3_15N", context="ambient-air", resolution=ORBI,
                            reflists_active=ACTIVE, height_gate_cps=1e3, degeneracy_cal=(0.0, 0.3))
    EV.apply_levels(led, run_inputs=ri)
    lv = set(led.loc[led["role"] == "M0", "evidence_level"])
    assert "NA" not in lv and lv <= set(SC.LEVELS) | set(SC.BUCKETS)


def test_the_na_short_circuit_runs_before_any_scale_work(monkeypatch):
    """NA before the run context and pass A; the pooled pair facts ride along (D17)."""
    monkeypatch.setattr(SRC, "context_of", lambda s: pytest.fail("run context on a TOF-class source"))
    monkeypatch.setattr(CP, "q1_pass", lambda *a, **k: pytest.fail("pass A on a TOF-class source"))
    src = EV.source_from_frames({"f1": _file([("C10H16O4", "[M-H]-")])},
                                run_inputs=SRC.RunInputs(summary=dict(reagent="NO3", context="ambient-air",
                                                                      resolution=TOF, per_file=[])), mode="run")
    out = EV.level_source(src)
    assert set(out["evidence_level"]) == {"NA"} and {"iso_veto", "lowconf"} <= set(out.columns)


# =========================================================================== other-source partners: pinned anchors only
#: a CH2 chain of [M-H]- ions that are also X.NO3- of C8H12O4 ... (two decompositions: the split stays open)
OPEN_CHAIN = ("C8H13NO7", "C9H15NO7", "C10H17NO7")
PINNED_CHAIN = ("C8H12O4", "C9H14O4", "C10H16O4")


def _chain_case(chain, *, partners):
    rows = [dict(neutral_formula=n, adduct="[M-H]-", committed_matched="C") for n in chain]
    comp = [(chain[1], "[M-H]-", "C7H18O5", "[M-H]-", "left")]
    part = {n: {"protonation": [f"src {n} [M+H]+ two routes"]} for n in (chain[0], chain[2])} if partners else None
    return _case(rows, "nitrate", comps=comp, partners=part, n_files=2).set_index("neutral_formula")


def test_an_other_source_partner_on_an_open_split_anchor_is_not_counted():
    """The in-pass route of an other-source partner counts only on a PINNED split: here the anchors' own splits are
    open, so they anchor nothing, the middle member keeps its competitor (5a) and the anchors say why."""
    out = _chain_case(OPEN_CHAIN, partners=True)
    mid = out.loc["C9H15NO7"]
    assert mid.evidence_level == "5a" and int(mid.n_series_excl) == 0 and mid.competitors_left == "C7H18O5 [M-H]-"
    for anchor in (OPEN_CHAIN[0], OPEN_CHAIN[2]):
        r = out.loc[anchor]
        assert not r.split_pinned and r.evidence_level == "4b"
        assert r.split_how.startswith("2 decompositions: ")
        assert "other-source route (protonation) not counted: this ion's own split is open" in r.tags.split(" | ")
        assert r.anchor_kind == "none" and r.inpass_why != "routes"
    assert _chain_case(OPEN_CHAIN, partners=False).loc["C9H15NO7", "evidence_level"] == "5a"


def test_partner_mutant_the_same_chain_with_pinned_anchors_is_anchored_and_excludes_the_competitor():
    out = _chain_case(PINNED_CHAIN, partners=True)
    assert out.loc["C8H12O4", "split_pinned"] and out.loc["C8H12O4", "inpass_why"] == "routes"
    mid = out.loc["C9H14O4"]
    assert mid.evidence_level == "4a" and int(mid.n_series_excl) == 1 and mid.competitors_left == ""
    assert "not counted" not in out.loc["C8H12O4", "tags"]
    assert _chain_case(PINNED_CHAIN, partners=False).loc["C9H14O4", "evidence_level"] == "5a"
