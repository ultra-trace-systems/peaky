"""scripts/level_ledger.py -- the executable reference of the evidence scale.

The script shares the engine's fact layer and states the decision again on its
own (step 0, the step-1 outcome, the split's outcome per rule, the level, what
would lift it). Here its decision is held against the engine's on small
synthetic sources: one passing case per level and per split rule, and for each
the mutant -- the smallest change that must move the outcome. Script and engine
must agree on the level, the would-lift text, the competitors left and the
split on every pair of every case. The real runs are compared outside the repo
(they hold the data).

Run: pytest tests/test_level_ledger.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import level_ledger as LL  # noqa: E402
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment.levels import competitors as CP  # noqa: E402
from peaky.assignment.levels import decide as DC  # noqa: E402
from peaky.assignment.levels import lists as LS  # noqa: E402
from peaky.assignment.levels import scale as SC  # noqa: E402
from peaky.assignment.levels import source as SRC  # noqa: E402
from peaky.assignment.levels import space as SP  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.chem.resolution import Resolution  # noqa: E402

ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()     # R(200) = 100 000
TOF = Resolution(coef=200 / 9000 / 200 ** 1.0, exponent=1.0, offset=0.0).as_dict()  # R(200) = 9 000
NO3 = ["[M+NO3]-", "[M-H]-"]
UR = ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"]
COMPARED = ("evidence_level", "claim", "would_lift", "competitors_left", "split_pinned")


# --------------------------------------------------------------------------- a hand-built source
class _Ctx:
    """The slice of a RunContext the decision reads."""

    def __init__(self, reagent="NO3", context="ambient-air", decomp=NO3, labelled=False):
        self.space = SP.Space(reagent, context, [], ())
        self.decomp_adducts = list(decomp)
        self.engine_channels = list(self.space.adducts)
        self.labelled = labelled
        self.alien = frozenset()

    def decompositions(self, counts, n, a, adducts=None):
        return SP.decompositions(self.space, counts, n, a, self.decomp_adducts if adducts is None else adducts)


class _NoLists:
    def hits(self, neutral):
        return []

    def mode_flag(self, *a):
        return ""

    def context(self, hits):
        return ""

    def context_source(self, hits):
        return "registry: pass-0 registry"


class _Gate:
    """The amine gate's verdicts, by hand: ``keep`` = {Y: (kept, track, how)}, ``absent`` = the Y whose uronium
    adduct ion the run lacks, ``other`` = the Y present only under another reading."""

    def __init__(self, keep=None, absent=(), other=()):
        self.keep, self.absent, self.other = dict(keep or {}), set(absent), set(other)

    def __call__(self, X):
        kept, track, how = self.keep.get(X, (False, False, "parent channels absent"))
        return dict(kept=kept, track=track, verdict="keep" if track else "x", r=np.nan, ov=0, how=how,
                    amine=C.format_formula({**C.parse_formula(X), "N": C.parse_formula(X).get("N", 0) + 1,
                                            "H": C.parse_formula(X).get("H", 0) + 3}))

    def nh4_admissibility(self, Y):
        if Y in self.absent:
            return dict(ok=False, why="no uronium adduct ion of Y present (merged + per-file ledgers): [M+H]+ m/z "
                        "100.00 in scan, absent", present=[], testable=["[M+H]+"], own_name=False)
        return dict(ok=True, why="uronium adduct ion present: [M+H]+ m/z 100.0000 as X [M+H]+ (merged)",
                    present=["[M+H]+"], testable=["[M+H]+"], own_name=Y not in self.other)


ROW = dict(ion="", height=1e5, n_files_obs=3, iso_veto=False, label_veto=False, lowconf=False, below=False,
           ion_only=False, tied=False, committed_contradicted=False, committed_reasons="", no_comp_info=False,
           ion_only_reading=False, lead_fact=False, committed_matched="", committed_matched_lines="",
           committed_plausible=True, n_excl_iso=0, n_competitors=0)


def _prepared(rows, *, comps=(), lists=None, below=None, ctx=None, pol="negative", gate=None, alien=(),
              arm=False, pf=None):
    P = pd.DataFrame([{**ROW, "mz": C.ion_mz(r["neutral_formula"], r["adduct"])
                       if r["adduct"] in C.ADDUCT_SHIFTS else 200.0, **r} for r in rows])
    cm = pd.DataFrame([dict(neutral_formula=c[0], adduct=c[1], competitor=f"{c[2]} {c[3]}", kind=c[5] if len(c) > 5
                            else "mass", ppm=0.1, status=c[4], how="isotopes" if c[4] == "excluded" else "", why="",
                            window="run", comp_neutral=c[2], comp_adduct=c[3]) for c in comps],
                      columns=list(CP.COMP_COLUMNS))
    P["n_competitors"] = [int(((cm["neutral_formula"] == n) & (cm["adduct"] == a)).sum())
                          for n, a in zip(P["neutral_formula"], P["adduct"])]
    P["n_excl_iso"] = [int(((cm["neutral_formula"] == n) & (cm["adduct"] == a) & (cm["status"] == "excluded")).sum())
                       for n, a in zip(P["neutral_formula"], P["adduct"])]
    if pf is None:
        pf = {"f1": pd.DataFrame(dict(role="M0", neutral_formula=P["neutral_formula"], adduct=P["adduct"],
                                      method="grid", height=1e5, mz=P["mz"]))}
    ctx = ctx or _Ctx()
    return DC.Prepared(name="t", P=P, comps=cm, pf=pf, ctx=ctx, lists=lists or _NoLists(), ts=None, tol_ppm=1.0,
                       pol=pol, run_classes=DC.run_classes_of(ctx.engine_channels), below=below or {},
                       alien=frozenset(alien), arm=arm, skip_m0=frozenset(), gate=gate, lead_by={})


def _engine(S, partners=None):
    res = DC.inpass(S, partners if not S.arm else None)
    rec = DC.records(S, res, DC.inpass_texts(S, res), DC.relevel(S, res))
    rec["claim"] = rec["evidence_level"].map(SC.claim_class)
    return rec


def both(S, partners=None) -> pd.DataFrame:
    """The script's decision, after asserting it equals the engine's on every pair."""
    core = _engine(S, partners).set_index(["neutral_formula", "adduct"])
    mine = LL.decide(S, partners).set_index(["neutral_formula", "adduct"])
    assert list(core.index) == list(mine.index)
    for col in COMPARED:
        a, b = core[col].tolist(), mine[col].tolist()
        assert a == b, f"{col}: engine {a} != script {b}"
    assert core["inpass_level"].tolist() == mine["inpass_level"].tolist()
    assert core["inpass_why"].tolist() == mine["inpass_why"].tolist()
    assert core["anchor_kind"].tolist() == mine["anchor_kind"].tolist()
    assert core["anchor_why"].tolist() == mine["anchor_why"].tolist()
    assert set(mine["anchor_kind"]) <= {"two routes", "ladder", "listed", "none"}
    assert core["n_series_excl"].tolist() == mine["n_series_excl"].tolist()
    return mine.reset_index()


def one(row, **kw):
    return both(_prepared([row], **kw)).iloc[0]


NITROPHENOL = dict(neutral_formula="C6H5NO3", adduct="[M-H]-", committed_matched="C", committed_matched_lines="13C")
ACID = dict(neutral_formula="C10H16O4", adduct="[M-H]-", committed_matched="C", committed_matched_lines="13C")
REGISTRY = LS.ContextLists("negative", "ambient-air", [])


# --------------------------------------------------------------------------- the levels, each with its mutant
def test_3c_and_its_mutant_without_the_named_entry():
    r = one(NITROPHENOL, lists=REGISTRY)
    assert r.evidence_level == "3c" and r.claim == "identified" and r.named_entry == "registry:nitroaromatic = nitrophenol"
    assert r.would_lift == "level 2 (MS2 / standards) is not automatic"
    m = one(NITROPHENOL)
    assert m.evidence_level == "4a" and m.would_lift == "3c needs a NAMED context-list entry naming the neutral"


def test_3c_needs_no_positive_fact_and_a_class_entry_is_no_name():
    assert one(dict(NITROPHENOL, committed_matched="", committed_matched_lines=""), lists=REGISTRY).evidence_level == "3c"
    r = one(dict(neutral_formula="C4HF7O2", adduct="[M-H]-", committed_matched="C"), lists=REGISTRY)   # perfluoroacid
    assert r.evidence_level == "4a" and r.class_entry == "registry:perfluoroacid"
    assert r.would_lift.endswith(" (the class-list match is a tag only)")


def test_4a_and_its_mutants():
    r = one(ACID)
    assert r.evidence_level == "4a" and r.split_rule == "only decomposition" and r.positive_fact.startswith(
        "own in-band isotope line(s) 13C (C of the neutral)")
    m = one(dict(ACID, committed_matched="", committed_matched_lines=""))
    assert m.evidence_level == "4b" and m.would_lift.endswith("; 3c needs a named context-list entry")
    assert one(dict(ACID, committed_matched="N", committed_matched_lines="15N")).evidence_level == "4b"


def test_4b_split_open_and_its_mutant_a_competitor_left():
    r = one(dict(ACID, adduct="[M+NO3]-"))
    assert r.evidence_level == "4b" and r.split_rule == "k decompositions"
    assert r.would_lift == "split not pinned: 2 decompositions: C10H16O4 [M+NO3]-; C10H17NO7 [M-H]-"
    m = one(dict(ACID, adduct="[M+NO3]-"), comps=[("C10H16O4", "[M+NO3]-", "C9H12O5", "[M+NO3]-", "left")])
    assert m.evidence_level == "5a" and m.competitors_left == "C9H12O5 [M+NO3]-"


def test_4b_ion_only_and_its_mutant_a_competitor_left():
    r = one(dict(ACID, adduct="[M]-.", ion_only=True, ion_only_reading=True))
    assert r.evidence_level == "4b" and r.would_lift == "ion-only channel: the neutral and the process stay open"
    m = one(dict(ACID, adduct="[M]-.", ion_only=True, ion_only_reading=True),
            comps=[("C10H16O4", "[M]-.", "C9H12O5", "[M]-.", "left")])
    assert m.evidence_level == "5a"


def test_5a_lists_every_competitor_and_its_mutant_the_isotopes_exclude_it():
    comps = [("C10H16O4", "[M-H]-", f"C{k}H12O5", "[M-H]-", "left") for k in range(5, 13)]
    r = one(ACID, comps=comps)
    assert r.evidence_level == "5a" and r.n_left == 8 and r.would_lift.endswith(" (+2 more)")
    assert one(ACID, comps=[c[:4] + ("excluded",) for c in comps]).evidence_level == "4a"


@pytest.mark.parametrize("change,lift", [
    (dict(lowconf=True, iso_veto=True), "refuted: iso_veto; lowconf"),
    (dict(label_veto=True), "refuted: label_veto"),
    (dict(committed_contradicted=True, committed_reasons="13C (0.11x) absent in 3/3 files"),
     "refuted: own isotopes: 13C (0.11x) absent in 3/3 files"),
    (dict(no_comp_info=True), "nothing could be enumerated or tested"),
])
def test_5b_and_its_mutant(change, lift):
    r = one(dict(ACID, **change))
    assert r.evidence_level == "5b" and r.would_lift == lift
    assert one(ACID).evidence_level == "4a"


def test_5b_untestable_beats_a_competitor_left():
    r = one(dict(ACID, no_comp_info=True), comps=[("C10H16O4", "[M-H]-", "C9H12O5", "[M-H]-", "left")])
    assert r.evidence_level == "5b" and r.would_lift == "nothing could be enumerated or tested"


def test_below_setters_o11_alone_is_a_tag_any_other_or_none_rejects():
    k = ("C10H16O4", "[M-H]-")
    r = one(dict(ACID, below=True), below={k: dict(setters=[LL.O11_SETTER])})
    assert r.evidence_level == "4a" and r.o11_tag
    r = one(dict(ACID, below=True), below={k: dict(setters=[LL.O11_SETTER, "O-monster"])})
    assert r.evidence_level == "5b" and r.would_lift == f"refuted: below: {LL.O11_SETTER} & O-monster"
    assert one(dict(ACID, below=True)).would_lift == "refuted: below: setter not found"
    assert LL.O11_SETTER == DC.O11_SETTER


def test_reagent_bucket_beats_a_rejection_and_its_mutant():
    r = one(dict(neutral_formula="HNO3", adduct="[M+NO3]-", lowconf=True))
    assert r.evidence_level == "reagent" and r.claim == "reagent" and r.reagent_identity == "HNO3"
    assert r.would_lift == "reagent ion / reagent cluster (not levelled)"
    assert one(dict(neutral_formula="H2O", adduct="[M+NO3]-")).evidence_level == "reagent"
    assert one(dict(neutral_formula="H2O", adduct="[M-H]-")).evidence_level != "reagent"   # water alone, bare


@pytest.mark.parametrize("n,a,pol", [("H2O", "[M+NO3]-", "negative"), ("H2O", "[M-H]-", "negative"),
                                     ("H2N2O6", "[M-H]-", "negative"), ("H3NO4", "[M+NO3]-", "negative"),
                                     ("HBr", "[M+Br]-", "negative"), ("H^NO3", "[M+^NO3]-", "negative"),
                                     ("HNO3", "[M+Cl]-", "negative"), ("HNO2", "[M-H]-", "negative"),
                                     ("CH6N2O2", "[M+H]+", "positive"), ("C2H8N4O2", "[M+NH4]+", "positive"),
                                     ("H3N", "[M+H]+", "positive"), ("H2O", "[M+H]+", "positive"),
                                     ("H2O", "[M+NH4]+", "positive"), ("C2H6O", "[M+H]+", "positive"),
                                     ("CH5NO", "[M+(CH4N2O)H]+", "positive")])
def test_reagent_identity_equals_the_engines(n, a, pol):
    assert LL.reagent_identity(n, a, pol) == DC.reagent_identity(n, a, pol)


# --------------------------------------------------------------------------- the split, rule by rule
def test_split_no_ion_composition_is_open():
    r = one(dict(neutral_formula="C10H16O4", adduct="junk"))
    assert r.split_rule == "no ion composition" and not r.split_pinned and r.evidence_level == "4b"
    assert r.would_lift == "split not pinned: no ion composition"


def test_split_label_pins_and_is_a_positive_fact_and_its_mutants():
    ctx = lambda: _Ctx(reagent="NO3_15N", decomp=["[M+^NO3]-", "[M-H]-"], labelled=True)  # noqa: E731
    lab = dict(neutral_formula="C10H16O4", adduct="[M+^NO3]-")
    r = one(lab, ctx=ctx())
    assert r.split_rule == "label" and r.evidence_level == "4a" and r.positive_fact == "15N label"
    alien = one(lab, ctx=ctx(), alien={("C10H16O4", "[M+^NO3]-")})            # rule K: the 14N twin argues
    assert alien.split_rule != "label"
    assert one(lab, ctx=_Ctx(reagent="NO3_15N", decomp=["[M+^NO3]-", "[M-H]-"])).split_rule != "label"  # unlabelled


def test_split_label_on_an_unlabelled_adduct_is_open():
    ctx = _Ctx(reagent="NO3_15N", decomp=["[M+^NO3]-", "[M-H]-"], labelled=True)
    r = one(dict(neutral_formula="C10H17^NO7", adduct="[M-H]-"), ctx=ctx)
    assert r.split_rule == "label on an unlabelled adduct" and not r.split_pinned
    assert r.would_lift == "split not pinned: label points to the cluster reading: C10H16O4 [M+^NO3]-"


def test_split_window_only_pin_and_the_reagent_isobar_opens():
    r = one(NITROPHENOL)                     # C6H4 [M+NO3]- fails the window alone: the pin stands, disclosed
    assert r.split_pinned and "[pinned only by the context window: C6H4 [M+NO3]-" in r.split_text
    iso = one(dict(neutral_formula="CH2O2", adduct="[M+NO3]-"))   # X+HNO3 [M-H]- fails the window alone: open
    assert not iso.split_pinned and "[X+reagent isobar admitted over the context window: CH3NO5 [M-H]- (X+HNO3; " \
        in iso.would_lift


def test_split_the_hbr_isobar_opens_a_bromide_cluster_in_engine_and_script():
    ctx = lambda: _Ctx(reagent="NO3", decomp=["[M+NO3]-", "[M-H]-", "[M+Br]-"])  # noqa: E731
    r = one(dict(ACID, neutral_formula="C4H6O4", adduct="[M+Br]-"), ctx=ctx())    # X+HBr [M-H]- fails the window alone
    assert not r.split_pinned and r.evidence_level == "4b"
    assert r.would_lift == ("split not pinned: 2 decompositions: C4H6O4 [M+Br]-; C4H7BrO4 [M-H]- [X+reagent isobar "
                            "admitted over the context window: C4H7BrO4 [M-H]- (X+HBr; window Br in neutral needs "
                            "C>=5 (got C=4); likely reagent alias)]")
    c5 = one(dict(ACID, neutral_formula="C5H8O4", adduct="[M+Br]-"), ctx=ctx())    # passes the window: a decomposition
    assert not c5.split_pinned and "isobar" not in c5.would_lift


def test_split_committed_neutral_not_plausible_is_open():
    r = one(dict(neutral_formula="C2H40O4", adduct="[M-H]-"))
    assert r.split_rule == "committed neutral not plausible" and r.evidence_level == "4b"
    assert r.would_lift.startswith("split not pinned: committed neutral not plausible (")


def _ur(rows, gate, **kw):
    return both(_prepared(rows, ctx=_Ctx(reagent="Ur", context="uronium", decomp=UR), pol="positive", gate=gate,
                          **kw))


def test_amine_gate_committed_nh4():
    row = dict(neutral_formula="C8H16O2", adduct="[M+NH4]+")
    r = _ur([row], _Gate(keep={"C8H16O2": (True, True, "tracks its own [M+H]+/urea parent, r 0.80 over 20 2-h bins")}))
    r = r.iloc[0]
    assert r.split_rule == "amine gate: committed NH4" and r.split_pinned and r.evidence_level == "4a"
    assert r.positive_fact == "NH4 adduct tracks its parent"
    kept = _ur([row], _Gate(keep={"C8H16O2": (True, False, "protected identity")})).iloc[0]
    assert kept.split_pinned and kept.evidence_level == "4b" and kept.would_lift.startswith("4a needs a positive fact")
    no = _ur([row], _Gate()).iloc[0]
    assert not no.split_pinned and no.would_lift == (
        "split not pinned: amine default (NH4 reading unconfirmed: parent channels absent; the engine's gate reads "
        "this ion as C8H19NO2 [M+H]+)")


def test_amine_gate_on_an_amine_reading():
    amine = dict(neutral_formula="C8H19N", adduct="[M+H]+")          # Y = C8H16 [M+NH4]+ is the alias
    kept = _ur([amine], _Gate(keep={"C8H16": (True, True, "tracks")})).iloc[0]
    assert not kept.split_pinned and kept.split_text.startswith("NH4 reading kept by the amine gate: C8H16 [M+NH4]+")
    default = _ur([amine], _Gate()).iloc[0]
    assert not default.split_pinned and default.split_text.startswith("amine default (NH4 reading unconfirmed: C8H16")
    other = _ur([amine], _Gate(other={"C8H16"})).iloc[0]
    assert "[NH4 admissibility rule: Y's uronium adduct ion is present under another reading: C8H16 -- " \
        in other.split_text
    # the NH4 admissibility rule: Y has no uronium adduct ion -> the NH4 reading leaves the set -> pinned
    inadm = _ur([amine], _Gate(absent={"C8H16"})).iloc[0]
    assert inadm.split_pinned and inadm.split_text.startswith(
        "NH4 reading inadmissible: no uronium adduct of Y (C8H16 [M+NH4]+ -- no uronium adduct ion of Y present")


def test_amine_gate_on_another_adduct_drops_or_keeps_the_nh4_reading():
    cl = dict(neutral_formula="C6H12O", adduct="[M+(CH4N2O)H]+")     # the same ion: C7H13NO2 [M+NH4]+
    dropped = _ur([cl], _Gate()).iloc[0]
    assert dropped.split_text == ("2 decompositions: C7H16N2O2 [M+H]+; C6H12O [M+(CH4N2O)H]+ [NH4 reading not kept "
                                  "by the amine gate: C7H13NO2 [M+NH4]+ -- parent channels absent]")
    kept = _ur([cl], _Gate(keep={"C7H13NO2": (True, False, "Si")})).iloc[0]
    assert kept.split_text.startswith("3 decompositions: ") and kept.split_text.endswith(
        "[NH4 reading kept by the amine gate: C7H13NO2 [M+NH4]+ (Si)]")
    # the mutant: no gate (a one-file source with no gate built) -- the NH4 reading stays live, no gate note
    off = _ur([cl], None).iloc[0]
    assert off.split_text.startswith("3 decompositions: ") and "amine gate" not in off.split_text


# --------------------------------------------------------------------------- step 1: the series exclusion
def _chain_files(n_files, anchors=("C8H12O4", "C10H16O4"), both_routes=True):
    """Per-file ledgers in which the anchors are seen on both nitrate routes in the same file."""
    pf = {}
    for k in range(n_files):
        rows = []
        for nn in ("C8H12O4", "C9H14O4", "C10H16O4"):
            rows.append(dict(role="M0", neutral_formula=nn, adduct="[M-H]-", method="grid", height=1e5,
                             mz=C.ion_mz(nn, "[M-H]-")))
            if both_routes and nn in anchors:
                rows.append(dict(role="M0", neutral_formula=nn, adduct="[M+NO3]-", method="grid", height=1e5,
                                 mz=C.ion_mz(nn, "[M+NO3]-")))
        pf[f"f{k}"] = pd.DataFrame(rows)
    return pf


CHAIN = [dict(neutral_formula=n, adduct="[M-H]-", committed_matched="C") for n in ("C8H12O4", "C9H14O4", "C10H16O4")]
CHAIN_COMP = [("C9H14O4", "[M-H]-", "C7H18O5", "[M-H]-", "left")]


def test_the_series_exclusion_lifts_a_5a_inside_an_anchored_series_and_its_mutants():
    rows = CHAIN + [dict(neutral_formula=n, adduct="[M+NO3]-") for n in ("C8H12O4", "C10H16O4")]
    out = both(_prepared(rows, comps=CHAIN_COMP, pf=_chain_files(2))).set_index("neutral_formula")
    r = out[out["adduct"] == "[M-H]-"]
    assert r.loc["C9H14O4", "evidence_level"] == "4a" and r.loc["C9H14O4", "n_series_excl"] == 1
    assert set(r.loc[["C8H12O4", "C10H16O4"], "inpass_why"]) == {"routes"}
    # one file: the routes need two co-files, nothing anchors the series, the competitor stays
    one_file = both(_prepared(rows, comps=CHAIN_COMP, pf=_chain_files(1))).set_index("neutral_formula")
    assert one_file[one_file["adduct"] == "[M-H]-"].loc["C9H14O4", "evidence_level"] == "5a"
    # ... unless the source is one file levelled alone (route co-files 1)
    arm = both(_prepared(rows, comps=CHAIN_COMP, pf=_chain_files(1), arm=True)).set_index("neutral_formula")
    assert arm[arm["adduct"] == "[M-H]-"].loc["C9H14O4", "evidence_level"] == "4a"
    # the competitor's own homologue is a member at the anchors' spacing: not excluded
    comp = [("C9H14O4", "[M-H]-", "C8H12O4", "[M-H]-", "left")]
    kept = both(_prepared(rows, comps=comp, pf=_chain_files(2))).set_index("neutral_formula")
    assert kept[kept["adduct"] == "[M-H]-"].loc["C9H14O4", "evidence_level"] == "5a"
    # an isotope-line competitor is never series-testable
    iso = [("C9H14O4", "[M-H]-", "C7H18O5", "[M-H]-", "left", "isoline")]
    il = both(_prepared(rows, comps=iso, pf=_chain_files(2))).set_index("neutral_formula")
    assert il[il["adduct"] == "[M-H]-"].loc["C9H14O4", "evidence_level"] == "5a"


def test_other_source_partners_anchor_a_series_only_on_a_pinned_split_and_never_on_a_one_file_source():
    partners = {n: {"protonation": [f"src {n} [M+H]+ two routes"]} for n in ("C8H12O4", "C10H16O4")}
    S = _prepared(CHAIN, comps=CHAIN_COMP, pf=_chain_files(2, both_routes=False))
    without = both(S).set_index("neutral_formula")
    assert without.loc["C9H14O4", "evidence_level"] == "5a"
    S = _prepared(CHAIN, comps=CHAIN_COMP, pf=_chain_files(2, both_routes=False))
    withp = both(S, partners).set_index("neutral_formula")
    assert withp.loc["C9H14O4", "evidence_level"] == "4a" and withp.loc["C8H12O4", "inpass_why"] == "routes"
    S = _prepared(CHAIN, comps=CHAIN_COMP, pf=_chain_files(2, both_routes=False), arm=True)
    assert both(S, partners).set_index("neutral_formula").loc["C9H14O4", "evidence_level"] == "5a"
    # a partner on a pair seen in one file is not counted
    S = _prepared([dict(r, n_files_obs=1) for r in CHAIN], comps=CHAIN_COMP, pf=_chain_files(2, both_routes=False))
    assert both(S, partners).set_index("neutral_formula").loc["C9H14O4", "evidence_level"] == "5a"


def test_an_other_source_partner_counts_only_on_a_pinned_split_in_engine_and_script():
    """The anchors' own splits are open (X.NO3- of C8H12O4 ... reads as the same ion): their other-source routes are
    not counted, nothing anchors the series, the middle member keeps its competitor."""
    chain = ("C8H13NO7", "C9H15NO7", "C10H17NO7")
    rows = [dict(neutral_formula=n, adduct="[M-H]-", committed_matched="C") for n in chain]
    comp = [("C9H15NO7", "[M-H]-", "C7H18O5", "[M-H]-", "left")]
    pf = {f"f{k}": pd.DataFrame([dict(role="M0", neutral_formula=n, adduct="[M-H]-", method="grid", height=1e5,
                                      mz=C.ion_mz(n, "[M-H]-")) for n in chain]) for k in range(2)}
    partners = {n: {"protonation": [f"src {n} [M+H]+ two routes"]} for n in (chain[0], chain[2])}
    S = _prepared(rows, comps=comp, pf=pf)
    out = both(S, partners).set_index("neutral_formula")
    assert out.loc["C9H15NO7", "evidence_level"] == "5a" and out.loc["C9H15NO7", "n_series_excl"] == 0
    assert set(out.loc[[chain[0], chain[2]], "anchor_kind"]) == {"none"}
    assert not out.loc[chain[0], "split_pinned"]
    core = _engine(_prepared(rows, comps=comp, pf=pf), partners).set_index("neutral_formula")
    assert "other-source route (protonation) not counted: this ion's own split is open" in core.loc[chain[0], "tags"]


def test_the_level_order_is_the_documented_one():
    base = dict(reagent="", rejected=[], untestable=False, left=[], ion_only=False, pinned=True, named=[{"x": 1}],
                posfact=True, track=False)
    assert LL.level_of(base) == "3c"
    assert LL.level_of(dict(base, named=[])) == "4a"
    assert LL.level_of(dict(base, named=[], posfact=False, track=True)) == "4a"
    assert LL.level_of(dict(base, named=[], posfact=False)) == "4b"
    assert LL.level_of(dict(base, pinned=False)) == "4b"
    assert LL.level_of(dict(base, ion_only=True)) == "4b"
    assert LL.level_of(dict(base, left=[{"name": "c"}])) == "5a"
    assert LL.level_of(dict(base, untestable=True, left=[{"name": "c"}])) == "5b"
    assert LL.level_of(dict(base, rejected=["lowconf"])) == "5a"                 # C47: the lowconf ceiling
    assert LL.level_of(dict(base, rejected=["lowconf", "iso_veto"])) == "5b"
    assert LL.level_of(dict(base, rejected=["lowconf"], left=[{"name": "c"}])) == "5a"
    assert LL.level_of(dict(base, rejected=["lowconf"], untestable=True)) == "5b"
    assert LL.level_of(dict(base, lead_reflist=True)) == "4a"                         # C47: 3c withheld
    assert LL.would_lift("4a", dict(base, lead_reflist=True, class_entry=[])) == LL.DC.LIFT_3C_WITHHELD
    assert LL.would_lift("5a", dict(base, rejected=["lowconf"])) == LL.DC.LOWCONF_CEILING_WHY
    assert LL.level_of(dict(base, reagent="HNO3", rejected=["lowconf"])) == "reagent"
    assert [r for r, _k in LL.SPLIT_RULES][:2] == ["no ion composition", "ion-only channel"]


# --------------------------------------------------------------------------- whole sources, the class gate, the CLI
def _file(rows):
    led = L.new_ledger(pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(rows))],
                                     "mz": [C.ion_mz(n, a) for n, a in rows], "height": [1e5] * len(rows)}))
    for i, (n, a) in enumerate(rows):
        L.commit_assignment(led, f"p{i}", neutral_formula=n, adduct=a, ion_formula=f"{n}{a[-1]}", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="x")
    return led


def _summary(resolution, cal=(0.0, 0.3)):
    """``cal`` = the file's persisted degeneracy_cal (mu, sigma); None = an uncalibrated file."""
    return dict(reagent="NO3", context="ambient-air", reflists_active=[], resolution=resolution,
                per_file=[dict(sample_id="f1", height_gate_cps=1e3,
                               degeneracy_cal=None if cal is None else {"mu": cal[0], "sigma": cal[1]})])


PAIRS = [("C10H16O4", "[M-H]-"), ("C9H14O4", "[M-H]-"), ("C6H5NO3", "[M-H]-"), ("HNO3", "[M+NO3]-"),
         ("C10H16O4", "[M+NO3]-")]


def test_a_whole_source_levels_alike_through_the_engines_entry_point():
    src = EV.source_from_frames({"f1": _file(PAIRS)}, run_inputs=SRC.RunInputs(summary=_summary(ORBI)),
                                mode="adapted")
    core = EV.level_source(src).set_index(["neutral_formula", "adduct"])
    mine = LL.decide_source(src).set_index(["neutral_formula", "adduct"])
    assert sorted(core.index) == sorted(mine.index) and len(mine) == len(PAIRS)
    for col in ("evidence_level", "would_lift", "competitors_left", "claim"):
        assert core.loc[mine.index, col].tolist() == mine[col].tolist(), col
    assert mine.loc[("HNO3", "[M+NO3]-"), "evidence_level"] == "reagent"


@pytest.mark.parametrize("mode", ["adapted", "run"])
def test_a_source_with_no_committed_pair_gives_no_row_in_script_and_engine(mode):
    src = EV.source_from_frames({"f1": _file([])}, run_inputs=SRC.RunInputs(summary=_summary(ORBI)), mode=mode)
    mine = LL.decide_source(src)
    assert mine.empty and list(mine.columns) == LL.DECISION_COLUMNS
    assert EV.level_source(src).empty


@pytest.mark.parametrize("res,why", [(TOF, "tof"), (None, "class-less")])
def test_the_class_gate_reads_na_and_its_mutant_is_levelled(res, why):
    src = EV.source_from_frames({"f1": _file(PAIRS)}, run_inputs=SRC.RunInputs(summary=_summary(res)), mode="adapted")
    out = LL.decide_source(src)
    assert set(out["evidence_level"]) == {"NA"} and set(out["claim"]) == {"not assessed"} and len(out) == len(PAIRS)
    assert out["evidence_level"].tolist() == EV.level_source(src).set_index(["neutral_formula", "adduct"]).loc[
        list(zip(out["neutral_formula"], out["adduct"])), "evidence_level"].tolist()


def _run_dir(root: Path, name="BATCH_2026-01-01T000000Z", resolution=ORBI, cal=(0.0, 0.3)) -> Path:
    import json
    rd = root / name
    (rd / "per_file").mkdir(parents=True)
    (rd / "tables").mkdir()
    led = _file(PAIRS)
    led["ppm_error_cal"] = led["ppm_error"]          # what the audit stage writes on a real ledger
    led.to_csv(rd / "per_file" / "f1_ledger.csv", index=False)
    m0 = led[led["role"] == "M0"]
    m0.iloc[:-1].to_csv(rd / "merged_ledger.csv", index=False)      # the last reading: no merged row
    (rd / "batch_summary.json").write_text(json.dumps(_summary(resolution, cal)))
    return rd


def test_the_cli_levels_a_run_dir_prints_the_vector_and_writes_the_rows(tmp_path, capsys):
    rd = _run_dir(tmp_path)
    out = tmp_path / "levels.csv"
    assert LL.main([str(rd), "--mode", "adapted", "--out", str(out), "--vector"]) == 0
    text = capsys.readouterr().out
    df = pd.read_csv(out, keep_default_na=False)
    assert len(df) == len(PAIRS) and set(df["source"]) == {rd.name}
    # the internal pass's raw tokens stay in memory; the table names the anchor
    assert not set(EV.INTERNAL_COLUMNS) & set(df.columns) and {"anchor_kind", "anchor_why"} <= set(df.columns)
    assert "3c/4a/4b/5a/5b/reagent/NA" in text and "% of committed M0 height:" in text and "unmatched 20.00" in text
    v = LL.vector(df)
    assert list(v) == ["3c", "4a", "4b", "5a", "5b", "reagent", "NA"] and sum(v.values()) == len(PAIRS)
    share = LL.height_share(df, {"f1": pd.read_csv(rd / "per_file" / "f1_ledger.csv")},
                            pd.read_csv(rd / "merged_ledger.csv"))
    assert share["unmatched"] == pytest.approx(20.0) and sum(share.values()) == pytest.approx(100.0)
    # the out dir that holds the run dir is found; a lone ledger without a width model reads NA
    assert LL.run_dir_of(str(tmp_path)) == str(rd)
    lone = LL.run([str(rd / "per_file" / "f1_ledger.csv")], log=lambda *a: None)
    assert set(lone["evidence_level"]) == {"NA"}


def test_a_lone_ledger_with_a_declared_width_model_is_levelled(tmp_path):
    rd = _run_dir(tmp_path)
    path = str(rd / "per_file" / "f1_ledger.csv")
    got = LL.run([path], resolving_power=100000.0, reagent="NO3", window=(0.0, 0.3), log=lambda *a: None)
    assert "NA" not in set(got["evidence_level"]) and len(got) == len(PAIRS)
    # the same file in memory through the engine's per-file stage
    src = EV.source_from_frames({"f1": pd.read_csv(path)}, run_inputs=EV.file_run_inputs(
        sample_id="f1", reagent="NO3", context=None, resolution=LL.resolution_for(100000.0),
        degeneracy_cal=(0.0, 0.3)), mode="adapted")
    core = EV.level_source(src).set_index(["neutral_formula", "adduct"])
    assert core.loc[list(zip(got["neutral_formula"], got["adduct"])), "would_lift"].tolist() == got["would_lift"].tolist()
    # no calibration and nothing to refit it from: no level, and why
    blind = LL.run([path], resolving_power=100000.0, reagent="NO3", log=lambda *a: None)
    assert set(blind["evidence_level"]) == {""} and set(blind["claim"]) == {"tentative"}
    assert set(blind["would_lift"]) == {EV.NO_WINDOW_TEXT}
    with pytest.raises(SystemExit, match="--reagent"):
        LL.run([str(rd / "per_file" / "f1_ledger.csv")], resolving_power=100000.0, log=lambda *a: None)
    assert Resolution.from_dict(LL.resolution_for(70000.0)).r_at(200.0) == pytest.approx(70000.0)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "levels"


@pytest.mark.parametrize("fixture,reagent,sid", [("v1_uronium", "Ur", "s01"), ("v1_nitrate", "NO3+NO3_15N", "s01")])
def test_a_lone_ledger_is_levelled_as_the_per_file_stage_of_a_single_sample_run(tmp_path, fixture, reagent, sid):
    """level_ledger.py <ledger.csv> --resolving-power R --reagent P [--window MU,SIGMA] == the per-file stage of
    `peaky assign` on that file: the profile's context, the lists that context and the profile label activate
    (the always-active ones included), the profile channels' halogen."""
    import gzip
    import shutil
    from types import SimpleNamespace

    from peaky.assignment import assign as A
    from peaky.assignment import reflists as RL
    from peaky.chem import contexts
    from peaky.chem import profiles as PR
    path = tmp_path / f"{sid}_ledger.csv"
    with gzip.open(FIXTURES / fixture / "per_file" / f"{sid}_ledger.csv.gz", "rb") as f, open(path, "wb") as g:
        shutil.copyfileobj(f, g)
    r200, window = 104_000.0, (0.0, 0.2)
    prof = PR.resolve(reagent)
    # the per-file stage itself, with the inputs `peaky assign` gives it (cli.py's activation, assign.run's state)
    lists, _tags, record = RL.activate(prof.context, prof.label, record=True, fields=("context", "reagent label"))
    assert any(L.id == "contaminants_keller2008" for L in lists)
    led = pd.read_csv(path, low_memory=False)
    st = SimpleNamespace(cfg=None, reagent_profile=prof.name, adducts=list(prof.adducts),
                         profile=contexts.get_context(prof.context), resolving_power=LL.resolution_for(r200),
                         reflists_active=lists, reflists_context=record, degeneracy_cal=window,
                         reagent_halogen=EV.channel_halogen(prof.adducts), sample_id=sid, led=led,
                         log=lambda *a: None)
    A._stage_evidence(st)
    stage = (led[led["role"] == "M0"].drop_duplicates(["neutral_formula", "adduct"])
             .set_index(["neutral_formula", "adduct"]))
    lone = LL.run([str(path)], resolving_power=r200, reagent=reagent, window=window,
                  log=lambda *a: None).set_index(["neutral_formula", "adduct"])
    assert len(lone) == len(stage) and set(lone.index) == set(stage.index)
    for col in ("evidence_level", "would_lift", "claim", "competitors_left"):
        want = stage.loc[lone.index, col].fillna("").astype(str).tolist()
        assert lone[col].fillna("").astype(str).tolist() == want, col


def test_corroborate_takes_partners_only_from_an_orbitrap_class_run_dir(tmp_path):
    seen = []
    LL.partners_of([str(_run_dir(tmp_path / "tof", resolution=TOF)), str(tmp_path / "nothing")], log=seen.append)
    assert any("tof source -- no partners" in s for s in seen) and any("not a run dir" in s for s in seen)
    got = LL.partners_of([str(_run_dir(tmp_path / "orbi"))], log=seen.append)
    assert isinstance(got, dict)


def test_a_corroborate_run_dir_with_no_calibrated_file_gives_no_partners_in_the_batch_and_the_script(tmp_path):
    """--corroborate an Orbitrap-class run dir none of whose files is calibrated: it has no run window, so its
    pairs carry no level and none anchors a partner. The batch and the script both skip it and log the same
    reason (the batch used to crash on the unlevelled frame); the same dir calibrated is levelled and asked."""
    from peaky.batch import assign_batch as AB
    uncal = _run_dir(tmp_path / "uncal", cal=None)
    assert EV.no_run_window(EV.source_from_run_dir(str(uncal)))
    seen, mine = [], []
    parts, counts = AB._corroborate_partners([str(uncal)], log=seen.append)
    assert dict(parts) == {} and counts == {}
    assert len(seen) == 1 and EV.NO_RUN_WINDOW_PARTNERS in seen[0], seen
    assert dict(LL.partners_of([str(uncal)], log=mine.append)) == {}
    assert len(mine) == 1 and EV.NO_RUN_WINDOW_PARTNERS in mine[0], mine
    # the control: calibrated, the same run dir is levelled and its partners taken (no skip, its count recorded)
    cal = _run_dir(tmp_path / "cal")
    seen.clear()
    mine.clear()
    _parts, counts = AB._corroborate_partners([str(cal)], log=seen.append)
    assert list(counts) == [cal.name] and not any(EV.NO_RUN_WINDOW_PARTNERS in s for s in seen)
    LL.partners_of([str(cal)], log=mine.append)
    assert not any(EV.NO_RUN_WINDOW_PARTNERS in s for s in mine) and "partner neutral(s)" in mine[0]


@pytest.mark.parametrize("edit, why", [
    ({"reagent": "UrCustom"}, "its reagent profile 'UrCustom' is not registered here"),
    ({"reagent": None}, "its batch summary names no reagent profile"),
    ({"context": "no-such-context"}, "its context 'no-such-context' is not known here"),
])
def test_a_corroborate_run_dir_this_process_cannot_level_gives_no_partners_and_never_crashes(tmp_path, edit, why):
    """--corroborate an Orbitrap-class run dir made under a profile this process does not know (a
    --reagent-config run), or whose summary names none: the batch and the script skip it with the reason
    (the batch used to crash with a KeyError after every per-file assignment); the partner tag is an extra."""
    import json
    from peaky.batch import assign_batch as AB
    rd = _run_dir(tmp_path / "custom")
    summ = json.loads((rd / "batch_summary.json").read_text())
    summ.update(edit)
    if summ.get("reagent") is None:
        summ.pop("reagent")
    (rd / "batch_summary.json").write_text(json.dumps(summ))
    assert EV.partner_source_problem(str(rd)) == why
    seen, mine = [], []
    parts, counts = AB._corroborate_partners([str(rd)], log=seen.append)
    assert dict(parts) == {} and counts == {}
    assert len(seen) == 1 and why in seen[0] and "no other-source partners" in seen[0], seen
    assert dict(LL.partners_of([str(rd)], log=mine.append)) == {}
    assert len(mine) == 1 and why in mine[0], mine
    assert EV.partner_source_problem(str(_run_dir(tmp_path / "known"))) == ""


def test_partners_are_computed_only_for_a_source_that_can_take_them(tmp_path, monkeypatch):
    tof, orbi = _run_dir(tmp_path / "tof", resolution=TOF), _run_dir(tmp_path / "orbi")
    asked = []
    monkeypatch.setattr(LL, "partners_of", lambda paths, log=print: asked.append(list(paths)) or {})
    assert set(LL.run([str(tof)], [str(orbi)], log=lambda *a: None)["evidence_level"]) == {"NA"} and asked == []
    LL.run([str(orbi)], [str(orbi)], mode="adapted", log=lambda *a: None)          # one-file minima: no partners
    assert asked == []
    LL.run([str(orbi)], [str(orbi)], log=lambda *a: None)
    assert asked == [[str(orbi)]]


def test_the_b_series_reference_is_kept_under_its_series_names():
    assert LL.SERIES_LEVEL_ORDER[:3] == ["1", "2a", "2b"] and callable(LL.series_run) and callable(LL.series_main)
    assert callable(LL.measure_source) and callable(LL.assign_levels) and callable(LL.series_level_of)
    assert LL.LEVEL_ORDER == SC.LEVEL_ORDER and LL.VECTOR_KEYS == SC.LEVELS + SC.BUCKETS
