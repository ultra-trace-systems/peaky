"""D4: the evidence scale's space opens a contaminant family only on real
evidence, and without cross-family mixtures.

1. A family committed rows opened (`contaminant:<family>` M0 rows) counts for a
   file only when a row of it is Assigned or its own diagnostic isotope line
   (34S / 29Si / 30Si / 37Cl / 81Br) is committed; 'fluorinated' needs a
   CF2-step chain of >= 3 Assigned F readings (`context.family_evidence`). The
   context's declared pass-3 families and the reagent's family are kept.
2. A family's profile carries another heteroatom of the context's budget only
   when an open family brings it (`space.level_profiles`): no C15H32FPSi alias.
3. What only the shadow space (every opened family, with mixtures) proposes is
   listed in `competitors_left` as not counted, never a competitor -- unless the
   committed reading needs the family itself (a curated perfluoroacid on a run
   whose fluorine has no evidence: `space.space_of`) or carries a heteroatom no
   own in-band line confirms (`competitors.unconfirmed_heteroatoms`).

Synthetic, offline."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_levels_decide as TD  # noqa: E402

from peaky.assignment import degeneracy as DG  # noqa: E402
from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment.levels import competitors as CP  # noqa: E402
from peaky.assignment.levels import context as CX  # noqa: E402
from peaky.assignment.levels import decide as DC  # noqa: E402
from peaky.assignment.levels import space as SP  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.chem.resolution import Resolution  # noqa: E402

ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()     # R(200) = 100 000
UR_FAMS = ("amine", "siloxane", "pdms", "glycol_peg", "phthalate")                  # the uronium context's own
CHO_UREA = ("C16H22O", "[M+(CH4N2O)H]+")                                            # a CHO urea adduct ...
FPSI = "C15H32FPSi [M+H]+"                                                          # ... and its 0.06 mDa alias


def _ledger(rows):
    """rows: (peak_id, neutral, adduct, tier, method)."""
    led = L.new_ledger(pd.DataFrame({"peak_id": [r[0] for r in rows],
                                     "mz": [C.ion_mz(r[1], r[2]) for r in rows], "height": [1e5] * len(rows)}))
    for p, f, a, tier, meth in rows:
        L.commit_assignment(led, p, neutral_formula=f, adduct=a, ion_formula=None, ion_score=0.9,
                            compound_score=0.9, ppm_error=0.0, pass_no=3 if meth.startswith("contaminant") else 1,
                            method=meth, confidence="Good", commentary="x")
    led["tier"] = led["peak_id"].map({r[0]: r[3] for r in rows})
    return led


def _child(led, parent, label):
    """Append the committed isotope child ``label`` of row ``parent``."""
    row = dict(peak_id=f"{parent}-{label}", mz=float(led.loc[led["peak_id"] == parent, "mz"].iloc[0]) + 1.0,
               height=1e3, role="iso_child", parent_peak_id=parent, iso_label=label)
    return pd.concat([led, pd.DataFrame([row])], ignore_index=True)


def _summary(per_file, reagent="Ur", context="uronium"):
    return dict(reagent=reagent, context=context, reflists_active=[], resolution=ORBI, per_file=list(per_file))


JUNK_F = ("j1", "C10H9F5O", "[M+H]+", "Candidate", "contaminant:fluorinated")
PFCA = [(f"f{k}", f"C{k}HF{2 * k - 1}O2", "[M-H]-", "Assigned", "known:perfluoroacid") for k in (2, 3, 4)]


# --------------------------------------------------------------------------- 1. family evidence
class TestFamilyEvidence:
    def test_fluorine_needs_a_cf2_chain_of_three_assigned_readings(self):
        assert CX.family_evidence(_ledger([JUNK_F]), "fluorinated") == ""
        led = _ledger([JUNK_F] + PFCA)
        assert CX.family_evidence(led, "fluorinated") == "a CF2-step chain of 3 Assigned F readings"
        assert CX.family_evidence(_ledger([JUNK_F] + PFCA[:2]), "fluorinated") == ""       # a chain of 2
        cand = [(p, f, a, "Candidate", m) for p, f, a, _t, m in PFCA]
        assert CX.family_evidence(_ledger([JUNK_F] + cand), "fluorinated") == ""            # Candidate members
        mixed = [PFCA[0], PFCA[1], (PFCA[2][0], PFCA[2][1], "[M+NO3]-", "Assigned", PFCA[2][4])]
        assert CX.family_evidence(_ledger(mixed), "fluorinated") == ""                      # one adduct per chain
        # one Assigned fluorinated row alone is no chain: its tier rests on mass alone
        one = ("j2", "C8HF15O2", "[M-H]-", "Assigned", "contaminant:fluorinated")
        assert CX.family_evidence(_ledger([one]), "fluorinated") == ""

    def test_cf2_chain_is_the_longest_connected_run_on_one_adduct(self):
        rows = [(f"C{k}HF{2 * k - 1}O2", "[M-H]-") for k in (2, 3, 4, 6, 7)]
        assert CX.cf2_chain(rows) == 3
        assert CX.cf2_chain(rows + [("C5HF9O2", "[M-H]-")]) == 6
        assert CX.cf2_chain([]) == 0 and CX.cf2_chain([(None, "[M-H]-"), ("C2HF3O2", "[M-H]-")]) == 1

    def test_an_isotope_family_opens_on_an_assigned_row_or_its_own_line(self):
        dbp = ("h1", "C6H4Cl2O", "[M+H]+", "Candidate", "contaminant:halogen_dbp")
        led = _ledger([dbp])
        assert CX.family_evidence(led, "halogen_dbp") == ""
        assert CX.family_evidence(_child(led, "h1", "13C"), "halogen_dbp") == ""            # not diagnostic
        assert CX.family_evidence(_child(led, "h1", "37Cl"), "halogen_dbp") == "its own 37Cl line"
        assert CX.family_evidence(_child(led, "h1", "13C+37Cl"), "halogen_dbp") == "its own 37Cl line"
        led2 = led.copy()
        led2.loc[led2["peak_id"] == "h1", "isotopologues"] = '[{"label": "37Cl", "score": 0.9, "peak_id": "x"}]'
        assert CX.family_evidence(led2, "halogen_dbp") == "its own 37Cl line"
        assert CX.family_evidence(_ledger([dbp[:3] + ("Assigned", dbp[4])]), "halogen_dbp") == "an Assigned row"
        # another family's line or row is no evidence for this one
        sil = ("s1", "C6H18O3Si3", "[M+H]+", "Candidate", "contaminant:siloxane")
        assert CX.family_evidence(_child(_ledger([sil]), "s1", "37Cl"), "siloxane") == ""
        assert CX.family_evidence(_child(_ledger([sil]), "s1", "29Si"), "siloxane") == "its own 29Si line"
        assert CX.family_evidence(_child(_ledger([sil]), "s1", "29Si"), "halogen_dbp") == ""

    def test_a_line_the_adduct_carries_is_no_evidence_for_the_neutral(self):
        """The 81Br line of a reagent cluster [M+Br]- is the adduct's: only a line of an element the NEUTRAL
        carries, and its adduct adds none of, is the row's own."""
        no_br = ("h1", "C6H5ClO", "[M+Br]-", "Candidate", "contaminant:halogen_dbp")
        led = _ledger([no_br])
        assert CX.family_evidence(_child(led, "h1", "81Br"), "halogen_dbp") == ""
        assert CX.family_evidence(_child(led, "h1", "37Cl"), "halogen_dbp") == "its own 37Cl line"
        both = ("h2", "C6H4BrClO", "[M+Br]-", "Candidate", "contaminant:halogen_dbp")
        assert CX.family_evidence(_child(_ledger([both]), "h2", "81Br"), "halogen_dbp") == ""
        own = ("h3", "C6H4BrClO", "[M-H]-", "Candidate", "contaminant:halogen_dbp")
        assert CX.family_evidence(_child(_ledger([own]), "h3", "81Br"), "halogen_dbp") == "its own 81Br line"

    def test_a_diagnostic_line_counts_in_every_label_spelling(self):
        """The ledgers spell halogen lines several ways (`isotopes.parse_label_part`): each one is the line."""
        dbp = ("h1", "C6H4Cl2O", "[M+H]+", "Candidate", "contaminant:halogen_dbp")
        for lab in ("37Cl2", "2x37Cl", "37Cl(pair)", "13C+37Cl2"):
            assert CX.family_evidence(_child(_ledger([dbp]), "h1", lab), "halogen_dbp") == "its own 37Cl line", lab
        mix = ("h2", "C6H4BrClO", "[M-H]-", "Candidate", "contaminant:halogen_dbp")
        assert CX.family_evidence(_child(_ledger([mix]), "h2", "81Br37Cl(pair)"), "halogen_dbp") == \
            "its own 37Cl/81Br line"
        assert CX.family_evidence(_child(_ledger([mix]), "h2", "81Br/37Cl(pair)"), "halogen_dbp") == \
            "its own 37Cl/81Br line"
        assert CX.family_evidence(_child(_ledger([dbp]), "h1", "M+2"), "halogen_dbp") == ""   # a generic line
        assert CX._diag_isotopes(["81Br(pair)", "2x34S", "30Si+13C", "odd"]) == {"81Br", "34S", "30Si", "13C", "odd"}

    def test_evidenced_families_keep_the_contexts_declared_families(self):
        s = _summary([])
        led = _ledger([("p1",) + CHO_UREA + ("Assigned", "cheminfo+grid"), JUNK_F,
                       ("s1", "C6H18O3Si3", "[M+H]+", "Candidate", "contaminant:siloxane")])
        assert CX.file_families(s, led) == UR_FAMS + ("fluorinated",)
        assert CX.evidenced_families(s, led) == UR_FAMS            # siloxane is declared; fluorine has no evidence
        assert CX.evidenced_families(s, _ledger([JUNK_F] + PFCA)) == UR_FAMS + ("fluorinated",)
        assert CX.evidenced_families(s, led, ()) == ()


# --------------------------------------------------------------------------- 2. no cross-family mixtures
class TestLevelProfiles:
    def test_a_family_profile_keeps_only_the_elements_of_open_families(self):
        fams = UR_FAMS + ("fluorinated",)
        old, new = DG.space_profiles("uronium", fams), SP.level_profiles("uronium", fams)
        assert new[0] == old[0] and len(new) == len(old) == 2                  # the context profile is unchanged
        f = new[1]
        assert (f.max_F, f.max_N, f.max_Si, f.max_S, f.max_P) == (17, 5, 12, 0, 0)   # Si: siloxane / pdms open
        assert (old[1].max_S, old[1].max_P) == (2, 1)
        g = SP.level_profiles("uronium", ("amine", "fluorinated"))[1]
        assert (g.max_F, g.max_Si, g.max_S, g.max_P) == (17, 0, 0, 0)
        a = SP.level_profiles("ambient-air", ("organosulfate", "nitrate", "siloxane", "amine", "fluorinated"))[1]
        assert (a.max_F, a.max_S, a.max_Si, a.max_Cl, a.max_Br) == (17, 1, 1, 0, 0)  # no Cl / Br family open
        assert SP.level_profiles("uronium", UR_FAMS) == DG.space_profiles("uronium", UR_FAMS)

    def test_a_family_the_audit_adds_no_space_for_adds_none_here(self, monkeypatch):
        """Each family's profile is the degeneracy audit's own: a family `space_profiles` skips is skipped."""
        real = DG.space_profiles
        monkeypatch.setattr(DG, "space_profiles",
                            lambda ctx, fams=(): real(ctx, tuple(f for f in fams if f != "fluorinated")))
        assert SP.level_profiles("uronium", UR_FAMS + ("fluorinated",)) == SP.level_profiles("uronium", UR_FAMS)

    def test_the_level_space_drops_the_fpsi_alias_and_the_shadow_keeps_it(self):
        fams = UR_FAMS + ("fluorinated",)
        mz = C.ion_mz(*CHO_UREA)
        names = {sp.mixtures: sorted(f"{v[0]} {v[1]}" for v in sp.enumerate(mz, -0.9, 0.9, sp.adducts).values())
                 for sp in (SP.Space("Ur", "uronium", [], fams), SP.Space("Ur", "uronium", [], fams, mixtures=False))}
        assert FPSI in names[True] and FPSI not in names[False]
        assert "C17H26N2O2 [M+H]+" in names[False]                             # the urea adduct's own ion
        sp = SP.Space("Ur", "uronium", [], fams, mixtures=False)
        assert sp.space_reason("C15H32FPSi") is not None and sp.relaxed("C15H32FPSi") is not None


# --------------------------------------------------------------------------- the run context and the competitors
def _run(f_rows, *, cal=True):
    rows = [("p1",) + CHO_UREA + ("Assigned", "cheminfo+grid")] + f_rows
    pf = {s: _ledger(rows) for s in ("a", "b")}
    stats = [dict(sample_id=s, height_gate_cps=100.0, **({"degeneracy_cal": {"mu": 0.0, "sigma": 0.3}} if cal else {}))
             for s in pf]
    s = _summary(stats)
    win = CX.run_windows(s, pf) if cal else {sid: dict(fams=CX.file_families(s, led), fit=(0.0, 0.3))
                                             for sid, led in pf.items()}
    return s, pf, CX.build_run_context(s, pf, win, name="t")


def _cands(ctx, pf, key=CHO_UREA):
    facts = pd.DataFrame([dict(neutral_formula=key[0], adduct=key[1], mz=C.ion_mz(*key), ion="", ppm=0.0)])
    pair = CX.pairs_from_files(pf, facts)[key]
    return CP.build_candidates(ctx, pair)[0]


class TestRunContext:
    @pytest.mark.parametrize("cal", [True, False])
    def test_junk_fluorine_rows_open_no_family_in_the_level_space(self, cal):
        _s, pf, ctx = _run([JUNK_F], cal=cal)
        assert ctx.families == UR_FAMS and ctx.families_unevidenced == ("fluorinated",)
        assert ctx.families_dropped == () and ctx.space_shadow is not None
        assert ctx.space_shadow.families == UR_FAMS + ("fluorinated",) and ctx.space_shadow.mixtures
        if cal:
            assert ctx.window_records["a"]["fams_ev"] == UR_FAMS
        cands = _cands(ctx, pf)
        assert [c["name"] for c in cands[1:]] == []                                 # nothing left: no 5a
        assert [c["name"] for c in cands[0]["uncounted"]] == [FPSI]                  # ... and the alias is listed

    def test_a_real_cf2_chain_opens_fluorine_without_the_p_and_s_mixtures(self):
        _s, pf, ctx = _run([JUNK_F] + PFCA)
        assert ctx.families == UR_FAMS + ("fluorinated",) and ctx.families_unevidenced == ()
        assert ctx.space_shadow is not None                                          # the mixtures differ
        cands = _cands(ctx, pf)
        assert FPSI not in [c["name"] for c in cands[1:]]
        assert FPSI in [c["name"] for c in cands[0]["uncounted"]]

    def test_no_opened_family_no_shadow(self):
        _s, pf, ctx = _run([])
        assert ctx.families == UR_FAMS and ctx.space_shadow is None and ctx.families_unevidenced == ()
        assert _cands(ctx, pf)[0]["uncounted"] == []

    def test_an_unevidenced_family_that_adds_no_profile_builds_no_shadow(self):
        """On ambient air the context's budget already holds Cl and Br, so the audit adds no profile for a
        halogen family: refused for want of evidence, it changes nothing, and nothing is enumerated twice."""
        rows = [("p1", "C10H16O4", "[M-H]-", "Assigned", "cheminfo+grid"),
                ("h1", "C6H4Cl2O", "[M-H]-", "Candidate", "contaminant:halogen_dbp")]
        pf = {s: _ledger(rows) for s in ("a", "b")}
        stats = [dict(sample_id=s, height_gate_cps=100.0, degeneracy_cal={"mu": 0.0, "sigma": 0.3}) for s in pf]
        s = _summary(stats, reagent="NO3", context="ambient-air")
        ctx = CX.build_run_context(s, pf, CX.run_windows(s, pf), name="t")
        assert ctx.families_unevidenced == ("halogen_dbp",) and ctx.space_shadow is None
        assert DG.space_profiles(ctx.space.ctx, ctx.families + ("halogen_dbp",)) == ctx.space.profiles
        assert _cands(ctx, pf, rows[0][1:3])[0]["uncounted"] == []

    def test_an_engine_alternative_only_the_shadow_admits_is_not_counted(self):
        _s, pf, ctx = _run([JUNK_F])
        for led in pf.values():
            led.loc[led["peak_id"] == "p1", "alternatives"] = (
                '[{"formula": "C15H32FPSi", "adduct": "[M+H]+", "ppm": 0.2}, '
                '{"formula": "C17H26N2O2", "adduct": "[M+H]+", "ppm": 0.0}]')
            led.loc[led["peak_id"] == "p1", "tied"] = True
        cands = _cands(ctx, pf)
        assert [c["name"] for c in cands[1:]] == []
        assert [c["name"] for c in cands[0]["uncounted"]] == [FPSI]

    def test_a_reading_that_needs_the_family_is_levelled_as_before(self):
        """A committed F reading on a run whose fluorine has no evidence (any neutral the level space does not
        admit) is levelled in the shadow space exactly as before: its own family's competitors count (a tie with
        an F4 reading stays a competitor), and its decompositions and aliases read the shadow (`space.space_of`);
        nothing is 'not counted'."""
        import copy
        key = JUNK_F[1:3]
        _s, pf, ctx = _run([JUNK_F])
        for led in pf.values():
            led.loc[led["peak_id"] == "j1", "alternatives"] = (
                '[{"formula": "C17H32F4", "adduct": "[M+H]+", "ppm": 0.1}]')
            led.loc[led["peak_id"] == "j1", "tied"] = True
        assert ctx.space.space_reason(key[0]) is not None
        assert SP.space_of(ctx, key[0]) is ctx.space_shadow and SP.space_of(ctx, CHO_UREA[0]) is ctx.space
        assert ctx.space_shadow.space_reason("C5H12O20") is not None        # outside both: D5-widened, as before
        assert SP.space_of(ctx, "C5H12O20") is ctx.space_shadow
        before = copy.copy(ctx)
        before.space, before.space_shadow = ctx.space_shadow, None        # the scale's space before D4
        got, want = _cands(ctx, pf, key), _cands(before, pf, key)
        assert got[0]["uncounted"] == [] and "C17H32F4 [M+H]+" in [c["name"] for c in got[1:]]
        assert [(c["name"], c["kind"]) for c in got] == [(c["name"], c["kind"]) for c in want]
        cnt = SP.ion_counts_of(*key)
        assert ctx.decompositions(cnt, *key) == before.decompositions(cnt, *key)
        assert all(d["ok"] for d in ctx.decompositions(cnt, *key) if d["adduct"] == key[1])

    def test_a_curated_fluorine_reading_meets_its_own_familys_competitors(self):
        """A curated formula passes every profile of the level space (the pass-0 registry), so `space_of` judges
        it by its heteroatoms: a perfluoroacid on a run whose fluorine has no evidence (a CF2 chain of 1) is
        levelled in the shadow, and its tied fluorine alternative counts, as for any fluorine reading. A curated
        formula whose elements a level profile carries stays in the level space."""
        junk = ("j1", "C10H9F5O2", "[M-H]-", "Candidate", "contaminant:fluorinated")
        pfca = ("k1", "C4HF7O2", "[M-H]-", "Candidate", "known:perfluoroacid")
        pf = {s: _ledger([junk, pfca]) for s in ("a", "b")}
        stats = [dict(sample_id=s, height_gate_cps=100.0, degeneracy_cal={"mu": 0.0, "sigma": 0.3}) for s in pf]
        summ = _summary(stats, reagent="NO3", context="ambient-air")
        ctx = CX.build_run_context(summ, pf, CX.run_windows(summ, pf), name="t")
        assert ctx.families_unevidenced == ("fluorinated",) and "C4HF7O2" in ctx.space.curated
        assert ctx.space.space_reason("C4HF7O2") is None                         # the curated bypass ...
        assert SP.space_of(ctx, "C4HF7O2") is ctx.space_shadow                  # ... does not decide
        assert SP.space_of(ctx, "H2SO4") is ctx.space and SP.space_of(ctx, "C6H5NO3") is ctx.space
        for led in pf.values():
            for pid in ("k1", "j1"):
                led.loc[led["peak_id"] == pid, "alternatives"] = (
                    '[{"formula": "C7H3F5S", "adduct": "[M-H]-", "ppm": 0.1}]')
                led.loc[led["peak_id"] == pid, "tied"] = True
        for key in (pfca[1:3], junk[1:3]):
            cands = _cands(ctx, pf, key)
            assert "C7H3F5S [M-H]-" in [c["name"] for c in cands[1:]], key
            assert cands[0]["uncounted"] == []


SI2 = ("C10H27NO3Si2", "[M+NH4]+")          # an Si2 reading: Si2H4 == CO3 (0.41 mDa) of a CHO urea adduct


def _q1_si2(monkeypatch, matched):
    """The Si2 reading on a run whose fluorine has no evidence, a tied F reading beside it; its own isotope lines
    confirm ``matched``."""
    from peaky.assignment.levels import lines as LN
    _s, pf, ctx = _run([JUNK_F, ("s1",) + SI2 + ("Assigned", "cheminfo+grid")])
    for led in pf.values():
        led.loc[led["peak_id"] == "s1", "alternatives"] = '[{"formula": "C17H24F2O", "adduct": "[M+H]+", "ppm": 0.1}]'
        led.loc[led["peak_id"] == "s1", "tied"] = True
    real = LN.q1_isotopes

    def own_lines(ctx_, pair, cands, nmin):
        out = real(ctx_, pair, cands, nmin)
        cands[0]["matched_els"] = set(matched)
        return out
    monkeypatch.setattr(LN, "q1_isotopes", own_lines)
    facts = pd.DataFrame([dict(neutral_formula=SI2[0], adduct=SI2[1], mz=C.ion_mz(*SI2), ion="", ppm=0.0)])
    pairs = CX.pairs_from_files(pf, facts)
    return ctx, CP.q1_pass(ctx, pairs, 1)[SI2], pairs


class TestUnconfirmedHeteroatom:
    def test_the_heteroatoms_no_own_line_confirms(self):
        assert CP.unconfirmed_heteroatoms("C10H27NO3Si2", {"C"}) == {"Si"}
        assert CP.unconfirmed_heteroatoms("C10H27NO3Si2", {"C", "Si"}) == frozenset()
        assert CP.unconfirmed_heteroatoms("C12H31N4O4P", {"C", "N"}) == {"P"}          # P has no minor line
        assert CP.unconfirmed_heteroatoms("C10H17NO7S", {"S"}) == frozenset()
        assert CP.unconfirmed_heteroatoms("C16H22O", set()) == frozenset()              # CHO(N): nothing to confirm
        assert SP.heteroatoms("C6H4BrClO") == {"Br", "Cl"} and SP.heteroatoms("x?") == frozenset()

    def test_an_unconfirmed_si2_reading_takes_no_competitor_away(self, monkeypatch):
        """Without an own Si line the Si2 reading is no better evidenced than the F reading beside it: the F
        reading counts (as before D4), so D4 alone cannot lift the Si2 reading out of 5a."""
        import copy
        ctx, q, pairs = _q1_si2(monkeypatch, {"C"})
        assert ctx.space.space_reason(SI2[0]) is None and ctx.families_unevidenced == ("fluorinated",)
        names = {c["name"]: c["kind"] for c in q["cands"][1:]}
        assert "tie" in names.get("C17H24F2O [M+H]+", "")       # the shadow proposes it by mass, the engine tied it
        assert q["cands"][0]["uncounted"] == []
        before = copy.copy(ctx)
        before.space, before.space_shadow = ctx.space_shadow, None        # the scale's space before D4
        want = CP.q1_pass(before, pairs, 1)[SI2]
        key = (lambda c: (c["name"], c["kind"], c["win"], c["contradicted"]))
        assert sorted(map(key, q["cands"][1:])) == sorted(map(key, want["cands"][1:]))
        assert q["n_alt"] == want["n_alt"]

    def test_a_confirmed_si2_reading_keeps_the_f_reading_uncounted(self, monkeypatch):
        _ctx, q, _pairs = _q1_si2(monkeypatch, {"C", "Si"})
        assert "C17H24F2O [M+H]+" not in [c["name"] for c in q["cands"][1:]]
        assert "C17H24F2O [M+H]+" in [u["name"] for u in q["cands"][0]["uncounted"]]


# --------------------------------------------------------------------------- 3. the record
def _one(row, comps):
    """test_levels_decide's one-pair source; pass B counts no uncounted row as a competitor."""
    S = TD._prepared([row], comps=comps)
    S.P["n_competitors"] = [sum(1 for c in comps if (c[0], c[1]) == (n, a) and c[4] != CP.UNCOUNTED)
                            for n, a in zip(S.P["neutral_formula"], S.P["adduct"])]
    return TD._level(S).iloc[0]


def test_uncounted_competitors_are_listed_and_never_level_5a():
    comps = [("C10H16O4", "[M-H]-", "C9H13FO3S", "[M-H]-", CP.UNCOUNTED)]
    r = _one(TD.ACID, comps)
    assert r.evidence_level == "4a"
    assert r.competitors_left == f"not counted ({DC.UNCOUNTED_TEXT}): C9H13FO3S [M-H]-"
    assert "ion: unique in the calibrated window; 1 not counted (element family not open on evidence)" in r.evidence
    comps.append(("C10H16O4", "[M-H]-", "C9H12N2O4", "[M-H]-", "left"))
    r = _one(TD.ACID, comps)
    assert r.evidence_level == "5a"
    assert r.competitors_left == f"C9H12N2O4 [M-H]- || not counted ({DC.UNCOUNTED_TEXT}): C9H13FO3S [M-H]-"
    assert r.would_lift == "competitors left: C9H12N2O4 [M-H]-"


def test_the_uncounted_list_is_full():
    """Every uncounted ion is named (the competitor table is not written out: the column is the record)."""
    comps = [("C10H16O4", "[M-H]-", f"C{k}H13FO3S", "[M-H]-", CP.UNCOUNTED) for k in range(5, 14)]
    r = _one(TD.ACID, comps)
    assert r.competitors_left == f"not counted ({DC.UNCOUNTED_TEXT}): " + "; ".join(c[2] + " [M-H]-" for c in comps)
    assert "more)" not in r.competitors_left


def test_an_uncounted_reading_still_vouches_for_its_lines_in_test_b(monkeypatch):
    """The isotope tests see the uncounted readings beside the counted ones (an in-band line only an uncounted
    reading explains still refutes the committed reading, as when it was a competitor); the result lists them by
    name only, never among the competitors."""
    from peaky.assignment.levels import lines as LN
    _s, pf, ctx = _run([JUNK_F])
    seen = []
    real = LN.q1_isotopes

    def spy(ctx_, pair, cands, nmin):
        seen.append([c["name"] for c in cands])
        return real(ctx_, pair, cands, nmin)
    monkeypatch.setattr(LN, "q1_isotopes", spy)
    facts = pd.DataFrame([dict(neutral_formula=CHO_UREA[0], adduct=CHO_UREA[1], mz=C.ion_mz(*CHO_UREA), ion="",
                               ppm=0.0)])
    q1 = CP.q1_pass(ctx, CX.pairs_from_files(pf, facts), 1)
    assert seen == [[" ".join(CHO_UREA), FPSI]]
    com = q1[CHO_UREA]["cands"][0]
    assert len(q1[CHO_UREA]["cands"]) == 1
    assert com["uncounted"] == [dict(name=FPSI, neutral="C15H32FPSi", adduct="[M+H]+",
                                     ppm=pytest.approx(com["uncounted"][0]["ppm"]))]
    assert "lines" not in com["uncounted"][0]


def test_the_executable_reference_restates_the_uncounted_list():
    """scripts/level_ledger.py states the D4 record on its own: same text as the engine, never a competitor."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import level_ledger as LL
    assert (LL.UNCOUNTED_STATUS, LL.UNCOUNTED_TEXT) == (CP.UNCOUNTED, DC.UNCOUNTED_TEXT)
    comps = [("C10H16O4", "[M-H]-", f"C{k}H13FO3S", "[M-H]-", CP.UNCOUNTED) for k in range(5, 14)]
    comps.append(("C10H16O4", "[M-H]-", "C9H12N2O4", "[M-H]-", "left"))
    S = TD._prepared([TD.ACID], comps=comps)
    S.P["n_competitors"] = [1]
    got = LL.decide(S).iloc[0]
    assert got.competitors_left == _one(TD.ACID, comps).competitors_left
    assert got.n_left == 1 and got.evidence_level == "5a"
