"""The PDF report's data-driven sentences say only what the run's data support.

Pure helpers on synthetic numbers (no run folder, no network)."""
import numpy as np

from peaky.batch import cluster as CL
from peaky.reporting import pdf_report as R


# --------------------------------------------------------------------------- the event sentence
def _hours(n):
    return np.linspace(0.0, 48.0, n)


def test_a_bump_below_the_transient_bar_is_no_transient_and_gives_the_band():
    tt = np.full(120, 100.0)
    tt[30] = 131.0                                    # 1.31x: under the 1.7x bar
    tt[60:70] = 70.0                                  # a dip the band must show
    s = R._event_sentence(_hours(120), tt)
    assert s.startswith("No transient event")
    assert "1.31x" in s and f"{CL.PEAK_RANGE:.1f}x transient bar" in s
    assert "transient event)" not in s and "decays" not in s
    assert "within 0.70-1.00x" in s


def test_a_peak_over_the_bar_that_decays_is_a_transient_event():
    tt = np.full(120, 100.0)
    tt[30:36] = [150.0, 250.0, 300.0, 240.0, 180.0, 130.0]
    s = R._event_sentence(_hours(120), tt)
    assert "a transient event" in s and "3.0x the late-run baseline" in s
    assert s.startswith(f"Total signal peaks at hour {_hours(120)[32]:.1f}")


def test_a_maximum_in_the_last_tenth_of_the_samples_is_too_close_to_the_end_to_call():
    tt = np.full(120, 100.0)
    tt[-3:] = [150.0, 260.0, 120.0]                   # over the bar and decaying, but at the very end
    s = R._event_sentence(_hours(120), tt)
    assert "a transient event." not in s and "decays" not in s
    assert "in the last 10% of the samples: too close to the end of the run to tell a transient event from a rise" in s
    assert "rise the run ends on" not in s            # it does not claim that no decay follows
    # the same late maximum under the bar is simply no transient
    tt[-3:] = [105.0, 108.0, 110.0]
    assert R._event_sentence(_hours(120), tt).startswith("No transient event")


def test_the_bar_is_the_reports_own_and_too_few_samples_say_nothing():
    tt = np.full(60, 100.0)
    tt[10] = 160.0                                    # 1.6x: over a 1.5x bar, under 1.7x
    assert "a transient event" in R._event_sentence(_hours(60), tt, bar=1.5)
    assert R._event_sentence(_hours(60), tt).startswith("No transient event")
    assert R._event_sentence(_hours(2), np.array([1.0, 2.0])) is None
    assert R._event_sentence(_hours(5), np.zeros(5)) is None


# --------------------------------------------------------------------------- the reference-list chance level
class _List:
    """The two attributes reflists.match_by_mass reads off a list."""
    id = "toy"

    def __init__(self, formulas):
        self._f = set(formulas)

    def pool(self, include_radicals=False):
        return set(self._f)


def test_the_chance_null_reruns_the_rescue_match_on_shifted_masses():
    from peaky.chem import chemistry as C
    toy = _List({"C10H16O4", "C8H12O5", "C12H20O6"})
    obs = [C.ion_mz(f, "[M-H]-") for f in ("C10H16O4", "C8H12O5")]
    null = R.reflist_chance(obs, [toy], ["[M-H]-"])
    assert set(null) == {-40.0, -25.0, -15.0, 15.0, 25.0, 40.0}
    assert all(v == 0 for v in null.values())          # a sparse list: nothing by chance
    assert R.reflist_chance(obs, [toy], ["[M-H]-"], shifts=(0.0,)) == {0.0: 2}
    s = R._reflist_chance_text(2, null)
    assert "±15-40 ppm give 0 matches within 1 ppm" in s and "exceed that chance level" in s


def test_the_chance_sentence_says_at_or_below_chance_when_the_null_matches_more():
    tof = {-40.0: 238, -25.0: 260, -15.0: 296, 15.0: 273, 25.0: 272, 40.0: 256}
    s = R._reflist_chance_text(110, tof)
    assert "give 238-296 matches" in s and "110 observed" in s
    assert "below that chance level" in s and "no evidence by itself" in s
    assert "within that chance level" in R._reflist_chance_text(250, tof)
    orbi = {-40.0: 50, -25.0: 44, -15.0: 103, 15.0: 88, 25.0: 49, 40.0: 74}
    s = R._reflist_chance_text(261, orbi)
    assert "exceed that chance level (4.2x its median)" in s
    assert R._reflist_chance_text(5, {}) == ""


# --------------------------------------------------------------------------- `peaky report` reads the run's record
def test_peaky_report_takes_the_batch_and_dataset_from_the_run_manifest(tmp_path, monkeypatch):
    import argparse
    import json

    from peaky import cli
    from peaky import pipeline as PL

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(
        {"input": {"batch_name": "Batch A", "dataset": "Workspace X", "reagent": "Br"}}))
    seen = {}

    def fake(ctx, ts, *, subject=None, **kw):
        seen.update(batch=ctx.batch_name, dataset=ctx.dataset, ts=ts)
        return {}

    monkeypatch.setattr(PL, "generate_report", fake)
    args = argparse.Namespace(reagent="Br", run_dir=str(run), batch=None, dataset=None, tag=None,
                              run_id=None, generated=None, ts=str(tmp_path / "ts.parquet"), subject=None)
    cli.cmd_report(args)
    assert seen["batch"] == "Batch A" and seen["dataset"] == "Workspace X"
    cli.cmd_report(argparse.Namespace(**{**vars(args), "batch": "B", "dataset": "Y"}))
    assert seen["batch"] == "B" and seen["dataset"] == "Y"           # the flags win
    (run / "run_manifest.json").unlink()
    cli.cmd_report(args)                                              # no manifest: as before
    assert seen["dataset"] is None and seen["batch"] == "Br- CIMS"
    assert cli.report_inputs_of(str(tmp_path / "nowhere")) == {}


# --------------------------------------------------------------------------- composition: carbon-free ions apart, Assigned only
def _merged_rows(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["neutral_formula", "adduct", "tier", "ion_only_of"])


def test_carbon_free_neutrals_are_their_own_class_and_the_backbone_is_unchanged():
    from peaky.batch import composition as CMP
    assert CMP.composition_class("HNO3") == CMP.INORGANIC and CMP.backbone("HNO3") == "CHON"
    assert CMP.composition_class("H2O2") == CMP.INORGANIC and CMP.backbone("H2O2") == "CHO"
    assert CMP.composition_class("CH2O2") == "CHO" and CMP.composition_class("C10H17NO7") == "CHON"
    assert not CMP.is_inorganic("") and not CMP.is_inorganic("nan")


def test_one_carbon_held_as_inorganic_carbon_is_inorganic_and_organic_one_carbon_species_are_not():
    from peaky.batch import composition as CMP
    # the pseudo-halides an iodide inlet reads (iodine cyanide, iodine isocyanate), the cyanide /
    # cyanate family, a nitryl cyanide, the carbon oxides and sulfides; a 15N label counts as N
    for f in ("CNI", "CINO", "C^NI", "CHNO", "CHN", "CClN", "CHNS", "CN2O2", "CN2O3", "CN2O4", "CN2",
              "CO", "CO2", "CO3", "COS", "CS2"):
        assert CMP.is_inorganic_carbon(f) and CMP.composition_class(f) == CMP.INORGANIC, f
    # formic acid, carbonic / performic acid, fluoroform, urea, nitromethane, methylamine,
    # chloropicrin, a perfluorinated one-carbon fit, cyanogen (two carbons) and the one-carbon
    # polynitro organics (nitroform, tetranitromethane, chlorodinitromethane, a bromo-dinitro
    # fit, a three-N fit) stay organic
    for f, kl in (("CH2O2", "CHO"), ("CH2O3", "CHO"), ("CHF3", "CHO"), ("CH4N2O", "CHON"), ("CH3NO2", "CHON"),
                  ("CH5N", "CHON"), ("CCl3NO2", "CHON"), ("CF5NO4", "CHON"), ("C2N2", "CHON"),
                  ("CHN3O6", "CHON"), ("CN4O8", "CHON"), ("CHClN2O4", "CHON"), ("CHBrN2O2", "CHON"),
                  ("CHN3O2", "CHON"), ("C10H16O4", "CHO")):
        assert not CMP.is_inorganic_carbon(f) and CMP.composition_class(f) == kl, f
    # a one-carbon species with an element outside C/H/N/O/S/halogens is never inorganic carbon
    for f in ("CHNP", "CH3NSi", "CNNa"):
        assert not CMP.is_inorganic_carbon(f), f
    assert not CMP.is_inorganic_carbon("HNO3")         # carbon-free: inorganic by the other rule


def test_the_assigned_composition_weights_assigned_readings_only_with_the_reagent_ion_apart():
    from peaky.batch import composition as CMP
    merged = _merged_rows([
        ("HNO3", "[M-H]-", "Assigned", None),               # the nitrate reagent ion, read as an analyte
        ("HNO3", "[M+NO3]-", "Assigned", None),             # its dimer
        ("C10H16O4", "[M+NO3]-", "Assigned", None),
        ("C10H17NO7", "[M+NO3]-", "Assigned", None),
        ("C20H30O12", "[M+NO3]-", "Candidate", None),        # a Candidate: left out
        ("C10H16O4", "[M]-.", "Candidate", "p1"),            # an ion-only row: never Assigned
    ])
    sig = {("HNO3", "[M-H]-"): 600.0, ("HNO3", "[M+NO3]-"): 300.0, ("C10H16O4", "[M+NO3]-"): 80.0,
           ("C10H17NO7", "[M+NO3]-"): 20.0, ("C20H30O12", "[M+NO3]-"): 500.0, ("C10H16O4", "[M]-."): 99.0}
    ac = CMP.assigned_composition(merged, sig)
    assert ac["n_readings"] == 4 and ac["total"] == 1000.0
    assert ac["inorganic_frac"] == 0.9
    assert ac["organic_frac"] == {"CHO": 0.8, "CHON": 0.2}             # of the organic 100 cps
    assert ac["count"] == {CMP.INORGANIC: 1, "CHO": 1, "CHON": 1}
    # the old all-tier backbone booked the reagent ion as CHON
    frac, _ = CMP.signal_by_backbone(merged, {"HNO3": 900.0, "C10H16O4": 179.0, "C10H17NO7": 20.0,
                                              "C20H30O12": 500.0})
    assert frac["CHON"] > 0.5


def test_the_findings_bullet_names_the_inorganic_share_and_gates_the_bright_cho_clause():
    from peaky.batch import composition as CMP
    lines = R._composition_lines({"assigned_comp": {
        "n_readings": 4, "total": 1000.0, "inorganic_frac": 0.9, "organic_frac": {"CHO": 0.8, "CHON": 0.2},
        "count": {"inorganic": 1, "CHO": 1, "CHON": 1}, "signal": {"inorganic": 900.0},
        "by_neutral": {"C10H16O4": ("CHO", 80.0), "C10H17NO7": ("CHON", 20.0)}}})
    txt = [t for _s, t in lines]
    assert "the Assigned organic readings are 80% CHO / 20% CHON" in txt[0]
    assert "50% CHON by count of the same neutrals" in txt[0]
    # the bright-CHO clause is scoped to the organic signal and sits before the inorganic line
    assert txt[1] == ("  The brightest CHO neutral carries 80% of the Assigned organic signal: a few "
                      "bright CHO species carry most of it.")
    assert txt[2].startswith("  Reagent and inorganic ions (carbon-free, or one carbon only as a carbon oxide / sulfide "
                             "or a cyanide / cyanate) carry 90% of the Assigned M0 signal")
    assert not any("ammonium/amine" in t for t in txt)    # no NH4+ / urea channel
    # five of eight equal CHO neutrals carry exactly half of the organic signal: the clause, at the bar
    spread = {f"C{k}H{2 * k}O5": ("CHO", 10.0) for k in range(5, 13)}           # 8 x 10 cps CHO
    spread["C10H17NO7"] = ("CHON", 20.0)
    ac = {"n_readings": 9, "total": 100.0, "inorganic_frac": 0.0, "organic_frac": {"CHO": 0.8, "CHON": 0.2},
          "count": {"CHO": 8, "CHON": 1}, "signal": {}, "by_neutral": spread}
    assert CMP.top_share(ac, "CHO", R.BRIGHT_CHO_N) == R.BRIGHT_CHO_SHARE == 0.5
    assert "The 5 brightest CHO neutrals carry 50% of the Assigned organic signal" in " ".join(
        t for _s, t in R._composition_lines({"assigned_comp": ac}))
    # CHO 82% of the organic signal, but spread over nine neutrals (top five 45%): no clause
    ac = dict(ac, organic_frac={"CHO": 90 / 110, "CHON": 20 / 110},
              by_neutral=dict(spread, C13H26O5=("CHO", 10.0)))
    assert CMP.top_share(ac, "CHO", 5) == 50 / 110
    txt = " ".join(t for _s, t in R._composition_lines({"assigned_comp": ac}))
    assert "82% CHO" in txt and "a few bright CHO species" not in txt and "Reagent and inorganic" not in txt
    # the ammonium/amine note needs an NH4+ or a urea channel, not just a positive polarity
    no_nh4 = {"assigned_comp": ac, "positive": True, "adduct_counts": {"[M+H]+": 3, "[M]+.": 2}}
    assert "ammonium/amine" not in " ".join(t for _s, t in R._composition_lines(no_nh4))
    nh4 = dict(no_nh4, adduct_counts={"[M+H]+": 3, "[M+NH4]+": 2})
    assert "ammonium/amine re-reads" in " ".join(t for _s, t in R._composition_lines(nh4))
    assert R._composition_lines({"assigned_comp": {"n_readings": 0}})[0][1].startswith("• No reading is held")


def test_top_species_split_organic_from_carbon_free():
    from peaky.batch import composition as CMP
    merged = _merged_rows([("HNO3", "[M-H]-", "Assigned", None), ("C10H16O4", "[M+NO3]-", "Assigned", None),
                           ("H2O2", "[M+NO3]-", "Assigned", None)])
    sig = {"HNO3": 900.0, "C10H16O4": 80.0, "H2O2": 20.0}
    org = CMP.top_species_by_signal(merged, sig, inorganic=False)
    ino = CMP.top_species_by_signal(merged, sig, inorganic=True)
    assert [r["neutral_formula"] for r in org] == ["C10H16O4"] and org[0]["frac"] == 0.08
    assert [r["neutral_formula"] for r in ino] == ["HNO3", "H2O2"] and ino[0]["klass"] == CMP.INORGANIC
    assert len(CMP.top_species_by_signal(merged, sig)) == 3


def test_a_share_never_rounds_a_nonzero_class_to_zero_or_a_partial_one_to_all():
    assert [R._share(x) for x in (0.0, 0.003, 0.2, 0.996, 1.0)] == ["0%", "<1%", "20%", ">99%", "100%"]
    assert [R._share1(x) for x in (0.0, 0.0002, 0.413)] == ["0.0%", "<0.1%", "41.3%"]


# --------------------------------------------------------------------------- the oligomer line: Assigned members only
def test_the_oligomer_list_keeps_assigned_neutrals_only():
    from peaky.batch import composition as CMP
    merged = _merged_rows([
        ("C20H30O14", "[M+NO3]-", "Assigned", None),
        ("C19H30O12", "[M+NO3]-", "Candidate", None),          # Candidate only: not listed
        ("C20H32O10", "[M+NO3]-", "Candidate", None),
        ("C20H32O10", "[M-H]-", "Assigned", None),             # Assigned on another channel: listed
        ("C21H34O12", "[M]-.", "Candidate", "p9"),             # ion-only: never
        ("C10H16O4", "[M+NO3]-", "Assigned", None),            # not high-C
    ])
    assert CMP.oligomer_flag(merged) == ["C20H30O14", "C20H32O10"]
    assert CMP.oligomer_flag(merged, tiers=None) == ["C21H34O12", "C20H30O14", "C20H32O10", "C19H30O12"]
    assert CMP.oligomer_flag(merged.drop(columns=["tier"])) == CMP.oligomer_flag(merged, tiers=None)


# --------------------------------------------------------------------------- the cover names the code
def test_the_cover_names_the_package_version_and_a_sha_only_when_there_is_one(tmp_path, monkeypatch):
    import json

    import peaky
    from peaky.reporting import provenance as PV
    monkeypatch.setattr(peaky, "__version__", "1.2.3", raising=False)
    monkeypatch.setattr(PV, "git_info", lambda path: {})              # a pip install: no checkout
    assert R._skill_version(str(tmp_path)) == "peaky 1.2.3"            # no 'git ?', no module version
    monkeypatch.setattr(PV, "git_info", lambda path: {"commit": "abcdef0123456789", "dirty": False})
    assert R._skill_version(None) == "peaky 1.2.3 · git abcdef0"
    # the run's own manifest names the code that assigned it; other report code is named beside it
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"code": {"package_version": "1.2.0", "git": {"commit": "0123456789abcdef", "dirty": True}}}))
    assert R._skill_version(str(tmp_path)) == ("assigned by peaky 1.2.0 · git 0123456+modified · "
                                               "report by peaky 1.2.3 · git abcdef0")
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"code": {"package_version": "1.2.3", "git": {"commit": "abcdef0123456789", "dirty": False}}}))
    assert R._skill_version(str(tmp_path)) == "peaky 1.2.3 · git abcdef0"
    (tmp_path / "run_manifest.json").write_text("{not json")
    assert R._skill_version(str(tmp_path)) == "peaky 1.2.3 · git abcdef0"


# --------------------------------------------------------------------------- the Methods text reads the run's record
def test_the_amine_r_minimum_is_the_runs_or_the_code_default_never_a_stale_constant():
    import inspect

    from peaky.batch import assign_batch as AB
    from peaky import pipeline as PL
    default = inspect.signature(AB.run).parameters["amine_r_min"].default
    # the batch path's own default and the pipeline's agree with what the page falls back to
    assert inspect.signature(PL.run_batch).parameters["amine_r_min"].default == default
    assert R._amine_r_min({"batch": {"amine_r_min": 0.8}}) == (0.8, True)
    assert R._amine_r_min({"batch": {"evidence_levels": {"amine_r_min": 0.55}}}) == (0.55, True)
    assert R._amine_r_min({"batch": {}}) == (float(default), False)
    assert R._amine_r_text({"batch": {"amine_r_min": 0.6}}) == "r>=0.6"
    assert "the code default" in R._amine_r_text({})
    assert "0.7" not in R._amine_r_text({"batch": {"amine_r_min": 0.6}})


def test_the_methods_name_the_scorer_the_run_recorded():
    assert "in-process (the local scorer)" in R._scorer_text({"batch": {"scorer": "local"}})
    assert "by the server (match_compounds)" in R._scorer_text({"batch": {"scorer": "server"}})
    assert "did not record which" in R._scorer_text({"batch": {}})


def test_recorded_numbers_print_formatted():
    assert R._num(9.22777806538331, ".2f") == "9.23" and R._num(6, ".2f") == "6.00"
    assert R._num(None, ".2f") == "?" and R._num("x", ".3g", "n/a") == "n/a"


# --------------------------------------------------------------------------- the 'neutral' claim is scoped to the declared channels
def test_the_neutral_claim_says_it_holds_among_the_declared_channels_everywhere_it_is_defined():
    import pandas as pd

    from peaky.assignment import evidence as EV
    from peaky.reporting import report as XL
    meaning = EV.CLAIM_MEANING["neutral"]
    assert "among the run's declared reagent channels" in meaning
    assert "side channels locked" in meaning and "[M-H]-" in meaning
    # the workbook's By claim sheet and Read me carry the same definition
    m0 = pd.DataFrame({"evidence_level": ["4a"], "tier": ["Assigned"], "height": [1.0], "mz": [100.0],
                       "neutral_formula": ["C5H8O4"], "adduct": ["[M-H]-"], "claim": ["neutral"]})
    bc = XL.claim_sheet(m0)
    assert bc[(bc.section == "summary") & (bc.claim == "neutral")]["meaning"].iloc[0] == meaning
    rm = XL.legend_sheet(claims=True)
    assert (rm.explanation == meaning).any()


# --------------------------------------------------------------------------- the sparse-calibration notice
def test_levels_not_assessed_only_when_no_file_calibrated():
    from peaky.assignment import evidence as EV
    why = EV.levels_not_assessed_reason([EV.NO_RUN_WINDOW_TEXT] * 3, ["", None, ""])
    assert why.startswith("Evidence levels were not assessed") and "at least 20" in why
    assert EV.levels_not_assessed_reason([EV.NO_WINDOW_TEXT], [None]) == why    # a file levelled alone
    assert EV.levels_not_assessed_reason([EV.NO_RUN_WINDOW_TEXT, "x"], ["", "4a"]) is None
    assert EV.levels_not_assessed_reason(["not assessed on this instrument class"], ["NA"]) is None
    assert EV.levels_not_assessed_reason([], []) is None and EV.levels_not_assessed_reason(None) is None


def test_the_workbook_summary_says_once_that_levels_were_not_assessed():
    import numpy as np
    import pandas as pd

    from peaky.assignment import evidence as EV
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.reporting import report as XL
    led = L.new_ledger(pd.DataFrame({"peak_id": ["A", "B"], "mz": [215.0925, 231.1238],
                                     "height": [1e5, 5e4]}))
    for pid, f in (("A", "C10H16O5"), ("B", "C11H20O5")):
        L.commit_assignment(led, pid, neutral_formula=f, adduct="[M-H]-", ion_formula=f, ion_score=0.9,
                            compound_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo",
                            confidence="High", commentary="stub")
    T.apply_tiers(led)
    m0 = led["role"] == L.ROLE_M0
    led["evidence_level"] = pd.Series(np.nan, index=led.index, dtype=object)
    led["evidence"] = pd.Series(np.nan, index=led.index, dtype=object)
    led.loc[m0, "evidence"] = EV.NO_WINDOW_TEXT
    ss = XL.summary_stats(led, context="ambient-air")
    na = ss[(ss.section == "Evidence levels") & (ss.metric == "not assessed")]
    assert len(na) == 1 and na.value.iloc[0].startswith("Evidence levels were not assessed")
    led.loc[m0, "evidence_level"] = "4b"
    led.loc[m0, "evidence"] = "ion established"
    ss = XL.summary_stats(led, context="ambient-air")
    assert not ((ss.section == "Evidence levels") & (ss.metric == "not assessed")).any()
