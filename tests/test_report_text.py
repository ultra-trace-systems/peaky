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


def test_a_maximum_in_the_last_tenth_of_the_samples_is_a_rise_the_run_ends_on():
    tt = np.full(120, 100.0)
    tt[-3:] = [150.0, 200.0, 260.0]                   # over the bar, but at the very end
    s = R._event_sentence(_hours(120), tt)
    assert "transient event" not in s and "rise the run ends on" in s
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
    assert "±15-40 ppm give 0-0 matches within 1 ppm" in s and "exceed that chance level" in s


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
    lines = R._composition_lines({"assigned_comp": {
        "n_readings": 4, "total": 1000.0, "inorganic_frac": 0.9, "organic_frac": {"CHO": 0.8, "CHON": 0.2},
        "count": {"inorganic": 1, "CHO": 1, "CHON": 1}, "signal": {"inorganic": 900.0}}})
    txt = " ".join(t for _s, t in lines)
    assert "the Assigned organic readings are 80% CHO / 20% CHON" in txt
    assert "50% CHON by count of the same neutrals" in txt
    assert "Carbon-free reagent and inorganic ions carry 90% of the Assigned M0 signal" in txt
    assert "a few bright CHO species" in txt
    lines = R._composition_lines({"assigned_comp": {
        "n_readings": 3, "total": 100.0, "inorganic_frac": 0.0, "organic_frac": {"CHO": 0.4, "CHON": 0.6},
        "count": {"CHO": 2, "CHON": 1}, "signal": {}}})
    txt = " ".join(t for _s, t in lines)
    assert "a few bright CHO species" not in txt and "Carbon-free" not in txt
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
