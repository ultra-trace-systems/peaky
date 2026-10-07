"""scripts/level_ledger.py — one synthetic run dir, one row per level.

The three real golden vectors (a coastal TOF campaign, and the two-instrument same-air pair)
live outside this repo with the data they were measured on. What is pinned here
is the decision table itself: a fixture built so that every level fires exactly
once, the axes that feed it, and the order the predicates are tried in — each
checked by a mutant that must change the level.

Run: pytest tests/test_level_ledger.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import level_ledger as LL  # noqa: E402

LEDGER_COLUMNS = [
    "role",
    "peak_id",
    "parent_peak_id",
    "iso_label",
    "neutral_formula",
    "adduct",
    "ion_formula",
    "mz",
    "height",
    "tier",
    "method",
    "confidence",
    "tied",
    "below_assignability",
    "degeneracy_density",
    "degeneracy_note",
    "resolvability",
    "series_unit",
    "anchor_peak_id",
    "isotopologues",
    "ppm_error_cal",
    "occurrence_y",
]


def m0(
    peak_id,
    neutral,
    adduct="[M-H]-",
    ion=None,
    mz=200.0,
    height=1000.0,
    tier="Assigned",
    method="pass2",
    confidence="High",
    tied=False,
    below=False,
    degeneracy=0.5,
    note="",
    resolvability="resolved",
    series_unit=None,
    anchor=None,
):
    """One committed neutral. Defaults are deliberately uncorroborated."""
    return {
        "role": "M0",
        "peak_id": peak_id,
        "parent_peak_id": None,
        "iso_label": None,
        "neutral_formula": neutral,
        "adduct": adduct,
        "ion_formula": ion or neutral,
        "mz": mz,
        "height": height,
        "tier": tier,
        "method": method,
        "confidence": confidence,
        "tied": tied,
        "below_assignability": below,
        "degeneracy_density": degeneracy,
        "degeneracy_note": note,
        "resolvability": resolvability,
        "series_unit": series_unit,
        "anchor_peak_id": anchor,
        "isotopologues": "",
        "ppm_error_cal": 0.4,
        "occurrence_y": 0.9,
    }


def child(peak_id, parent, label, height):
    """One isotope satellite hanging off an M0 row."""
    row = m0(peak_id, None, adduct=None, height=height)
    row.update(
        role="iso_child",
        parent_peak_id=parent,
        iso_label=label,
        tier="Assigned",
        degeneracy_density=None,
    )
    return row


# One nitrate-channel source. Every neutral is distinct so the per-neutral facts
# (second channel, acid branch) cannot leak between the cases.
NITRATE_ROWS = [
    # 5b -- the arbiter broke a near-tie, whatever else the row has
    m0("p_tie", "C6H8O4", tied=True, anchor="p_anchor"),
    # 5b -- mass-degenerate with no axis at all
    m0("p_degen", "C6H10O4", degeneracy=5.0),
    # 2b -- a curated identity on a formula with essentially one structure
    m0("p_known", "HNO3", method="known:atmospheric"),
    # 3a -- a curated class, isomers open
    m0("p_class", "C8HF15O2", method="known:perfluoroacid"),
    # 3b -- the same neutral deprotonated AND clustered: a substituent only
    m0("p_acid1", "C10H16O5", adduct="[M-H]-"),
    m0("p_acid2", "C10H16O5", adduct="[M+NO3]-", mz=262.0),
    # 4c -- unopposed on a separable peak, but nothing corroborates it
    m0("p_uniq", "C7H12O3", degeneracy=0.5, resolvability="resolved"),
    # 5a -- the same row on a peak the width model calls blended
    m0("p_blend", "C7H12O4", degeneracy=0.5, resolvability="blended"),
    # 4b -- exactly one axis (a homologous-series tie), nothing outside it
    m0("p_anchor", "C9H14O4", series_unit="CH2"),
    # 4a -- two axes, one of them the other source
    m0("p_cross", "C10H16O4", ion="C10H15O4", height=1000.0),
    child("c_cross", "p_cross", "13C+1", 107.0),
]

# One bromide-channel source: its clusters carry Br, so an 81Br satellite pins
# the ion and not the neutral.
BROMIDE_ROWS = [
    m0("b_reag", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br", mz=221.0),
    child("b_reag_iso", "b_reag", "81Br+1", 950.0),
    m0("b_bulk", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br", mz=235.0),
]

# The corroborating source: it must pin the neutral by an axis of its OWN (here a
# 13C line at the ratio of ten carbons) -- a source corroborates only what it holds
# at 4b or better on its own evidence.
OTHER_ROWS = [
    m0("o_cross", "C10H16O4", adduct="[M+NO3]-", ion="C10H16NO7", mz=262.0),
    child("o_cross_iso", "o_cross", "13C+1", 107.0),
]


def write_ledger(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LEDGER_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def sources(tmp_path):
    """A run dir with a per_file ledger, plus two bare ledger CSVs."""
    run_dir = tmp_path / "run" / "NITRATE_2026"
    write_ledger(run_dir / "per_file" / "s1_ledger.csv", NITRATE_ROWS)
    bromide = write_ledger(tmp_path / "bromide_ledger.csv", BROMIDE_ROWS)
    other = write_ledger(tmp_path / "other_ledger.csv", OTHER_ROWS)
    return run_dir, bromide, other


def levels(df):
    return dict(zip(zip(df.neutral, df.adduct), df.level))


def test_every_level_fires_once(sources):
    run_dir, _, other = sources
    df = LL.run([str(run_dir)], [str(other)])
    got = levels(df)
    assert got[("C6H8O4", "[M-H]-")] == "5b"
    assert got[("C6H10O4", "[M-H]-")] == "5b"
    assert got[("HNO3", "[M-H]-")] == "2b"
    assert got[("C8HF15O2", "[M-H]-")] == "3a"
    assert got[("C10H16O5", "[M-H]-")] == "3b"
    assert got[("C10H16O5", "[M+NO3]-")] == "3b"
    assert got[("C7H12O3", "[M-H]-")] == "4c"
    assert got[("C7H12O4", "[M-H]-")] == "5a"
    assert got[("C9H14O4", "[M-H]-")] == "4b"
    assert got[("C10H16O4", "[M-H]-")] == "4a"


def test_reagent_only_isotope_is_4d(sources):
    """The one level with no Schymanski analogue: the ion, not the neutral."""
    _, bromide, _ = sources
    df = LL.run([str(bromide)], [])
    got = levels(df)
    assert got[("C8H14O2", "[M+Br]-")] == "4d"
    row = df[df.neutral == "C8H14O2"].iloc[0]
    assert row.reagent_only_iso and row.iso and not row.neutral_backed
    # the row with no satellite at all has no axis and a unique, resolved peak
    assert got[("C9H16O2", "[M+Br]-")] == "4c"


def test_reagent_halogen_is_read_from_the_cluster_adduct(sources):
    """A stray [M+Br]- on a nitrate channel must not make it a bromide one."""
    run_dir, bromide, _ = sources
    assert LL.run([str(bromide)], []).reagent_halogen.iloc[0] == "Br"
    assert LL.run([str(run_dir)], []).reagent_halogen.iloc[0] == ""


def test_a_13c_satellite_at_the_wrong_ratio_is_not_evidence(tmp_path):
    rows = [
        m0("p", "C10H16O4", ion="C10H15O4"),
        child("c", "p", "13C+1", 107.0),  # 10 carbons -> 0.107 of the parent
    ]
    good = write_ledger(tmp_path / "good_ledger.csv", rows)
    assert LL.run([str(good)], []).iso.iloc[0]
    rows[1]["height"] = 900.0  # eight times what ten carbons can give
    bad = write_ledger(tmp_path / "bad_ledger.csv", rows)
    assert not LL.run([str(bad)], []).iso.iloc[0]


def test_corroboration_is_symmetric_and_external_sources_are_not_levelled(sources):
    """Naming two sources levels both and gives each the other as an axis."""
    run_dir, _, other = sources
    both = LL.run([str(run_dir), str(other)], [])
    assert set(both.source) == {"NITRATE_2026", "other_ledger"}
    assert levels(both)[("C10H16O4", "[M-H]-")] == "4a"
    assert both[both.source == "other_ledger"].corroborated.all()
    external = LL.run([str(run_dir)], [str(other)])
    assert set(external.source) == {"NITRATE_2026"}
    # the axis is the same either way
    assert levels(external)[("C10H16O4", "[M-H]-")] == "4a"
    alone = LL.run([str(run_dir)], [])
    assert levels(alone)[("C10H16O4", "[M-H]-")] == "4b"


def test_a_source_that_only_carries_the_neutral_does_not_corroborate(tmp_path, sources):
    """A 5a sighting (exact mass, no axis) is the other grid enumerating the same
    formula, not a second sighting -- while the direction that IS a sighting
    still counts: the run pins C10H16O4 by its own 13C line (4b), so it
    corroborates the bare source's row."""
    run_dir, _, _ = sources
    bare = write_ledger(tmp_path / "bare_ledger.csv", [m0("o", "C10H16O4", adduct="[M+NO3]-", mz=262.0, degeneracy=None)])
    assert levels(LL.run([str(run_dir)], [str(bare)]))[("C10H16O4", "[M-H]-")] == "4b"
    both = LL.run([str(run_dir), str(bare)], [])
    run_row = both[(both.source == "NITRATE_2026") & (both.neutral == "C10H16O4")].iloc[0]
    bare_row = both[both.source == "bare_ledger"].iloc[0]
    assert not run_row.corroborated and run_row.level == "4b"
    assert bare_row.corroborated and bare_row.level == "4b"


def test_two_sources_that_only_agree_cannot_lift_each_other(tmp_path):
    x = write_ledger(tmp_path / "x_ledger.csv", [m0("p", "C6H10O4", degeneracy=5.0)])
    y = write_ledger(tmp_path / "y_ledger.csv", [m0("q", "C6H10O4", adduct="[M+NO3]-", mz=208.0, degeneracy=5.0)])
    both = LL.run([str(x), str(y)], [])
    assert set(both.level) == {"5b"} and not both.corroborated.any()


def test_a_merged_ledger_corroborates_by_its_stored_level_without_its_own_axis(tmp_path, sources):
    run_dir, _, _ = sources
    merged_dir = tmp_path / "OTHER_RUN"
    merged_dir.mkdir()
    pd.DataFrame([
        dict(neutral_formula="C10H16O4", adduct="[M+NO3]-", evidence_level="4b", evidence_axes="corroborated|files:2"),
        dict(neutral_formula="C9H14O4", adduct="[M+NO3]-", evidence_level="4a", evidence_axes="iso|corroborated|files:2"),
    ]).to_csv(merged_dir / "merged_ledger.csv", index=False)
    got = levels(LL.run([str(run_dir)], [str(merged_dir)]))
    assert got[("C10H16O4", "[M-H]-")] == "4b"     # the stored 4b was the axis alone: no sighting
    assert got[("C9H14O4", "[M-H]-")] == "4a"      # the stored 4a keeps its own 13C: series tie + corroboration


def test_resolvability_only_binds_where_it_was_measured(tmp_path):
    """The cover path never measures it; a row is not punished for its absence."""
    rows = [m0("p", "C7H12O3", degeneracy=0.5, resolvability="blended")]
    measured = write_ledger(tmp_path / "measured_ledger.csv", rows)
    assert LL.run([str(measured)], []).level.iloc[0] == "5a"
    rows[0]["resolvability"] = None
    absent = write_ledger(tmp_path / "absent_ledger.csv", rows)
    assert LL.run([str(absent)], []).level.iloc[0] == "4c"


def test_hard_evidence_problems_outrank_everything(sources):
    """5b is tried first: a curated identity does not rescue a near-tie."""
    run_dir, _, other = sources
    rows = [dict(r) for r in NITRATE_ROWS]
    for row in rows:
        if row["peak_id"] == "p_known":
            row["tied"] = True
    write_ledger(run_dir / "per_file" / "s1_ledger.csv", rows)
    assert levels(LL.run([str(run_dir)], [str(other)]))[("HNO3", "[M-H]-")] == "5b"


def test_low_confidence_and_below_assignability_also_reach_5b(tmp_path):
    for column, value in (("confidence", "Low (0.31)"), ("below_assignability", True)):
        rows = [m0("p", "HNO3", method="known:atmospheric")]
        rows[0][column] = value
        path = write_ledger(tmp_path / f"{column}_ledger.csv", rows)
        assert LL.run([str(path)], []).level.iloc[0] == "5b"


def test_missing_columns_and_nulls_do_not_raise(tmp_path):
    """A ledger from an older run is levelled, not crashed on."""
    thin = pd.DataFrame(
        {
            "role": ["M0", "M0"],
            "peak_id": ["p1", "p2"],
            "neutral_formula": ["C5H8O2", None],
            "adduct": ["[M-H]-", None],
            "mz": [100.0, 200.0],
            "height": [500.0, None],
            "tier": ["Assigned", "Candidate"],
        }
    )
    path = tmp_path / "thin_ledger.csv"
    thin.to_csv(path, index=False)
    df = LL.run([str(path)], [])
    assert len(df) == 2
    assert set(df.level) <= set(LL.LEVEL_ORDER)
    assert df.res_ok.all()  # never measured, so never held against the row


def test_cli_writes_the_levelled_rows(tmp_path, sources, capsys):
    run_dir, _, other = sources
    out = tmp_path / "levels" / "levels.csv"
    assert LL.main([str(run_dir), "--corroborate", str(other), "--out", str(out)]) == 0
    written = pd.read_csv(out)
    assert len(written) == 10
    assert "level" in written.columns and "n_axes" in written.columns
    printed = capsys.readouterr().out
    assert "2b/3a/3b/4a/4b/4c/4d/5a/5b" in printed


def test_run_dir_is_found_through_its_out_dir(tmp_path):
    out_dir = tmp_path / "out"
    write_ledger(out_dir / "RUN_2026" / "per_file" / "s1_ledger.csv", NITRATE_ROWS)
    assert len(LL.run([str(out_dir)], [])) == 10


def test_merged_ledger_is_used_when_there_is_no_per_file_dir(tmp_path):
    run_dir = tmp_path / "RUN_2026"
    write_ledger(run_dir / "merged_ledger.csv", NITRATE_ROWS)
    assert len(LL.run([str(run_dir)], [])) == 10


# --------------------------------------------------------------------------- family scope (2026-09-22)
def _curated(neutral, family, adduct="[M-H]-"):
    """The level of one curated commit, on its own, uncorroborated."""
    led = pd.DataFrame([m0("p", neutral, adduct=adduct, method=f"known:{family}")],
                       columns=LEDGER_COLUMNS)
    led["__file"] = "f"
    return LL.assign_levels(LL.measure_source("f", led, None), set()).level.iloc[0]


def test_curated_scope_follows_the_spec_not_a_hand_made_set():
    """docs/EVIDENCE_LEVELS.md section 4.1: 2b = a COMPOUND-scope family AND a
    one-structure formula in the isomer space; every other curated commit is 3a.
    Until 2026-09-22 this script kept two hand-made sets that read cyclosiloxane
    as a class (3a, in core 2b) and knew no contaminant:silanediol."""
    assert _curated("C6H18O3Si3", "cyclosiloxane", "[M+H]+") == "2b"          # D3: one structure
    assert _curated("C2H8O2Si", "contaminant:silanediol", "[M+NO3]-") == "3a"  # class scope
    assert _curated("C6H5NO3", "nitroaromatic") == "3a"                        # compound scope, three isomers
    assert _curated("C2HF3O2", "perfluoroacid") == "3a"                        # class scope, even at one structure
    assert _curated("HNO3", "atmospheric") == "2b"
    assert _curated("C99H99O99", "atmospheric") == "3a"                        # compound scope, not in the space
    assert LL.KNOWN_FAMILY_SCOPE["contaminant:silanediol"] == "class"
    assert LL.plausible_structures("C6H18O3Si3") == 1 and LL.plausible_structures("C99H99O99") is None
