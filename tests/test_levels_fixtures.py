"""The evidence scale end to end on two real Orbitrap batches (the level fixtures).

tests/fixtures/levels/v1_nitrate/ (a labelled-nitrate batch, 12 files) and
v1_uronium/ (a uronium batch, 10 files) are batch run directories cut to what
the scale's pooled stage reads: every per-file ledger row (the isotope probes
need every peak, not only the committed ones) in the columns the stage reads,
the merged ledger's reading columns, the time-series rows the stage reads (the
M0 groups route co-variation joins, the m/z bins the amine gate looks up), the
batch checks' tables and the batch summary's width model, reagent, context,
reference lists and their activation record, per-file gates. Sample ids
(s01..) and peak ids (p00001..) are stand-ins in the originals' order.

When they were cut, every set levelled from the fixture equalled the level of
its full run directory (no partners) on every output and fact column, so the
fixtures carry every input the stage reads. `expected_levels_v1.csv.gz` is the
level table the core gave then, and the golden vectors are its counts. Both
are pinned until the user signs off the vectors. A change that moves them is a
change of the scale and must say so.

Regenerate from the fixtures in the repo (no run directories needed):
`build_fixtures.py --expected-only` (the cutter, kept outside the repository
because it reads the real run directories).

`scripts/level_ledger.py` re-levels the same run directory with its own
decision layer and must agree row for row. It is skipped, with the reason,
until the script has the scale's decision layer (its `--mode` option).
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment.levels import scale as SC

FIX = Path(__file__).resolve().parent / "fixtures" / "levels"
REPO = Path(__file__).resolve().parents[1]
SETS = ("v1_nitrate", "v1_uronium")
EXPECTED = FIX / "expected_levels_v1.csv.gz"
VECTOR_ORDER = "/".join([*SC.LEVELS, *SC.BUCKETS])
#: the golden vectors (pairs, 3c/4a/4b/5a/5b/reagent/NA) -- pending the user's sign-off
GOLDEN = {
    "v1_nitrate": (1850, "10/186/289/472/892/1/0"),
    "v1_uronium": (1161, "17/75/913/125/31/0/0"),
}


def materialize(name: str, dest: Path) -> Path:
    """The fixture set as a plain run directory (CSV, not gz) under dest."""
    src = FIX / name
    rd = dest / name
    for p in sorted(src.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = rd / (str(rel)[:-3] if p.suffix == ".gz" else str(rel))
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".gz":
            with gzip.open(p, "rb") as a, open(out, "wb") as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy(p, out)
    return rd


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("level_fixtures")
    return {s: materialize(s, base) for s in SETS}


@pytest.fixture(scope="module")
def levels(run_dirs):
    return {s: EV.level_source(EV.source_from_run_dir(str(rd))) for s, rd in run_dirs.items()}


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(EXPECTED, dtype=str, keep_default_na=False)


def _text(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Columns as the CSV prints them (NaN / NA / None -> '', bools 'True'/'False')."""
    buf = frame[cols].to_csv(index=False)
    return pd.read_csv(pd.io.common.StringIO(buf), dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------- the fixtures
def test_each_fixture_is_a_run_directory_the_stage_reads():
    for s in SETS:
        d = FIX / s
        summary = json.loads((d / "batch_summary.json").read_text())
        assert summary["resolution"]["r_at_200"] >= EV.ORBITRAP_R200        # Orbitrap-class: levelled
        assert {p["sample_id"] for p in summary["per_file"]} == \
            {p.name.split("_")[0] for p in (d / "per_file").glob("*_ledger.csv.gz")}
        assert summary["reflists_context"]["matched"]                       # how the lists were activated
        for p in (d / "per_file").glob("*_ledger.csv.gz"):
            led = pd.read_csv(p, low_memory=False)
            # every row, not only the committed ones: the probes look at every peak
            assert {"M0", "iso_child"} <= set(led["role"]) and len(set(led["role"])) > 2
            assert led["peak_id"].str.fullmatch(r"p\d{5}").all()
        assert (d / "per_file" / "_batch_ts.parquet").is_file() and (d / "merged_ledger.csv.gz").is_file()


def test_the_fixtures_carry_no_internal_identifier():
    """Stand-in ids, no run / site / instrument names (scripts/privacy_scan.py's rules
    over the decompressed text; a 16-letter molecular formula is no id)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import privacy_scan as PS
    import re
    formula = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
    for p in sorted(FIX.glob("v1_*/**/*")) + [EXPECTED]:
        if p.is_dir() or p.suffix == ".parquet":
            continue
        text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
        hits = [h for h in PS.scan_text(text, str(p)) if not (h.rule == "mascope-id" and formula.match(h.found))]
        assert not hits, [str(h) for h in hits[:5]]
    for s in SETS:
        ts = pd.read_parquet(FIX / s / "per_file" / "_batch_ts.parquet")
        assert ts["sample_item_id"].str.fullmatch(r"(s\d{2}|t\d{3})").all()


# --------------------------------------------------------------------------- the pinned levels
@pytest.mark.parametrize("name", SETS)
def test_the_core_levels_each_fixture_exactly_as_pinned(name, levels, expected):
    exp = expected[expected["set"] == name].drop(columns="set").reset_index(drop=True)
    got = levels[name].sort_values(["neutral_formula", "adduct"], kind="mergesort")
    got = _text(got, list(exp.columns)).reset_index(drop=True)
    assert len(got) == len(exp) == GOLDEN[name][0]
    assert list(zip(got["neutral_formula"], got["adduct"])) == list(zip(exp["neutral_formula"], exp["adduct"]))
    for c in exp.columns:
        bad = got[c] != exp[c]
        assert not bad.any(), (c, int(bad.sum()), exp.loc[bad, ["neutral_formula", "adduct", c]].head(3).to_dict(
            "records"), got.loc[bad, c].head(3).tolist())


@pytest.mark.parametrize("name", SETS)
def test_the_golden_vector(name, levels):
    lv = levels[name]["evidence_level"].astype(str)
    vec = "/".join(str(int((lv == k).sum())) for k in VECTOR_ORDER.split("/"))
    assert (len(lv), vec) == GOLDEN[name], f"{VECTOR_ORDER}: {vec}"
    # every pair carries one token of the scale and the claim that token implies
    assert set(lv) <= set(SC.LEVELS) | set(SC.BUCKETS)
    assert (levels[name]["claim"] == lv.map(SC.claim_class)).all()


def test_every_named_case_of_the_batches_is_in_the_fixtures(levels):
    """The pairs tests/test_levels_named.py rebuilds synthetically sit in the
    fixtures with the level the core gives them on the real batch."""
    want = {
        "v1_nitrate": {("C2HF3O2", "[M+NO3]-"): "3c", ("C2HF3O2", "[M+^NO3]-"): "3c",
                       ("C6H5NO3", "[M-H]-"): "3c", ("C6H5NO4", "[M-H]-"): "3c", ("HBr", "[M+^NO3]-"): "reagent"},
        "v1_uronium": {("C8H19N", "[M+H]+"): "4b", ("C12H27N", "[M+H]+"): "3c", ("C12H23N", "[M+H]+"): "4a",
                       ("C9H13N", "[M+H]+"): "4b", ("C16H32O2", "[M+H]+"): "3c", ("C18H36O2", "[M+H]+"): "3c",
                       ("C18H34O2", "[M+H]+"): "3c"},
    }
    for s, cases in want.items():
        L = levels[s]
        got = dict(zip(zip(L["neutral_formula"], L["adduct"]), L["evidence_level"]))
        assert {k: got.get(k) for k in cases} == cases, s


def test_the_side_channel_note_and_the_ion_form_flag_print_on_the_real_batch(levels):
    L = levels["v1_nitrate"].set_index(["neutral_formula", "adduct"])
    tfa = L.loc[("C2HF3O2", "[M+NO3]-")]
    assert "[ion form differs: entry = the anion [M-H]-, this pair = [M+NO3]-]" in tfa["named_list"]
    assert "pinned only by the context window" in L.loc[("C6H5NO3", "[M-H]-"), "split_how"]
    hbr = L.loc[("HBr", "[M+^NO3]-")]
    assert hbr["evidence"].startswith("reagent bucket · reagent identity (HBr)") and hbr["claim"] == "reagent"
    U = levels["v1_uronium"].set_index(["neutral_formula", "adduct"])
    assert U.loc[("C12H27N", "[M+H]+"), "split_how"].startswith(
        "NH4 reading inadmissible: no uronium adduct of Y (C12H24 [M+NH4]+ -- no uronium adduct ion of Y present")
    assert U.loc[("C16H32O2", "[M+H]+"), "named_mode_flag"] != ""


# --------------------------------------------------------------------------- the executable reference
def _level_ledger_or_skip():
    script = REPO / "scripts" / "level_ledger.py"
    try:
        help_ = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                               timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as e:      # pragma: no cover
        pytest.skip(f"scripts/level_ledger.py does not run: {e}")
    if "--mode" not in help_:
        pytest.skip("scripts/level_ledger.py has not the evidence scale's decision layer yet (no --mode option; "
                    "it is being rewritten beside this test): the integrator runs this comparison")
    return script


@pytest.mark.parametrize("name", SETS)
def test_level_ledger_decides_every_fixture_pair_like_the_core(name, run_dirs, levels, tmp_path):
    script = _level_ledger_or_skip()
    out = tmp_path / f"{name}.csv"
    r = subprocess.run([sys.executable, str(script), str(run_dirs[name]), "--out", str(out)],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = pd.read_csv(out, dtype=str, keep_default_na=False)
    core = _text(levels[name], ["neutral_formula", "adduct", *[c for c in ("evidence_level", "would_lift", "claim")
                                                               if c in ref.columns]])
    m = core.merge(ref, on=["neutral_formula", "adduct"], how="outer", suffixes=("", "_script"), indicator=True)
    assert (m["_merge"] == "both").all(), m.loc[m["_merge"] != "both", ["neutral_formula", "adduct", "_merge"]].head()
    for c in ("evidence_level", "would_lift", "claim"):
        if c + "_script" in m.columns:
            bad = m[m[c] != m[c + "_script"]]
            assert bad.empty, (c, len(bad), bad[["neutral_formula", "adduct", c, c + "_script"]].head(5).to_dict(
                "records"))
