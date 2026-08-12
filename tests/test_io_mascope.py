"""Tests for io_mascope.py.

Offline: parser tests against tests/fixtures/match_tree.json (always run).
Live:    smoke test against Mascope, only when MASCOPE_LIVE=1.
Run: python3 tests/test_io_mascope.py   [MASCOPE_LIVE=1 for live]
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import io_mascope as IO  # noqa: E402

PASS = FAIL = 0
HERE = Path(__file__).resolve().parent


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# ---------- isotope label parsing ----------
check("base formula -> M0", IO.parse_isotope_label("C4H5O2-") == ("M0", True))
check("13C single", IO.parse_isotope_label("[13C]C3H5O2-") == ("13C", False))
check("13C2 multiple", IO.parse_isotope_label("[13C]2C2H5O2-") == ("13C2", False))
check("81Br", IO.parse_isotope_label("[81Br]CH2O2-") == ("81Br", False))
check("combined 13C+81Br",
      IO.parse_isotope_label("[13C]2[81Br]C2H6O2-") == ("13C2+81Br", False),
      IO.parse_isotope_label("[13C]2[81Br]C2H6O2-"))
check("18O", IO.parse_isotope_label("[18O]CH2BrO-") == ("18O", False))

# ---------- flatten_match_tree against the real fixture ----------
fix = HERE / "fixtures" / "match_tree.json"
if not fix.exists():
    check("fixture present", False, f"missing {fix}")
else:
    tree = json.load(open(fix))
    df = IO.flatten_match_tree(tree)
    check("flatten returns rows", len(df) > 0, len(df))
    cols = {"compound_formula", "compound_score", "ion_formula", "ion_score",
            "mechanism_id", "iso_label", "is_base", "theo_mz", "iso_score",
            "sample_peak_id", "ppm_error"}
    check("flatten has expected columns", cols <= set(df.columns),
          cols - set(df.columns))
    # exactly one base (M0) per ion
    per_ion = df.groupby(["compound_formula", "ion_formula"])["is_base"].sum()
    check("exactly one M0 base per ion", (per_ion == 1).all(), per_ion.to_dict())
    # C4H6O2 present with a base ion C4H5O2-
    sub = df[(df.compound_formula == "C4H6O2") & (df.is_base)]
    check("C4H6O2 base ion C4H5O2- present",
          "C4H5O2-" in set(sub.ion_formula), sorted(set(sub.ion_formula)))
    # base ion of a matched compound has a sample_peak_id and a finite iso_score
    base_matched = sub[sub.ion_formula == "C4H5O2-"]
    check("matched base ion has a sample_peak_id",
          base_matched["sample_peak_id"].notna().any(),
          base_matched[["sample_peak_id", "iso_score"]].to_dict("records"))
    check("base ion iso_score is high (>0.9)",
          (base_matched["iso_score"].fillna(0) > 0.9).any(),
          base_matched["iso_score"].tolist())
    # ppm error is defined only for genuinely matched isotopes, and is small
    real = df[df["ppm_error"].notna()]
    check("matched isotopes have a real sample_peak_id",
          real["sample_peak_id"].notna().all() and len(real) > 0, len(real))
    check("ppm errors |ppm|<10 for matched isotopes",
          (real["ppm_error"].abs() < 10).all(), real["ppm_error"].describe().to_dict())
    # brominated ion form present too ([M+Br]- => C4H6BrO2-)
    check("Br adduct ion form present",
          any("Br" in f for f in df[df.compound_formula == "C4H6O2"].ion_formula.dropna()))

# ---------- empty input safety ----------
import pandas as pd  # noqa: E402
check("flatten([]) -> empty df", len(IO.flatten_match_tree([])) == 0)


# ---------- score_candidates failure policy ----------
# This exercises the SERVER (match_compounds) batching/failure path; local scoring
# is now the default, so force the server path for these checks.
class _Matching:
    def match_compounds(self, sample_id, formulas, match_params, ionization_mechanism_ids):
        if "BAD" in formulas:
            raise RuntimeError("boom")
        return []


class _Client:
    matching = _Matching()


_prev_local = os.environ.get("PEAKY_LOCAL_SCORING")
os.environ["PEAKY_LOCAL_SCORING"] = "0"
try:
    try:
        IO.score_candidates(_Client(), "SID", ["A", "BAD"], batch=1, workers=1)
        raised = False
    except RuntimeError as e:
        raised = "match_compounds failed" in str(e) and "BAD" in str(e)
    check("score_candidates raises on partial batch failure by default", raised)

    partial = IO.score_candidates(_Client(), "SID", ["A", "BAD"], batch=1,
                                  workers=1, allow_partial=True)
    check("score_candidates allow_partial records failures",
          len(partial.attrs.get("match_batch_failures", [])) == 1,
          partial.attrs)
finally:
    if _prev_local is None:
        os.environ.pop("PEAKY_LOCAL_SCORING", None)
    else:
        os.environ["PEAKY_LOCAL_SCORING"] = _prev_local

# ---------- fetch_batch_peaks passes a compiled literal batch-name pattern -----
# The SDK escapes plain strings itself but uses a compiled pattern as-is; a
# pre-escaped STRING would be escaped twice and silently match nothing. The ^ in
# a '^Nitrate' (15N) batch name must reach the SDK escaped inside a Pattern.
class _LP:
    seen = None
    def load_peaks(self, *, dataset, batches, **kwargs):   # **kwargs: tolerate confirm_above=
        _LP.seen = batches
        return pd.DataFrame({"mz": [100.0], "height": [1.0], "sample_item_id": ["s"]})
_name = "^Nitrate (synthetic) m/z 100-200"
IO.fetch_batch_peaks(_LP(), "DS", _name)
check("fetch_batch_peaks passes a compiled, escaped batch-name pattern",
      isinstance(_LP.seen, re.Pattern) and _LP.seen.pattern == re.escape(_name)
      and (_LP.seen.flags & re.IGNORECASE), _LP.seen)

# ---------- fetch_batch_samples passes an escaped STRING batch name -----------
# The opposite SDK contract: samples.list treats a plain string as a RAW regex
# (so it must arrive escaped) and REJECTS compiled patterns (pandas cannot
# combine case=False with a compiled pattern).
class _SL:
    seen = None
    def list(self, *, batch, dataset=None, drop_columns=None):   # noqa: A001
        _SL.seen = batch
        return pd.DataFrame({"sample_item_id": ["s"]})
class _CS:
    samples = _SL()
IO.fetch_batch_samples(_CS(), _name)
check("fetch_batch_samples passes an escaped string batch name",
      isinstance(_SL.seen, str) and _SL.seen == re.escape(_name), _SL.seen)

# ---------- flatten_match_tree re-anchors a 100%-labelled (^N) reagent ----------
# The 15N nitrate adduct: server tags the all-light 14N form as the M0 base (no
# signal) and the single-15N line (the REAL monoisotopic ion) as a '15N' child.
# flatten must move is_base onto the 15N line, else the whole channel is dropped.
def _iso(isof, mz, pmz, inten, pid):
    return {"target_isotope_formula": isof, "mz": mz, "sample_peak_mz": pmz,
            "sample_peak_intensity": inten, "sample_peak_id": pid,
            "relative_abundance": 1.0, "match_score": 0.96, "match_category": 2,
            "match_abundance_error": 0.0}
_no3_tree = [{"target_compound_formula": "C4H6O4", "match_score": 0.96, "match_category": 2,
    "children": [{"target_ion_formula": "C4H6O7^N-", "match_score": 0.96, "match_category": 2,
        "ionization_mechanism_id": "m15", "children": [
            _iso("C4H6O7^N-", 180.015, 180.005, 0.0, "p0"),       # 14N phantom base, no signal
            _iso("[15N]C4H6O7-", 181.012, 181.012, 6312.0, "p1")]}]}]   # real 15N peak
_fn = IO.flatten_match_tree(_no3_tree)
_b = _fn[_fn["is_base"]]
check("flatten re-anchors ^N reagent base onto the 15N peak",
      len(_b) == 1 and abs(float(_b["theo_mz"].iloc[0]) - 181.012) < 1e-3
      and float(_b["sample_peak_intensity"].iloc[0]) == 6312.0, _b[["theo_mz", "is_base"]].to_dict())
check("flatten leaves the phantom 14N base non-base",
      not bool(_fn[(_fn["theo_mz"] == 180.015)]["is_base"].iloc[0]))
# control: a NON-labelled ion is untouched (base stays the all-light M0)
_plain = [{"target_compound_formula": "C9H14O5", "match_score": 0.9, "match_category": 2,
    "children": [{"target_ion_formula": "C9H13O5-", "match_score": 0.9, "match_category": 2,
        "ionization_mechanism_id": "mH", "children": [
            _iso("C9H13O5-", 201.076, 201.076, 5000.0, "q0"),
            _iso("[13C]C8H13O5-", 202.079, 202.079, 500.0, "q1")]}]}]
_pf = IO.flatten_match_tree(_plain)
check("flatten does NOT re-anchor a normal (no-^) ion",
      abs(float(_pf[_pf["is_base"]]["theo_mz"].iloc[0]) - 201.076) < 1e-3)

# ---------- estimate_offset: rough offset from the sample's own matches ----------
from peaky import chemistry as _C  # noqa: E402
# build a synthetic match table at a uniform -1.9 ppm offset (Br-CIMS)
_off_rows = []
for f, mech in [("C10H16O4", "+Br-"), ("C10H16O3", "+Br-"), ("C5H10O3", "+Br-"),
                ("C2HF3O2", "+Br-"), ("C9H14O4", "-H+"), ("C6H12O3", "+Br-"),
                ("C4H6O4", "-H+"), ("HNO3", "+Br-"), ("C8H12O4", "+Br-"),
                ("C10H18O4", "+Br-")]:
    add = IO.MECH_TO_ADDUCT[mech]
    mz = _C.ion_mz(f, add) * (1 - 1.9e-6)            # observed = -1.9 ppm
    _off_rows.append({"target_compound_formula": f, "ionization_mechanism": mech,
                      "target_isotope_formula": f, "mz": mz})
# a heavy-isotope row + an unmatched row must be ignored
_off_rows.append({"target_compound_formula": "C10H16O4", "ionization_mechanism": "+Br-",
                  "target_isotope_formula": "[13C]C9H16BrO4-", "mz": 999.0})
_off_rows.append({"target_compound_formula": None, "ionization_mechanism": "+Br-",
                  "target_isotope_formula": None, "mz": 300.0})
off = IO.estimate_offset(pd.DataFrame(_off_rows))
check("estimate_offset recovers the -1.9 ppm instrument offset",
      off is not None and abs(off + 1.9) < 0.2, off)
check("estimate_offset None when too few matches",
      IO.estimate_offset(pd.DataFrame(_off_rows[:3])) is None)
check("estimate_offset None on a table with no match columns",
      IO.estimate_offset(pd.DataFrame({"mz": [1.0, 2.0]})) is None)

# ---------- _find_env precedence: explicit > $MASCOPE_ENV > project-local .env ----------
import tempfile  # noqa: E402

_cwd0, _menv0, _search0 = os.getcwd(), os.environ.pop("MASCOPE_ENV", None), IO.ENV_SEARCH
try:
    with tempfile.TemporaryDirectory() as _d:
        proj = os.path.join(_d, ".env"); open(proj, "w").write("MASCOPE_URL=x\n")
        other = os.path.join(_d, "other.env"); open(other, "w").write("MASCOPE_URL=y\n")
        os.chdir(_d)
        IO.ENV_SEARCH = [".env"]                       # isolate from home/repo paths
        check("_find_env: explicit path wins", IO._find_env(other) == other)
        os.environ["MASCOPE_ENV"] = other
        check("_find_env: $MASCOPE_ENV honored (no explicit)", IO._find_env() == other)
        os.environ.pop("MASCOPE_ENV", None)
        check("_find_env: finds a project-local ./.env", os.path.samefile(IO._find_env(), proj))
finally:
    os.chdir(_cwd0); IO.ENV_SEARCH = _search0
    if _menv0 is not None:
        os.environ["MASCOPE_ENV"] = _menv0
check("_REPO_ENV points at the repo-root .env (next to the package)",
      IO._REPO_ENV == os.path.join(
          os.path.dirname(os.path.dirname(os.path.abspath(IO.__file__))), ".env"))

# ---------- legacy (workspace-based) server resolution (offline, monkeypatched) ----------
_WS = pd.DataFrame([
    {"workspace_name": "Demo acquisitions", "workspace_id": "WACQ", "workspace_type": "ACQUISITION"},
    {"workspace_name": "Sandbox", "workspace_id": "WSBX", "workspace_type": "ANALYSIS"},
])
_BATCHES = pd.DataFrame([
    {"workspace_id": "WACQ", "sample_batch_name": "Sample run Uronium acquisition",
     "sample_batch_id": "BURO", "polarity": "+"},
    {"workspace_id": "WSBX", "sample_batch_name": "Chamber tests", "sample_batch_id": "BCHM", "polarity": "+-"},
    {"workspace_id": "WSBX", "sample_batch_name": "Uronium scratch copy", "sample_batch_id": "BDUP", "polarity": "+"},
])

_orig_batches, _orig_ws = IO._legacy_all_batches, IO._legacy_workspaces
IO._legacy_all_batches = lambda client: _BATCHES.copy()
IO._legacy_workspaces = lambda client: _WS.copy()
try:
    check("resolve_batch_id exact match",
          IO.resolve_batch_id(None, "Sample run Uronium acquisition") == "BURO")
    check("resolve_batch_id case-insensitive substring",
          IO.resolve_batch_id(None, "chamber TESTS") == "BCHM")
    amb = False                                  # 'Uronium' substring hits two batches
    try:
        IO.resolve_batch_id(None, "Uronium")
    except RuntimeError as e:
        amb = "disambiguate" in str(e)
    check("resolve_batch_id ambiguous raises", amb)
    check("resolve_batch_id workspace-scoped disambiguates",
          IO.resolve_batch_id(None, "Uronium", dataset="Sandbox") == "BDUP")
    nf = False
    try:
        IO.resolve_batch_id(None, "no such batch")
    except RuntimeError:
        nf = True
    check("resolve_batch_id not-found raises", nf)

    class _FakeDS:
        def list(self):
            return None              # legacy server: /api/datasets absent

    class _FakeClient:
        datasets = _FakeDS()

        def __getattr__(self, _):    # batches.list(...) -> AttributeError -> legacy fallback
            raise AttributeError

    ds = IO.list_datasets(_FakeClient())
    check("list_datasets falls back to workspaces (reshaped)",
          list(ds["dataset_name"]) == ["Demo acquisitions", "Sandbox"]
          and "dataset_id" in ds.columns and "dataset_type" in ds.columns)
    lb = IO.list_batches(_FakeClient(), dataset="Demo acquisitions")
    check("list_batches legacy fallback filters by workspace",
          len(lb) == 1 and lb.iloc[0]["sample_batch_id"] == "BURO")
finally:
    IO._legacy_all_batches, IO._legacy_workspaces = _orig_batches, _orig_ws

IO._patch_datasets_list_for_legacy_servers()     # idempotent + swallows NotFoundError
IO._patch_datasets_list_for_legacy_servers()
from mascope_sdk.resources.datasets import DatasetsResource as _DSR  # noqa: E402
check("legacy datasets.list patch installed + idempotent",
      getattr(_DSR.list, "_legacy_safe", False))


# ---------- live smoke (opt-in) ----------
if os.environ.get("MASCOPE_LIVE") == "1":
    print("\n-- live smoke --")
    cl = IO.connect()
    mech = IO.resolve_mechanism_ids(cl, ["-H+", "+Br-"])
    check("live: mechanisms resolved", set(mech) == {"-H+", "+Br-"}, mech)
    SID = os.environ.get("MASCOPE_SID")          # set to one of YOUR sample ids
    if SID:
        peaks = IO.fetch_peaks(cl, SID, use_cache=False)
        check("live: peaks fetched", peaks is not None and len(peaks) > 0,
              None if peaks is None else peaks.shape)
        scored = IO.score_candidates(cl, SID, ["C4H6O2", "C6H8O3"], mechanism_ids=None)
        check("live: score_candidates returns per-iso rows", len(scored) > 0, len(scored))
        check("live: at least one high ion_score",
              (scored["ion_score"].fillna(0) > 0.8).any())
    else:
        print("(set MASCOPE_SID=<sample id> to also smoke peak fetch + scoring)")
else:
    print("\n(live smoke skipped; set MASCOPE_LIVE=1 to run)")

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
