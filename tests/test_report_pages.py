"""The PDF pages print what the run's data support.

Each test builds a mini run folder in tmp_path, loads it through
`pdf_report.load_context` and renders one page section into a throwaway PDF,
reading back the text the page drew: the lines handed to `_text_lines`
(unwrapped) and every text artist on the page. So a page that went back to an
old sentence, or a context key that stopped being computed, fails here even
when the helper behind it is still right (tests/test_report_text.py pins the
helpers)."""
import json
import re

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from peaky.assignment import evidence as EV
from peaky.assignment import reflists as RL
from peaky.chem import chemistry as C
from peaky.reporting import pdf_report as R

LABEL = "NO3- CIMS"


def _mz(formula, adduct, fallback):
    try:
        return float(C.ion_mz(formula, adduct))
    except Exception:  # noqa: BLE001 -- a test m/z only needs to be distinct
        return fallback


def _contaminant(adduct="[M-H]-"):
    """A closed-shell formula on a list the report's context unlocks (the
    always-active lab contaminants), and its ion m/z under `adduct`."""
    lists, _tags = RL.activate("", "", LABEL)
    for L in lists:
        for f in sorted(L.formulas):
            if C.parse_formula(f).get("H", 0) >= 2:
                return f, float(C.ion_mz(f, adduct))
    raise AssertionError("no reference list is active on a plain context")


# (neutral, adduct, tier, ion_only_of, per-file height): a nitrate-source batch whose
# reagent ion reads as an analyte (HNO3 on two channels), a pseudo-halide (INCO), the
# organic CHO / CHON readings, an Assigned and a Candidate-only accretion product and an
# ion-only row (synthetically at tier Assigned: the ion-only filters must hold on their own)
NEG_ROWS = [
    ("HNO3", "[M-H]-", "Assigned", None, 5000.0),
    ("HNO3", "[M+NO3]-", "Assigned", None, 2000.0),
    ("CINO", "[M+NO3]-", "Assigned", None, 800.0),
    ("C10H16O4", "[M+NO3]-", "Assigned", None, 3000.0),
    ("C8H12O5", "[M+NO3]-", "Assigned", None, 400.0),
    ("C10H15NO6", "[M+NO3]-", "Assigned", None, 300.0),
    ("C20H30O14", "[M+NO3]-", "Assigned", None, 200.0),
    ("C19H28O12", "[M+NO3]-", "Candidate", None, 150.0),
    ("C21H32O12", "[M]-.", "Assigned", "p9", 10000.0),
]


def _write_run(d, rows, *, levels="none", summary=None, manifest=None, ts=None):
    """A run folder: merged ledger, two per-file ledgers (each M0 row at its height,
    a reagent peak and one unexplained peak on a reference-list mass),
    batch_summary.json, run_manifest.json and an optional TS parquet.
    `levels`: 'none' = no file calibrated (every pair NO_RUN_WINDOW_TEXT),
    'assessed' = levelled, one reading with no pooled pair."""
    (d / "per_file").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    merged = []
    for i, (nf, ad, tier, ioo, _h) in enumerate(rows):
        if levels == "none":
            lv, ev = "", EV.NO_RUN_WINDOW_TEXT
        elif i == len(rows) - 2:
            lv, ev = "", EV.NO_POOLED_PAIR_TEXT
        else:
            lv, ev = "4b", "ion established; split open"
        merged.append(dict(mz=_mz(nf, ad, 100.0 + i), neutral_formula=nf, adduct=ad, tier=tier,
                           ion_score=0.9, n_files=2, formula_agree=True, ion_only_of=ioo,
                           evidence_level=lv, evidence=ev))
    pd.DataFrame(merged).to_csv(d / "merged_ledger.csv", index=False)
    un_f, un_mz = _contaminant()
    for s in ("s1", "s2"):
        pf = [dict(peak_id=i + 1, mz=m["mz"], role="M0", neutral_formula=m["neutral_formula"],
                   adduct=m["adduct"], height=rows[i][4], tier=m["tier"], ppm_error=0.2, parent_peak_id=None)
              for i, m in enumerate(merged)]
        pf.append(dict(peak_id=90, mz=61.98837, role="reagent", neutral_formula=None, adduct=None,
                       height=1e5, tier=None, ppm_error=None, parent_peak_id=None))
        pf.append(dict(peak_id=91, mz=un_mz, role="unexplained", neutral_formula=None, adduct=None,
                       height=50.0, tier=None, ppm_error=None, parent_peak_id=None))
        pd.DataFrame(pf).to_csv(d / "per_file" / f"{s}_ledger.csv", index=False)
    (d / "batch_summary.json").write_text(json.dumps(summary or {}))
    if manifest is not None:
        (d / "run_manifest.json").write_text(json.dumps(manifest))
    if ts is not None:
        ts.to_parquet(d / "ts.parquet")
    return un_f


def _flat_ts(n=40, bump=1.10, at=10):
    """A per-sample total that is flat at 1000 cps but for one sample at `bump` x."""
    t0 = pd.Timestamp("2001-01-01T00:00:00Z")
    return pd.DataFrame([dict(sample_item_id=f"x{i:03d}", datetime_utc=(t0 + pd.Timedelta(minutes=30 * i)).isoformat(),
                              mz=61.98837, height=1000.0 * (bump if i == at else 1.0)) for i in range(n)])


def _ctx(d, *, ts=False, label=LABEL):
    return R.load_context(str(d), tag="T", label=label, ts_path=str(d / "ts.parquet") if ts else None)


def _page(monkeypatch, tmp_path, section, ctx) -> list:
    """The text one section draws: its `_text_lines` lines (unwrapped), then every
    text artist on its figures."""
    raw = []
    orig_lines, orig_close = R._text_lines, R._close

    def lines(fig, ls, **kw):
        raw.extend(t for st, t in ls if st != "gap" and isinstance(t, str))
        return orig_lines(fig, ls, **kw)

    def close(pdf, fig):
        from matplotlib.text import Text
        raw.extend(t.get_text() for t in fig.findobj(Text) if t.get_text())
        orig_close(pdf, fig)

    with monkeypatch.context() as mp:
        mp.setattr(R, "_text_lines", lines)
        mp.setattr(R, "_close", close)
        with PdfPages(tmp_path / f"page_{section.__name__}.pdf") as pdf:
            section(ctx, pdf)
    return raw


def _txt(lines) -> str:
    return " ".join(" ".join(lines).split())


# --------------------------------------------------------------------------- Findings
def test_findings_page_flat_total_composition_top_species_and_oligomers(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS, ts=_flat_ts())
    ctx = _ctx(d, ts=True)
    lines = _page(monkeypatch, tmp_path, R.findings, ctx)
    txt = _txt(lines)
    # (a) the event sentence: a 1.10x maximum is no transient event
    assert "No transient event: the total signal's maximum (hour 5.0) is 1.10x the late-run baseline" in txt
    assert "a transient event)" not in txt and "then decays" not in txt
    # (c) composition by signal over the Assigned readings only: the reagent ion (HNO3) and the
    # pseudo-halide (INCO) apart; the ion-only row's 10000 cps and the Candidate's never count.
    # Assigned organic = C10H16O4 6000 + C8H12O5 800 + C20H30O14 400 (CHO) + C10H15NO6 600 (CHON)
    assert "By signal, the Assigned organic readings are 92% CHO / 8% CHON" in txt
    assert "41% CHO" not in txt and "By signal the assigned chemistry is" not in txt
    # the bright-CHO clause is scoped to the organic signal and printed before the inorganic line
    i_bright = next(i for i, t in enumerate(lines) if "a few bright CHO species" in t)
    i_inorg = next(i for i, t in enumerate(lines) if "Reagent and inorganic ions" in t)
    assert i_bright < i_inorg
    assert "The 3 brightest CHO neutrals carry 92% of the Assigned organic signal" in lines[i_bright]
    # inorganic = HNO3 14000 + INCO 1600 of 23400 Assigned
    assert "carry 67% of the Assigned M0 signal; they are kept out of CHO / CHON / CHOS" in lines[i_inorg]
    assert "ammonium/amine" not in txt                       # no NH4+ / urea channel
    # top species: organic rows only; the reagent / inorganic ions on their own line
    head = next(i for i, t in enumerate(lines) if t.strip().startswith("reagent and inorganic ions"))
    rows = [t for t in lines[:head] if re.match(r"^\s+\d+\.\d%\s", t)]
    assert rows and not any("HNO3" in t or "CINO" in t for t in rows)
    assert any("C10H16O4" in t for t in rows)
    assert lines[head + 1].strip().startswith("HNO3 ") and "CINO" in lines[head + 1]
    # (d) the oligomer line: Assigned neutrals only (not the Candidate, not the ion-only row)
    oi = next(i for i, t in enumerate(lines) if t.startswith("Accretion / oligomer products"))
    assert lines[oi + 1].strip() == "C20H30O14"
    assert "2 more high-C high-O neutral(s) hold no Assigned reading" in txt


def test_findings_page_amine_note_only_with_an_ammonium_or_urea_channel(tmp_path, monkeypatch):
    rows = [("C10H16O2", "[M+H]+", "Assigned", None, 4000.0), ("C10H16O3", "[M+H]+", "Assigned", None, 3000.0),
            ("C9H14O4", "[M+H]+", "Assigned", None, 500.0), ("C10H17NO3", "[M+H]+", "Assigned", None, 400.0)]
    d = tmp_path / "pos"
    _write_run(d, rows, ts=_flat_ts())
    txt = _txt(_page(monkeypatch, tmp_path, R.findings, _ctx(d, ts=True, label="NO+ CIMS")))
    assert "the Assigned organic readings are" in txt and "ammonium/amine" not in txt
    d2 = tmp_path / "nh4"
    _write_run(d2, rows[:3] + [("C10H14O3", "[M+NH4]+", "Assigned", None, 400.0)], ts=_flat_ts())
    txt = _txt(_page(monkeypatch, tmp_path, R.findings, _ctx(d2, ts=True, label="NH4+ CIMS")))
    assert "the count is inflated by mass-degenerate ammonium/amine re-reads" in txt


# --------------------------------------------------------------------------- Composition
def test_composition_page_counts_the_inorganic_class_apart(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS)
    txt = _txt(_page(monkeypatch, tmp_path, R.composition, _ctx(d)))
    # 8 distinct neutrals: HNO3 + INCO inorganic, 5 CHO (one ion-only), 1 CHON
    assert re.search(r"inorg\. 2 \(25%\)", txt) and re.search(r"CHON 1 \(12%\)", txt)
    assert "Assigned readings, weighted by signal" in txt
    assert "92% of the Assigned organic signal" in txt and "67% of all Assigned M0 signal" in txt


# --------------------------------------------------------------------------- Reference lists
def test_reference_lists_page_prints_the_measured_chance_level(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS)
    ctx = _ctx(d)
    assert set(ctx["reflist"]["null_near"]) == set(R.REFLIST_NULL_SHIFTS_PPM)
    txt = _txt(_page(monkeypatch, tmp_path, R.reference_lists, ctx))
    assert "Chance level: the same unexplained peaks shifted by ±15-40 ppm give" in txt
    assert "observed near-0-ppm matches" in txt
    assert "single-digit" not in txt and "far exceed chance" not in txt


# --------------------------------------------------------------------------- Cover
def test_cover_names_the_runs_code_formats_the_stamp_window_and_says_levels_not_assessed(tmp_path, monkeypatch):
    d = tmp_path / "run"
    summary = {"traces": {"n_rows": 10, "n_recentred": 3, "median_abs_move_ppm": 1.234567, "n_traces": 9,
                          "n_collapsed": 1, "sigma_ppm": 3.14159265, "stamp_tol_ppm": 9.22777806538331}}
    _write_run(d, NEG_ROWS, summary=summary,
               manifest={"code": {"package_version": "0.0.1", "git": {"commit": "0123456789abcdef"}}})
    ctx = _ctx(d)
    txt = _txt(_page(monkeypatch, tmp_path, R.cover, ctx))
    assert "assigned by peaky 0.0.1 · git 0123456" in txt and "assign v" not in txt
    assert "stamping window ±9.23 ppm" in txt and "9.2277" not in txt
    assert "median move 1.23 ppm" in txt and "per-ion scatter 3.14 ppm" in txt
    # no batch_summary key: the notice is read off the merged rows (every pair has no run window)
    assert ctx["levels_not_assessed"] == EV.levels_not_assessed_reason([EV.NO_RUN_WINDOW_TEXT], [None])
    assert "Evidence levels were not assessed: no file had enough isotope-backed core rows" in txt


def test_cover_prints_the_batch_summarys_notice_and_none_on_a_levelled_run(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS, summary={"levels_not_assessed_reason": "Evidence levels were not assessed: recorded."})
    assert "Evidence levels were not assessed: recorded." in _txt(_page(monkeypatch, tmp_path, R.cover, _ctx(d)))
    d2 = tmp_path / "levelled"
    _write_run(d2, NEG_ROWS, levels="assessed")
    ctx = _ctx(d2)
    assert ctx["levels_not_assessed"] is None
    assert "not assessed:" not in _txt(_page(monkeypatch, tmp_path, R.cover, ctx))


# --------------------------------------------------------------------------- Claims + Evidence levels
def test_claims_and_levels_pages_scope_the_neutral_claim_and_give_the_true_no_level_reason(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS)
    ctx = _ctx(d)
    claims = _txt(_page(monkeypatch, tmp_path, R.claims, ctx))
    assert "neutral = 4a (the neutral established among the run's declared reagent channels)" in claims
    assert "side channels locked" in claims
    n = len(NEG_ROWS)
    assert (f"{n} merged row(s) carry no level because no file of the run calibrated the degeneracy window "
            "(see the Evidence levels page); they read tentative.") in claims
    assert "batch-level re-read" not in claims
    levels = _txt(_page(monkeypatch, tmp_path, R.evidence_levels, ctx))
    assert "Evidence levels were not assessed: no file had enough isotope-backed core rows" in levels
    assert (f"{n} merged row(s) carry no level because no file of the run calibrated the degeneracy window "
            "(see above); they read tentative.") in levels
    assert "batch-level re-read" not in levels and "no pooled pair" not in levels


def test_claims_page_names_a_batch_level_re_read_only_where_the_evidence_says_so(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _write_run(d, NEG_ROWS, levels="assessed")
    ctx = _ctx(d)
    claims = _txt(_page(monkeypatch, tmp_path, R.claims, ctx))
    assert ("1 merged row(s) carry no level because their reading exists in no pooled pair (a batch-level "
            "re-read); they read tentative.") in claims
    assert "calibrated the degeneracy window" not in claims
    levels = _txt(_page(monkeypatch, tmp_path, R.evidence_levels, ctx))
    assert "not assessed: no file" not in levels and "exists in no pooled pair" in levels


# --------------------------------------------------------------------------- Coverage + Methods
def test_coverage_and_methods_read_the_runs_amine_r_and_scorer(tmp_path, monkeypatch):
    rows = [("C10H16O2", "[M+H]+", "Assigned", None, 4000.0), ("C10H14O3", "[M+NH4]+", "Assigned", None, 900.0),
            ("C9H14O4", "[M+H]+", "Candidate", None, 300.0)]
    d = tmp_path / "nh4"
    _write_run(d, rows, levels="assessed", summary={"amine_r_min": 0.55, "scorer": "local"})
    ctx = _ctx(d, label="NH4+ CIMS")
    cov = _txt(_page(monkeypatch, tmp_path, R.coverage, ctx))
    assert "kept as NH4 only when its trace co-varies (r>=0.55) with the protonated/urea parent" in cov
    assert "r>=0.7" not in cov and "server isotope-scored" not in cov
    meth = _txt(_page(monkeypatch, tmp_path, R.methods, ctx))
    assert "isotope-pattern scoring of every candidate in-process (the local scorer)" in meth
    assert "amine co-variation r>=0.55." in meth and "its trace co-varies with its parent, r>=0.55" in meth
    assert "r>=0.7" not in meth and "server isotope-scored matching" not in meth
    # a run that recorded neither: the code default, said to be one, and no scorer claimed
    d2 = tmp_path / "old"
    _write_run(d2, rows, levels="assessed", summary={})
    meth = _txt(_page(monkeypatch, tmp_path, R.methods, _ctx(d2, label="NH4+ CIMS")))
    r0 = R._amine_r_min({})[0]
    assert f"amine co-variation r>={r0:g}, the code default; the run did not record it" in meth
    assert "this run did not record which" in meth
    assert np.isclose(r0, 0.6)
