"""Local, in-process scoring backend — a drop-in for the network `match_compounds`.

peaky's hot loop enumerates candidate neutral formulas and asks Mascope to score
them. The backend `match_compounds` endpoint is a *deep-annotation* primitive: per
(formula x adduct) it computes the full theoretical isotope envelope and returns
the whole compound->ion->isotopologue tree (matched AND unmatched), which is
`O(candidates x adducts x envelope)` work + payload (tens of thousands of rows per
call) for an `O(matches)` signal. That drove the timeouts and the OOM.

`mascope_tools.composition` (public PyPI, same authors) provides the SAME scoring
maths locally and vectorised: `predict_isotopes` (IsoSpec) for the envelope and
`score_pattern_v2` for the per-ion score. So we run the screening in-process: no
network, no 30k-row trees, only matched isotopologues emitted.

What a candidate is judged at is the sample's own measurement, carried in a
`PatternScoring`: the fitted width and offset of its mass errors, the window a
predicted line may be matched in, the depth its envelope is predicted to. Its
predecessor `score_pattern` averaged a mass and an intensity term over the lines
a candidate MATCHED and scaled the mass by a fixed 5 ppm - an Orbitrap's number,
which on a TOF at 5-15 ppm is a candidate's whole mass error, so every candidate
scored near zero and a TOF reference run committed almost nothing. v2 is a
Gaussian mass likelihood at the sample's own width times an intensity likelihood
at the peak's own signal-to-noise, and it charges a predicted line that is absent
where the noise says it should have been visible.

`score_candidates_local` returns the SAME columns as `io_mascope.flatten_match_tree`
so peaky's passes/arbitration are unchanged — only the *source* of the scores moves
from the server to the local library. Mascope is still the scorer (its authored,
released code), just executed locally and pinned to a library version - and since
that library holds the fit and the class widths as well as the score, the two
engines judge a mass error at one width rather than at two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Category thresholds on the v2 fit's scale. v1 put a correct assignment near
# 0.95 whatever its envelope did, so 0.8/0.4 were bands on a number that barely
# moved; the fit charges a missing line and a mass off the sample's own width,
# and a correct ion with a lone unremarkable peak lands in the 0.5-0.8 range
# these bands now split. Mascope's own targeted matcher re-read the same bands
# for v2 as 0.8/0.5 (DEFAULT_*_MATCH_THRESHOLD_V2); `possible` sits lower here
# because peaky arbitrates between candidates afterwards and a band is a
# shortlist, not a verdict.
PROBABLE_THRESHOLD = 0.8
POSSIBLE_THRESHOLD = 0.4
INTENSITY_TOLERANCE = 0.4  # mascope_tools ISOTOPE_MATCHING_INTENSITY_TOLERANCE

# ---------------------------------------------------------------------------
# The signal-to-noise the score reads (C46)
# ---------------------------------------------------------------------------
# Three terms of the v2 fit are set by a peak's signal-to-noise: whether an
# ABSENT predicted line is charged (it is, where rel * SNR_base >= k_detect),
# the intensity tolerance of a matched line (its Poisson-like 1/SNR), and the
# centroiding term of its mass width. The column the server sends is taken at
# its word -- and on one TOF it is not a signal-to-noise at all: over a file's
# ~1500-2600 picked peaks it does not track height (Spearman -0.07..0.17; a
# 478-count peak carries 1.1, a 2-count peak 12), where every Orbitrap file
# gives 0.999 with a flat ~19 cps implied noise. Read as an SNR, that column
# excuses every missing line of a bright ion (no 81Br line of a bromide cluster
# is ever charged) and charges dim ions for lines they could never show.
# A file whose column fails the test is scored at the counting-statistics SNR
# of an ion-counting detector instead: h / sqrt(h + edge^2), the Poisson noise
# of the peak's own counts in quadrature with the picker's detection edge (the
# 1st percentile of the file's picked heights, passes.config.noise_edge). The
# ratio tolerance that gives a matched line is then its real Poisson scatter,
# an absent line is charged where the counts say it was within reach, and a
# handful-of-ions centroid is judged at the width such a centroid has.
SNR_ASSESS_MIN_PEAKS = 30   # fewer peaks: the column cannot be judged, keep it
SNR_MIN_SPEARMAN = 0.5      # a signal-to-noise tracks height; under this it is not one
SNR_SOURCE_SERVER = "server"
SNR_SOURCE_POISSON = "poisson_fallback"
SNR_SOURCE_NONE = "none"


def assess_snr(peaks: pd.DataFrame, *, height_col: str = "height",
               snr_col: str = "signal_to_noise", peak_id_col: str = "peak_id") -> dict:
    """Is the table's signal-to-noise column one? {source, spearman, n, edge}.

    `source` is `SNR_SOURCE_NONE` when no peak carries a value (the score runs
    in its no-SNR mode, as before), `SNR_SOURCE_SERVER` when the column tracks
    height (Spearman >= SNR_MIN_SPEARMAN over the file's peaks) or there are
    too few peaks to judge it, and `SNR_SOURCE_POISSON` when it does not --
    then `edge` (the detection edge the fallback uses) is set. One peak per
    `peak_id` (the raw server table has one row per match)."""
    from peaky.assignment.passes.config import noise_edge

    if peaks is None or snr_col not in getattr(peaks, "columns", []) or height_col not in peaks.columns:
        return {"source": SNR_SOURCE_NONE, "spearman": None, "n": 0, "edge": None}
    p = peaks.drop_duplicates(peak_id_col) if peak_id_col in peaks.columns else peaks
    h = pd.to_numeric(p[height_col], errors="coerce")
    s = pd.to_numeric(p[snr_col], errors="coerce")
    ok = h.notna() & s.notna() & np.isfinite(h) & np.isfinite(s) & (s > 0)
    n = int(ok.sum())
    if n == 0:
        return {"source": SNR_SOURCE_NONE, "spearman": None, "n": 0, "edge": None}
    edge = noise_edge(h[h.notna()].to_numpy())
    if n < SNR_ASSESS_MIN_PEAKS:
        return {"source": SNR_SOURCE_SERVER, "spearman": None, "n": n, "edge": edge}
    # Spearman as the Pearson correlation of the ranks: a constant column (every
    # peak the same value) has no rank correlation and reads 0 -- it does not
    # track height -- without scipy's warning about it
    hr, sr = h[ok].rank(), s[ok].rank()
    rho = 0.0 if hr.nunique() < 2 or sr.nunique() < 2 else float(hr.corr(sr))
    if not np.isfinite(rho):
        rho = 0.0
    source = SNR_SOURCE_SERVER if rho >= SNR_MIN_SPEARMAN else SNR_SOURCE_POISSON
    return {"source": source, "spearman": round(rho, 4), "n": n, "edge": edge}


def poisson_snr(heights, edge: float | None) -> np.ndarray:
    """The counting-statistics signal-to-noise of each peak: h / sqrt(h + edge^2).
    A non-positive or unknown `edge` leaves sqrt(h) alone; a non-finite or
    non-positive height reads NaN (no SNR for that peak)."""
    h = np.asarray(heights, dtype=float)
    e2 = float(edge) ** 2 if edge is not None and np.isfinite(edge) and edge > 0 else 0.0
    out = np.full(h.shape, np.nan)
    ok = np.isfinite(h) & (h > 0)
    out[ok] = h[ok] / np.sqrt(h[ok] + e2)
    return out


def with_poisson_snr(peaks: pd.DataFrame, edge: float | None, *, height_col: str = "height",
                     snr_col: str = "signal_to_noise") -> pd.DataFrame:
    """A copy of `peaks` whose `snr_col` is the counting-statistics SNR."""
    p = peaks.copy()
    p[snr_col] = poisson_snr(pd.to_numeric(p[height_col], errors="coerce").to_numpy(dtype=float), edge)
    return p


def pred_nat_i(pred_int, i) -> float:
    """line i's natural-abundance share of the M0 (index 0)"""
    return float(pred_int[i] / pred_int[0])


def _respond(pred_rel, labels, snr0, resp):
    """The lower end of every minor carbon / oxygen line's band: its predicted share x the response's lower
    end at that line's expected S/N (the M0's S/N x its predicted share; iso_response.band); other lines
    and the M0 unchanged."""
    from peaky.assignment import iso_response as IR

    out = np.array(pred_rel, dtype=float, copy=True)
    for i, lab in enumerate(labels):
        if i == 0 or not IR.is_scaled_label(lab):
            continue
        out[i] = IR.band(resp, out[i], snr0)
    return out


def adduct_to_mech(adduct: str) -> str:
    """peaky adduct label -> the mechanism string the library scores it as.

    The library reads the standard adduct notation, which peaky's labels are
    written in, so this is the label's one spelling: its terms in the library's
    order, a grouped term split ('[M+(CH4N2O)H]+' -> '[M+CH4N2O+H]+',
    '[M+HBr+Br]-' -> '[M+Br+HBr]-'), a labelled reagent as it is ('[M+^NO3]-').
    What it says about the ion is what the label says: '[M-H]-' is the
    deprotonated anion and '[M-H]+' the hydride-abstracted cation. The legacy
    '<operation><moiety><moiety charge>' spelling this used to emit got both
    backwards once the library read it by its grammar ('-H-' is a hydride
    removed, a cation), which is why peaky main held the library below the
    release that does.

    A label that adds and removes at once ('[M-H+I2]-') is a decomposition
    alias with no mechanism of its own, and the library refuses it with a
    ValueError, as it does anything that is not a label at all; the raise is
    the contract the relabel-only channels rely on."""
    from mascope_tools.composition import standard_notation

    return standard_notation(adduct.strip())


def _category(score: float) -> str:
    if score >= PROBABLE_THRESHOLD:
        return "probable"
    if score >= POSSIBLE_THRESHOLD:
        return "possible"
    return "unlikely"


def _caret_ion_formula(neutral: str, im) -> str:
    """Ion-formula string (charge sign stripped) for a caret heavy-isotope
    neutral under mechanism `im`, e.g. ('C5H8^NO6', -H-) -> 'C5H7^NO6'. Adds or
    subtracts the mechanism's own composition (which may itself carry '^N', as
    the 15N-nitrate adduct does) so the total heavy-isotope count is exact."""
    from peaky.chem import chemistry as C

    cnt = dict(C.parse_formula(neutral))
    add = C.parse_formula(getattr(im, "formula", "") or "")
    sign = 1 if getattr(im, "addition", True) else -1
    for el, n in add.items():
        cnt[el] = cnt.get(el, 0) + sign * n
    cnt = {k: v for k, v in cnt.items() if v > 0}
    return C.format_formula(cnt)


def score_candidates_local(
    peaks: pd.DataFrame,
    formulas: list[str],
    adducts: list[str] | None = None,
    *,
    mechanisms: list[str] | None = None,
    scoring=None,
    centre=None,
    purity: float | None = None,
    mz_col: str = "mz",
    intensity_col: str = "height",
    peak_id_col: str = "peak_id",
    snr_col: str = "signal_to_noise",
    iso_response: dict | None = None,
) -> pd.DataFrame:
    """Score candidate NEUTRAL formulas against a sample's peaks, locally.

    Channels are given either as peaky `adducts` (labels like '[M+Br]-') or as
    already-resolved mascope mechanism strings via `mechanisms` (e.g. '+Br-', as
    `io_mascope.score_candidates` has them) - the latter skips `adduct_to_mech`.

    `scoring` is a `mascope_tools.composition.PatternScoring`: the sample's
    fitted mass width and offset, and the window a predicted line is matched in.
    `io_mascope.scoring_for_sample` builds it from the sample's own targeted
    matches and its instrument class. Passing nothing scores at the library's
    defaults, which are an Orbitrap's.

    `centre` (C42) is the sample's own mass-dependent centre, a
    `masscal.MassTrend` its calibration accepted: each line's error is then
    judged against the centre at THAT line's m/z (clamped to the trend's m/z
    coverage) instead of the one offset in `scoring`. An Orbitrap whose error
    runs as 1/mz (+0.8 ppm at m/z 131, -0.2 at 400 on the labelled-nitrate run)
    otherwise has every low-mass ion judged ~1 sigma off a centre it does not
    sit at. None keeps the constant offset.

    Every row also carries `ion_score_massfree`: the same score with every
    matched line's mass error set to the centre, i.e. what the isotope pattern
    alone says. The calibration selects the rows it fits the mass trend on by
    it (passes.core.calibrate), so the trend is not measured on rows a
    mis-centred mass term already filtered to sit near the constant offset.

    A `signal_to_noise` column on `peaks` is carried into the score, where it
    decides how a faint line is judged: an absent line is charged only where the
    noise says it should have been visible. Without the column the score runs in
    its no-SNR mode and charges on predicted abundance alone.

    Mirrors `io_mascope.score_candidates` + `flatten_match_tree`: returns one row
    per (compound, ion, isotopologue) with the same columns the passes consume.
    Only isotopologues that the predicted envelope produces are emitted; unmatched
    isotopologues carry `sample_peak_id=None` and `ppm_error=None`.
    """
    from mascope_tools.composition import PatternScoring, utils
    from mascope_tools.composition.heuristic_filter import (
        DETECT_SNR_K,
        anchor_on_monoisotopic,
        predict_isotopes,
        score_pattern_v2,
    )

    from peaky.assignment import masscal as MC
    from peaky.chem import isotopes as ISO

    # A '^X' candidate's envelope carries the reagent's unlabelled impurity line,
    # whose height is 1 - the bottle's isotopic purity. Unset => the run's active
    # reagent purity (ReagentProfile.purity, published by assign.run), which
    # defaults to isotopes.LABEL_PURITY_15N. Inert for non-labelled ions.
    if purity is None:
        purity = ISO.label_purity()

    scoring = scoring if scoring is not None else PatternScoring()
    has_snr = snr_col in peaks.columns
    # scored by another column (area): the lines' HEIGHTS are still what the rows report as their
    # intensity -- every reader of `sample_peak_intensity` compares it with height gates in cps
    keep_h = intensity_col != "height" and "height" in peaks.columns
    columns = ([mz_col, intensity_col, peak_id_col] + (["height"] if keep_h else [])
               + ([snr_col] if has_snr else []))
    peaks = (
        peaks[columns]
        .dropna(subset=[mz_col])
        .drop_duplicates(peak_id_col)  # raw server peaks have one row per match
        .sort_values(mz_col)
    )
    mzs = peaks[mz_col].to_numpy(dtype=float)
    ints = peaks[intensity_col].to_numpy(dtype=float)
    hts = peaks["height"].to_numpy(dtype=float) if keep_h else ints
    pids = peaks[peak_id_col].to_numpy()
    # NaN where the file records no estimate for a peak. The score treats that
    # as "not measured" and judges the line at the instrument width, which is
    # what a missing estimate means; a zero would mean "measured, and buried".
    snrs = (
        pd.to_numeric(peaks[snr_col], errors="coerce").to_numpy(dtype=float)
        if has_snr
        else None
    )

    if mechanisms is not None:
        mechs = {m: utils.parse_ionization(m) for m in mechanisms}
    else:
        mechs = {a: utils.parse_ionization(adduct_to_mech(a)) for a in (adducts or [])}
    rows: list[dict] = []

    for neutral in formulas:
        for adduct, im in mechs.items():
            try:
                if "^" in neutral:
                    # caret heavy-isotope neutral ('^N' = 15N): pyteomics (used by
                    # combine_formula_and_ionization) cannot parse the bare caret,
                    # so build the ion element counts ourselves. predict_isotopes
                    # DOES accept the caret ion string directly.
                    ion_body = _caret_ion_formula(neutral, im)
                else:
                    ion_body = utils.combine_formula_and_ionization(neutral, im)[:-1]
                pred_mz, pred_int, labels = predict_isotopes(
                    ion_body, im.charge, purity
                )
                # The ion's own line first, whatever order IsoSpec returned it
                # in. Everything below reads index 0 as the ion: the intensity
                # the envelope is normalised to, the line matched before any
                # other, the anchor the score measures every term against, and
                # `is_base`, which the network path derives from the label. For
                # a dibromide the most abundant configuration is the mixed
                # 79/81 line, two mass units above the peak the candidate was
                # proposed for.
                pred_mz, pred_int, labels = anchor_on_monoisotopic(
                    pred_mz, pred_int, labels
                )
                ion = ion_body + ("-" if im.charge < 0 else "+")
            except Exception:
                continue
            if len(pred_mz) == 0:
                continue
            pred_rel = pred_int / pred_int[0]

            obs_mz = np.zeros_like(pred_mz)
            obs_int = np.zeros_like(pred_mz)
            obs_ppm = np.zeros_like(pred_mz)
            obs_int_err = np.zeros_like(pred_mz)
            # Per-line signal-to-noise, NaN where nothing matched or the file
            # carries none: in the score SNR only ever widens a tolerance, so a
            # NaN costs a candidate nothing it had earned.
            obs_snr = np.full(pred_mz.size, np.nan)
            obs_h = np.zeros_like(pred_mz)
            matched_pid: list = [None] * len(pred_mz)
            base_int = None
            cand_ints, resp = ints, iso_response
            band_lo = None

            for i, pmz in enumerate(pred_mz):
                d = pmz * scoring.mz_tolerance_ppm * 1e-6
                lo = np.searchsorted(mzs, pmz - d, "left")
                hi = np.searchsorted(mzs, pmz + d, "right")
                if lo >= hi:
                    continue
                k = lo + int(np.argmin(np.abs(mzs[lo:hi] - pmz)))
                line_snr = float(snrs[k]) if snrs is not None else np.nan
                # Signed, (observed - predicted)/predicted: the targeted
                # matcher's match_mz_error convention, and what the score
                # subtracts the sample's fitted offset from.
                line_ppm = (mzs[k] - pmz) / pmz * 1e6
                if i == 0:  # monoisotopic / base
                    if keep_h and not (np.isfinite(ints[k]) and ints[k] > 0):
                        # no usable area on this M0: the candidate is read by height, unscaled
                        cand_ints, resp = hts, None
                    base_int = cand_ints[k]
                    obs_int[0] = cand_ints[k]
                    obs_h[0] = hts[k]
                    obs_mz[0] = mzs[k]
                    obs_ppm[0] = line_ppm
                    obs_snr[0] = line_snr
                    matched_pid[0] = pids[k]
                    if resp:
                        # the file's own minor-line response (iso_response.py): a C / O line near the
                        # scans' floor reads anywhere from its censored share up to its natural share
                        band_lo = _respond(pred_rel, labels, line_snr, resp)
                        pred_rel = band_lo.copy()
                    continue
                if not base_int:
                    continue
                rel_obs = cand_ints[k] / base_int
                if keep_h and not (np.isfinite(rel_obs) and rel_obs > 0) and obs_h[0] > 0:
                    # no usable area on this line: its height ratio (the lines of one envelope share a width)
                    rel_obs = hts[k] / obs_h[0]
                if band_lo is not None and band_lo[i] < pred_nat_i(pred_int, i):
                    # inside [censored share, natural share]: consistent as read; outside: the nearer end
                    pred_rel[i] = min(max(rel_obs, band_lo[i]), pred_nat_i(pred_int, i))
                ierr = abs(pred_rel[i] - rel_obs) / pred_rel[i]
                # A line whose height is nowhere near its prediction is not this
                # ion's line. What the score then sees is an ABSENT line, which
                # the detectability gate charges or ignores according to what
                # the noise says it would have looked like.
                if ierr <= INTENSITY_TOLERANCE:
                    obs_int[i] = rel_obs * base_int
                    obs_h[i] = hts[k]
                    obs_mz[i] = mzs[k]
                    obs_ppm[i] = line_ppm
                    obs_int_err[i] = ierr
                    obs_snr[i] = line_snr
                    matched_pid[i] = pids[k]

            if base_int is None:  # M0 not detected -> not a candidate at all
                continue
            if centre is not None:
                line_centre = MC.centre_array(centre.a, centre.b, pred_mz,
                                              centre.mz_lo, centre.mz_hi)
            else:
                line_centre = scoring.mu_ppm
            score = float(
                score_pattern_v2(
                    obs_ppm - line_centre,
                    obs_int,
                    obs_snr,
                    pred_rel,
                    sigma_ppm=scoring.sigma_ppm,
                    k_detect=DETECT_SNR_K,
                )
            )
            # the isotope pattern alone: every matched line at the centre
            score_massfree = float(
                score_pattern_v2(
                    np.zeros_like(obs_ppm),
                    obs_int,
                    obs_snr,
                    pred_rel,
                    sigma_ppm=scoring.sigma_ppm,
                    k_detect=DETECT_SNR_K,
                )
            )
            cat = _category(score)

            for i, label in enumerate(labels):
                matched = matched_pid[i] is not None
                rows.append(
                    {
                        "compound_formula": neutral,
                        "compound_score": score,
                        "compound_category": cat,
                        "ion_formula": ion,
                        "ion_score": score,
                        "ion_score_massfree": score_massfree,
                        "ion_category": cat,
                        "mechanism_id": im.mascope_notation,
                        "isotope_formula": ion if label == "M0" else f"[{label}]{ion}",
                        "iso_label": label,
                        "is_base": (i == 0),
                        "theo_mz": float(pred_mz[i]),
                        "rel_abundance": float(pred_rel[i]),
                        "iso_score": score if matched else None,
                        "iso_category": cat if matched else None,
                        "sample_peak_id": matched_pid[i],
                        "sample_peak_mz": float(obs_mz[i]) if matched else None,
                        "sample_peak_intensity": float(obs_h[i]) if matched else None,
                        "ppm_error": float(obs_ppm[i]) if matched else None,
                        "abundance_error": float(obs_int_err[i])
                        if matched and i > 0
                        else None,
                    }
                )

    return pd.DataFrame(rows)
