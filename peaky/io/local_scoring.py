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

import re

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


def adduct_to_mech(adduct: str) -> str:
    """peaky adduct label -> mascope_tools ionization-mechanism string.
    '[M+Br]-' -> '+Br-' ; '[M-H]-' -> '-H-' ; '[M+(CH4N2O)H]+' -> '+(CH4N2O)H+' ;
    '[M+NH4]+' -> '+NH4+' ; '[M+^NO3]-' -> '+^NO3-' (15N nitrate).

    Multi-part adducts (more than one +/- term) are collapsed by concatenating the
    added pieces, e.g. '[M+HBr+Br]-' -> '+HBrBr-' (= +HBr2) and '[M+HBr+CO3]-' ->
    '+HBrCO3-'. parse_ionization/parse_composition then sum the atoms."""
    m = re.match(r"^\[M(.+)\]([+-])$", adduct.strip())
    if not m:
        raise ValueError(f"unrecognised adduct label {adduct!r}")
    core, charge = m.group(1), m.group(2)
    terms = re.findall(r"[+-][^+-]+", core)  # ['+HBr', '+CO3'] | ['-H'] | ['+Br']
    adds = "".join(t[1:] for t in terms if t.startswith("+"))
    subs = "".join(t[1:] for t in terms if t.startswith("-"))
    if adds and not subs:
        return f"+{adds}{charge}"
    if subs and not adds:
        return f"-{subs}{charge}"
    raise ValueError(f"mixed +/- adduct not supported: {adduct!r}")


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
    purity: float | None = None,
    mz_col: str = "mz",
    intensity_col: str = "height",
    peak_id_col: str = "peak_id",
    snr_col: str = "signal_to_noise",
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

    scoring = scoring if scoring is not None else PatternScoring()
    has_snr = snr_col in peaks.columns
    columns = [mz_col, intensity_col, peak_id_col] + ([snr_col] if has_snr else [])
    peaks = (
        peaks[columns]
        .dropna(subset=[mz_col])
        .drop_duplicates(peak_id_col)  # raw server peaks have one row per match
        .sort_values(mz_col)
    )
    mzs = peaks[mz_col].to_numpy(dtype=float)
    ints = peaks[intensity_col].to_numpy(dtype=float)
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
            matched_pid: list = [None] * len(pred_mz)
            base_int = None

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
                    base_int = ints[k]
                    obs_int[0] = ints[k]
                    obs_mz[0] = mzs[k]
                    obs_ppm[0] = line_ppm
                    obs_snr[0] = line_snr
                    matched_pid[0] = pids[k]
                    continue
                if not base_int:
                    continue
                rel_obs = ints[k] / base_int
                ierr = abs(pred_rel[i] - rel_obs) / pred_rel[i]
                # A line whose height is nowhere near its prediction is not this
                # ion's line. What the score then sees is an ABSENT line, which
                # the detectability gate charges or ignores according to what
                # the noise says it would have looked like.
                if ierr <= INTENSITY_TOLERANCE:
                    obs_int[i] = ints[k]
                    obs_mz[i] = mzs[k]
                    obs_ppm[i] = line_ppm
                    obs_int_err[i] = ierr
                    obs_snr[i] = line_snr
                    matched_pid[i] = pids[k]

            if base_int is None:  # M0 not detected -> not a candidate at all
                continue
            score = float(
                score_pattern_v2(
                    obs_ppm - scoring.mu_ppm,
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
                        "sample_peak_intensity": float(obs_int[i]) if matched else None,
                        "ppm_error": float(obs_ppm[i]) if matched else None,
                        "abundance_error": float(obs_int_err[i])
                        if matched and i > 0
                        else None,
                    }
                )

    return pd.DataFrame(rows)
