"""Episode evidence in the merge: a short bright episode's readings judged on
the episode's own spectra (docs/MERGE.md step 6g), and the persistent ions'
clusters with a plume solvent (step 1c).

A reading seen in one to three files of a batch has no cross-file vote behind
it. When those files are an episode's (the episode stage of sampling.py picked
them and flagged their rows `episode`), the episode's own spectra can still
decide the ion -- by the isotope lines and adduct partners the reading
predicts IN THOSE SPECTRA, never by what else rises with it.

**The flag on the merged row.** `assign_batch.align` carries the per-file
`episode` flag onto the merged row (any file's row of the bin flagged ->
flagged; `_M0_COLS`); a batch whose per-file ledgers carry no such column
(no episode file) has none, and nothing here runs.

**Part B -- one episode's evidence (`judge`).** For a merged M0 reading of an
episodic bin with `n_files <= MAX_FILES`, the episode files E = the files
whose row of the reading's ion is flagged. On the rows of E's samples in the
batch series (`per_file/_batch_ts.parquet`) it reads:

(a) the mass: each E file's reading error inside that file's calibrated
    window (|ppm_error - mu| <= K_SIGMA x sigma, the file's degeneracy
    calibration; a file with none has no window and (a) cannot hold);
(b) the isotopes: the reading's own 13C line at a consistent carbon count
    (|c - nC| <= max(C_TOL_ABS, C_TOL_REL x nC), the levels' band) in EVERY E
    file where its predicted S/N reaches MIN_SNR, at least one such file; and
    each S / Cl / Br the ion carries confirmed by its own 34S / 37Cl / 81Br
    line (LINE_BAND x its prediction) in every file where that line is
    testable, at least one -- an untested heteroatom is no evidence (the tier
    engine's own rule);
(c) the rivals: no exact-alias rival left that these isotope facts cannot
    exclude -- the four 15N-label alias classes (Cl = C2H-4 ^N, C2F3 =
    H2 ^N O4, C8S = H ^N O7, C O H = N ^N: the [M+CO3]- / [M+^NO3]- pair), in
    both directions, on a batch whose reagent carries the label; and the
    exotic rivals within ALIAS_MDA of the ion that the files' engine and the
    batch levels left open (the per-file `alternatives`, the pooled pair's
    `competitors_left`). A rival is excluded by its carbon count read off the
    13C line (outside the band of ITS count in every testable file), by an
    S / Cl / Br / Si line (29Si / 30Si) that it predicts and the spectra do not
    show, or that the reading predicts and they show, or by the label atoms'
    14N twin (the reagent's 14N share per label atom, one 15N-14N step below)
    where the two carry different label counts and the twin is testable;
(d) no hold: neither the low-file isotopologue hold nor the satellite-artefact
    hold (batch/sideband.py), nor the isotope-contradiction guard
    (assignment/isoguard.py) holds the row;
(e) a second fact of the same ion: an adduct partner of the same neutral (the
    reagent profile's other ion forms) in the same spectra, at >= PAIR_X x the
    file's edge, OR a second episode file reading the same ion.

Co-occurrence with the episode's other lines is never evidence. Cross-mode
partners in the same minutes (another reagent's batch) are not available
inside one batch run and are not read.

All five hold -> the reading is Assigned from that episode (tier Assigned, the
note says which facts; a Candidate is promoted). Otherwise it is not Assigned
FROM ITS EPISODE: a Candidate stays Candidate, and a reading the vote made
Assigned (MODE, `--episode-evidence`) -- "measured", the default -- goes to
Candidate only on a contradicting MEASUREMENT, never on a missing one: its mass
outside an existing calibrated window ((a); no window is untestable), an
isotope line read against it ((b)), a rival a measured isotope fact favours
((c); a rival left with untestable isotopes is untestable), a hold ((d)); (e)
never demotes (a missing adduct pair or second file is no measurement). The
row's `measurement` column names what contradicted it. "promote-only" keeps
every such row; "strict" demotes on any failed clause. A curated neutral
(`curated_exempt`: the pass-0 registry + the active reference lists' formulas)
keeps the vote's tier in every mode. When
no isotope
line was testable the claim is `single episode, isotopes untestable`
(UNTESTABLE_CLAIM; the evidence level is the level machinery's, untouched).
Promotion is open only to readings whose ion carries no element the isotopes
cannot speak for (PROMOTABLE: C H N O S Cl Br; F, Si, P, I never), and never
to an ion-only reading.

**Part C -- a persistent ion's cluster with a plume solvent
(`read_solvent_clusters`).** During an industrial-solvent episode
(solvent_clusters.episode_solvents: NMP, DMSO) every persistent ion can gain a
cluster line at its own m/z + the solvent's neutral mass. A merged row there
whose presence is confined to the solvent's episode (>= CONFINED_FRAC of its
spectra, in at most CLUSTER_MAX_FRAC of the batch's) -- the spectra where one of
the solvent's own ion forms (its reagent-profile ion forms and its proton-bound
dimer) stands at >= SOLVENT_X x the edge and, for a form present in most
spectra, >= SOLVENT_LIFT x its median (a solvent background everywhere is fine;
its episodes are what count) -- beside a known PERSISTENT Assigned ion (in >=
PERSIST_FRAC of the batch's spectra), and that shares at least TRACK_MIN_N of the
episode's spectra with the parent and the solvent ion (fewer: nothing is read),
is read as that ion's cluster: the parent's neutral, the adduct naming the
solvent (`[M+(C5H9NO)H]+`), not a new neutral. That reading rests on the
composition (the line's ion = a persistent Assigned ion + the solvent) and on
co-occurrence with the solvent; so its tier is capped at Candidate with the note
"solvent cluster of <X>: co-occurs with the solvent" unless the track is
SPECIFIC (`pair_test`): r(cluster, parent x solvent) >= TRACK_R and >= the
solvent alone's r + SPECIFIC_GAIN, with the parent varying over the episode
(CV >= PARENT_CV_MIN), on >= SPECIFIC_MIN_N spectra -- then the parent's tier.
A merged reading of ANOTHER ion composition at that mass is re-read as the
cluster only on the specific test. The reagent-cluster strip of the merge spares
a row read so.

Pure: reads the merged ledger, the per-file ledgers and the batch series;
mutates only the merged rows it judges or relabels; returns a table and a
summary each.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from peaky.assignment import isoguard as IG
from peaky.assignment import solvent_clusters as SC
from peaky.batch import iso_checks as IC
from peaky.batch import sideband as SB
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

__version__ = "0.2.0"   # the three modes; measured = demote only on a contradicting measurement; the
#                        specific plume-solvent track; one curated-exemption source (curated_exempt)
#                        0.1.0 Parts B and C

# --------------------------------------------------------------------------- Part B constants
#: the most files a reading may sit in and still be judged on its episode alone
MAX_FILES = 3
#: the calibrated half-window, in the file's sigma (levels.context.K_SIGMA)
K_SIGMA = 3.0
#: an isotope line or an adduct partner is read within this many ppm of its spacing
#: from the row's own line in that spectrum (rule C's window)
LINE_PPM = IC.C_TOL_PPM
#: a line is testable where its predicted S/N reaches this (rule C's C_MIN_SNR13,
#: the levels' DET_SNR): the row's server S/N x the line's share, else its height
#: over the file's noise edge
MIN_SNR = IC.C_MIN_SNR13
#: the carbon-count band (the levels' test (c), levels.lines.C_TOL_*): |c - nC| <= max(abs, rel x nC)
C_TOL_ABS, C_TOL_REL = IC.C_TOL_ABS, IC.C_TOL_REL
#: a heteroatom line confirms its count at this multiple of its prediction (satellites.OWN_LINE_RATIO)
LINE_BAND = (0.5, 2.0)
#: the heteroatom lines (b) reads: element -> (spacing Da, per-atom ratio, label)
HETERO_LINES = {"S": (ISO.D_34S, ISO.R_34S_PER_S, "34S"), "Cl": (ISO.D_37CL, ISO.R_37CL_PER_CL, "37Cl"),
                "Br": (ISO.D_81BR, ISO.R_81BR_PER_BR, "81Br")}
#: ... and the lines a rival's silicon predicts (rival exclusion only)
SI_LINES = ((ISO.D_29SI, ISO.R_29SI_PER_SI, "29Si"), (ISO.D_30SI, ISO.R_30SI_PER_SI, "30Si"))
#: the 15N - 14N step: a label atom's 14N twin sits this far below the line
D_TWIN = C.M["^N"] - C.M["N"]
#: an adduct partner counts at this multiple of the file's noise edge
PAIR_X = 3.0
#: an exotic rival is an exact alias within this many mDa of the reading's ion
ALIAS_MDA = 0.5
#: the elements a promoted reading's ion may carry (isotope facts speak for each)
PROMOTABLE = frozenset({"C", "H", "N", "^N", "O", "S", "Cl", "Br"})
#: the four 15N-label alias classes, as ion-composition swaps (applied with both signs)
ALIAS_SWAPS = (
    ("Cl = C2H-4 ^N", {"Cl": -1, "C": 2, "H": -4, "^N": 1}),
    ("C2F3 = H2 ^N O4", {"C": -2, "F": -3, "H": 2, "^N": 1, "O": 4}),
    ("C8S = H ^N O7", {"C": -8, "S": -1, "H": 1, "^N": 1, "O": 7}),
    ("CO3 = ^NO3 (C O H = N ^N)", {"C": -1, "O": -1, "H": -1, "N": 1, "^N": 1}),
)
#: the ion forms a 15N-label alias rival may take beside the reagent profile's own: the
#: carbonate radical anion's cluster and the radical anion itself (the [M+CO3]- / [M+^NO3]- pair)
ALIAS_EXTRA_ADDUCTS = ("[M+CO3]-", "[M]-.")
#: what a failing reading the vote made Assigned becomes (`--episode-evidence`): "measured" (the
#: default) -- Candidate only on a contradicting MEASUREMENT, never on a missing one: (a) its mass
#: outside an EXISTING calibrated window; (b) an isotope line read against it; (c) a rival that a
#: measured isotope fact favours (`_favours`); (d) a hold; (e) never ((e) is a promotion criterion only);
#: "promote-only" -- always kept with the note; "strict" -- Candidate whenever a clause fails (Assigned
#: from one episode only when all hold). A curated neutral (`curated_exempt`) keeps the vote's tier in
#: every mode; a Candidate is promoted in every mode.
MODES = ("measured", "promote-only", "strict")
MODE = "measured"
NOTE_MARK = "episode evidence"
UNTESTABLE_CLAIM = "single episode, isotopes untestable"
ASSIGNED_CLAIM = "single episode, isotope facts hold"
TABLE_COLUMNS = ["mz", "neutral_formula", "adduct", "tier_in", "tier_out", "verdict", "episode_claim",
                 "n_files", "episode_files", "a_mass", "b_isotopes", "b_detail", "c_rivals", "c_detail",
                 "d_hold", "e_second", "e_detail", "note", "measured_failure", "mode", "prior_reason", "measurement"]

# --------------------------------------------------------------------------- Part C constants
#: the solvent's own ion is bright in a spectrum at this multiple of its noise edge
SOLVENT_X = 5.0
#: a line present in most spectra (a background) is ELEVATED only at >= this multiple of its median
#: height there (the episode stage's lift): the solvent's episode and the cluster's spectra are read so
SOLVENT_LIFT = 5.0
#: a cluster line sits within this many ppm of parent + solvent
CLUSTER_PPM = 3.0
#: a parent is persistent when its line is in at least this share of the batch's spectra
PERSIST_FRAC = 0.5
#: a cluster's presence is confined to the episode: at least this share of its spectra
#: hold the solvent ion bright, and it is in at most CLUSTER_MAX_FRAC of the batch's spectra
CONFINED_FRAC = 0.9
CLUSTER_MAX_FRAC = 0.5
#: the pair test: the cluster tracks parent x solvent (Pearson r) on at least this many spectra
TRACK_R = 0.7
TRACK_MIN_N = 4
#: the pair test is SPECIFIC -- the parent, not the solvent episode alone, explains the line -- only
#: where the product beats the solvent alone by this much in r, the parent varies over the episode's
#: spectra (coefficient of variation >= PARENT_CV_MIN: a flat persistent parent makes parent x solvent
#: the solvent's own profile, and any line confined to the episode then "tracks" it), and on at least
#: SPECIFIC_MIN_N spectra. Measured on two uronium batches of a field campaign: with the plain track
#: (r >= 0.7) 94-97 % of RANDOM persistent parents passed beside an episode-confined line
SPECIFIC_GAIN = 0.2
PARENT_CV_MIN = 0.3
SPECIFIC_MIN_N = 6
CO_OCCURS = "co-occurs with the solvent"
CLUSTER_MARK = "solvent cluster"
CLUSTER_COLUMNS = ["mz", "was_neutral", "was_adduct", "neutral_formula", "adduct", "solvent", "parent_mz",
                   "parent_reading", "parent_tier", "tier_in", "tier_out", "n_spectra", "n_in_episode",
                   "n_track", "track_r", "r_solvent", "cv_parent", "pair_test"]


def curated_exempt(context, reflists_active=()) -> frozenset:
    """The curated neutrals no failed episode clause ever demotes: the pass-0 registry of the
    context's polarity plus every formula of the run's active reference lists
    (`passes.curated_formulas(polarity, context, reflists.prior_formulas(lists))`) -- the exempt
    set the merge's removal gates use too. ONE source: the batch and an offline re-judge call it."""
    from peaky.assignment import passes as PS
    from peaky.assignment import reflists as RL
    from peaky.chem import contexts as CX
    try:
        pol = CX.get_context(context).polarity if context else "negative"
    except Exception:  # noqa: BLE001 -- an unknown context: the registry's default
        pol = "negative"
    return frozenset(PS.curated_formulas(pol, context, RL.prior_formulas(reflists_active) if reflists_active else ()))


def curated_exempt_from_summary(summ: dict, catalog: dict | None = None) -> frozenset:
    """`curated_exempt` rebuilt from a finished run's batch_summary.json: its context and its active
    reference lists (`reflists_context.active`, [id, version, how]) read from the catalog
    (`reflists.load_catalog`); a list whose version differs from the recorded one raises (the set
    would not be the run's)."""
    from peaky.assignment import reflists as RL
    act = ((summ or {}).get("reflists_context") or {}).get("active") or []
    cat = catalog if catalog is not None else (RL.load_catalog() if act else {})
    lists = []
    for lid, ver, *_how in act:
        L = cat.get(lid)
        if L is None or str(L.data_version) != str(ver):
            raise ValueError(f"reference list {lid} {ver} of the run is not in the catalog at that version")
        lists.append(L)
    return curated_exempt((summ or {}).get("context"), lists)


def empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


def empty_clusters() -> pd.DataFrame:
    return pd.DataFrame(columns=CLUSTER_COLUMNS)


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    return "" if s in ("nan", "<NA>", "None") else s


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def truthy(v) -> bool:
    return _s(v).strip().lower() in ("1", "true", "yes")


def _counts(neutral, adduct) -> dict:
    """The ion's element counts, the label atom as '^N' (levels.space.ion_counts_of); {} unreadable."""
    from peaky.assignment.levels import space as SP
    try:
        return {k: int(v) for k, v in (SP.ion_counts_of(_s(neutral), _s(adduct)) or {}).items() if v}
    except Exception:  # noqa: BLE001 -- an unreadable reading carries no element
        return {}


def _ion_mass(counts: dict, sign: str) -> float:
    """m/z of a singly charged ion of `counts` ('^N' the label atom)."""
    m = sum(C.M[el] * n for el, n in counts.items())
    return m - C.M_E if sign == "+" else m + C.M_E


def _adduct_delta(adduct: str) -> dict | None:
    """What an ion form adds to its neutral, as element counts ('^N' kept)."""
    base = _counts("C", adduct)
    if not base:
        return None
    d = dict(base)
    d["C"] = d.get("C", 0) - 1
    return {k: v for k, v in d.items() if v}


class _Spectrum:
    """One sample's peaks of the batch series, sorted by m/z, and its noise edge."""

    def __init__(self, sub: pd.DataFrame, edge: float):
        sub = sub.sort_values("mz", kind="mergesort")
        self.mz = sub["mz"].to_numpy(dtype=float)
        self.h = pd.to_numeric(sub["height"], errors="coerce").to_numpy(dtype=float)
        self.snr = (pd.to_numeric(sub["signal_to_noise"], errors="coerce").to_numpy(dtype=float)
                    if "signal_to_noise" in sub.columns else np.full(len(sub), np.nan))
        self.pid = sub["peak_id"].astype(str).to_numpy() if "peak_id" in sub.columns else None
        self.edge = float(edge) if edge is not None and np.isfinite(_num(edge)) and _num(edge) > 0 else np.nan

    def near(self, m: float, ppm: float = LINE_PPM):
        """(height, snr, m/z) of the tallest peak within `ppm` of m/z `m`, or (0, nan, nan)."""
        w = m * ppm * 1e-6
        lo, hi = np.searchsorted(self.mz, m - w, "left"), np.searchsorted(self.mz, m + w, "right")
        if hi <= lo:
            return 0.0, float("nan"), float("nan")
        j = lo + int(np.nanargmax(self.h[lo:hi]))
        return float(self.h[j]), float(self.snr[j]), float(self.mz[j])

    def by_id(self, pid: str):
        if self.pid is None or not pid:
            return None
        k = np.flatnonzero(self.pid == pid)
        if not len(k):
            return None
        j = int(k[0])
        return float(self.h[j]), float(self.snr[j]), float(self.mz[j])


def _pred_snr(h0: float, snr0: float, share: float, edge: float) -> float:
    """The predicted S/N of a line at `share` x a parent line of height h0 (server S/N snr0)."""
    if np.isfinite(snr0) and snr0 > 0:
        return snr0 * share
    if np.isfinite(edge) and edge > 0:
        return h0 * share / edge
    return float("nan")


def _band(nc: int) -> float:
    return max(C_TOL_ABS, C_TOL_REL * nc)


def _file_facts(sp: _Spectrum, row, counts: dict, sign: str) -> dict:
    """One episode file's line facts for the reading: its line (h0, snr0, mz0), the
    13C count (testable, c), and every heteroatom / Si line's observed ratio and testability."""
    got = sp.by_id(_s(row.get("peak_id")))
    if got is None or not got[0] > 0:
        got = sp.near(_num(row.get("mz")))
    h0, snr0, mz0 = got
    out = {"h0": h0, "snr0": snr0, "mz0": mz0, "edge": sp.edge, "lines": {}}
    if not (h0 > 0 and np.isfinite(mz0)):
        out["c13"] = (False, float("nan"))
        return out
    nc = int(counts.get("C", 0))
    share13 = IC.R13C * max(nc, 1)
    test13 = bool(nc >= 1 and _pred_snr(h0, snr0, share13, sp.edge) >= MIN_SNR)
    h13, _sn, _m = sp.near(mz0 + IC.D13C)
    c = (h13 / h0 - counts.get("O", 0) * IC.R17O) / IC.R13C
    out["c13"] = (test13, float(c))
    out["c13_ratio"] = h13 / h0
    for el, (d, r, lab) in list(HETERO_LINES.items()) + [("Si29", SI_LINES[0]), ("Si30", SI_LINES[1])]:
        hh, _sn, _m = sp.near(mz0 + d)
        out["lines"][el] = hh / h0
    hh, _sn, _m = sp.near(mz0 - D_TWIN)            # the 14N twin of a 15N label atom
    out["lines"]["twin"] = hh / h0
    return out


def _line_state(facts: list[dict], el: str, n: int, n_ref: int = 0) -> tuple[int, int]:
    """(testable files, consistent files) of element el's line for a count n (the files'
    observed ratio inside LINE_BAND x n's prediction; n = 0: the line absent, read
    against n_ref's prediction for testability)."""
    key, (d, r, _lab) = ((el, HETERO_LINES[el]) if el in HETERO_LINES
                         else (el, dict(Si29=SI_LINES[0], Si30=SI_LINES[1])[el]))
    tested = ok = 0
    for f in facts:
        if not f["h0"] > 0:
            continue
        pred = r * max(n, n_ref)
        if not (pred > 0 and _pred_snr(f["h0"], f["snr0"], pred, f["edge"]) >= MIN_SNR):
            continue
        tested += 1
        obs = f["lines"].get(key, 0.0)
        if n > 0:
            ok += int(LINE_BAND[0] * r * n <= obs <= LINE_BAND[1] * r * n)
        else:
            ok += int(obs < LINE_BAND[0] * r * n_ref)
    return tested, ok


def _isotopes(facts: list[dict], counts: dict) -> tuple[str, str]:
    """(b): 'holds' / 'failed' / 'untestable', and the detail text."""
    nc = int(counts.get("C", 0))
    det, failed, untested = [], False, False
    t13 = [f for f in facts if f["c13"][0]]
    if nc < 1:
        untested = True
        det.append("no carbon")
    elif not t13:
        untested = True
        det.append(f"13C untestable in {len(facts)} file(s)")
    else:
        bad = [f for f in t13 if abs(f["c13"][1] - nc) > _band(nc)]
        cs = ", ".join(f"{f['c13'][1]:.1f}" for f in t13)
        det.append(f"13C reads {cs} C for {nc} (band {_band(nc):g}) in {len(t13)} file(s)")
        failed |= bool(bad)
    for el in ("S", "Cl", "Br"):
        n = int(counts.get(el, 0))
        if n < 1:
            continue
        tested, ok = _line_state(facts, el, n)
        lab = HETERO_LINES[el][2]
        if not tested:
            untested = True
            det.append(f"{lab} untestable")
        else:
            det.append(f"{lab} at its count in {ok}/{tested} file(s)")
            failed |= ok < tested
    return ("failed" if failed else "untestable" if untested else "holds"), "; ".join(det)


def _excluded(facts: list[dict], counts: dict, rival: dict) -> str | None:
    """Why the isotope facts exclude a rival ion (counts `rival`), or None."""
    nc, nr = int(counts.get("C", 0)), int(rival.get("C", 0))
    t13 = [f for f in facts if f["c13"][0]]
    if nr != nc and t13 and all(abs(f["c13"][1] - nr) > _band(nr) for f in t13):
        cs = ", ".join("%.1f" % f["c13"][1] for f in t13)
        return f"13C reads {cs} C, not {nr}"
    for el in ("S", "Cl", "Br"):
        n_s, n_r = int(counts.get(el, 0)), int(rival.get(el, 0))
        if n_s == n_r:
            continue
        tested, ok = _line_state(facts, el, n_r, n_ref=n_s)
        if tested and ok == 0:
            return f"its {HETERO_LINES[el][2]} line reads against {el}{n_r}"
    n_l, n_rl = int(counts.get("^N", 0)), int(rival.get("^N", 0))
    if n_l != n_rl:
        # the label atoms' 14N twin (the reagent's 14N share f per label atom, one 15N-14N step below)
        f = 1.0 - ISO.label_purity()
        tested = ok = 0
        for x in facts:
            pred = f * max(n_l, n_rl)
            if not (x["h0"] > 0 and pred > 0 and _pred_snr(x["h0"], x["snr0"], pred, x["edge"]) >= MIN_SNR):
                continue
            tested += 1
            obs = x["lines"].get("twin", 0.0)
            ok += int(LINE_BAND[0] * f * n_rl <= obs <= LINE_BAND[1] * f * n_rl) if n_rl else int(
                obs < LINE_BAND[0] * f * n_l)
        if tested and ok == 0:
            return f"its 14N twin line reads against ^N{n_rl}"
    n_si, n_rsi = int(counts.get("Si", 0)), int(rival.get("Si", 0))
    if n_rsi > n_si:
        for key in ("Si30", "Si29"):
            tested, ok = _line_state(facts, key, 0, n_ref=n_rsi)
            if tested and ok == tested:
                return f"no {key[2:]}Si line where Si{n_rsi} predicts one"
    return None


def _favours(facts: list[dict], counts: dict, rival: dict) -> str | None:
    """Why a MEASURED isotope fact favours a rival ion (counts `rival`) over the reading, or None:
    the 13C count reads nearer the rival's carbon count in every testable file (inside the rival's
    band); an S / Cl / Br / Si line, or the label's 14N twin, that the rival predicts and the reading
    does not (or at another count) sits at the rival's share in every file where it is testable."""
    nc, nr = int(counts.get("C", 0)), int(rival.get("C", 0))
    t13 = [f for f in facts if f["c13"][0]]
    if nr != nc and t13 and all(abs(f["c13"][1] - nr) < abs(f["c13"][1] - nc) and abs(f["c13"][1] - nr) <= _band(nr)
                               for f in t13):
        cs = ", ".join("%.1f" % f["c13"][1] for f in t13)
        return f"13C reads {cs} C: nearer its {nr} than {nc}"
    for el in ("S", "Cl", "Br"):
        n_s, n_r = int(counts.get(el, 0)), int(rival.get(el, 0))
        if n_r > n_s:
            tested, ok = _line_state(facts, el, n_r, n_ref=n_s)
            if tested and ok == tested:
                return f"its {HETERO_LINES[el][2]} line reads {el}{n_r}"
    n_si, n_rsi = int(counts.get("Si", 0)), int(rival.get("Si", 0))
    if n_rsi > n_si:
        tested, ok = _line_state(facts, "Si30", n_rsi)
        if tested and ok == tested:
            return f"its 30Si line reads Si{n_rsi}"
    n_l, n_rl = int(counts.get("^N", 0)), int(rival.get("^N", 0))
    if n_rl > n_l:
        f14 = 1.0 - ISO.label_purity()
        tested = ok = 0
        for x in facts:
            pred = f14 * n_rl
            if not (x["h0"] > 0 and pred > 0 and _pred_snr(x["h0"], x["snr0"], pred, x["edge"]) >= MIN_SNR):
                continue
            tested += 1
            ok += int(LINE_BAND[0] * pred <= x["lines"].get("twin", 0.0) <= LINE_BAND[1] * pred)
        if tested and ok == tested:
            return f"its 14N twin reads ^N{n_rl}"
    return None


def _alias_rivals(counts: dict, sign: str, adducts, labelled: bool) -> list[tuple[str, dict]]:
    """The reading's 15N-label alias rivals: (class, rival ion counts) -- an ion one of
    the ALIAS_SWAPS makes (either sign) that some reagent ion form reads as a closed-shell
    neutral without the label atom."""
    if not labelled:
        return []
    out = []
    adducts = list(adducts) + [a for a in ALIAS_EXTRA_ADDUCTS if a not in adducts]
    for name, swap in ALIAS_SWAPS:
        for sgn in (1, -1):
            r = dict(counts)
            for el, d in swap.items():
                r[el] = r.get(el, 0) + sgn * d
            if any(v < 0 for v in r.values()):
                continue
            r = {k: v for k, v in r.items() if v}
            for a in adducts:
                if (a.endswith("+") and sign != "+") or (a.endswith("-") and sign != "-"):
                    continue
                d = _adduct_delta(a)
                if d is None:
                    continue
                y = dict(r)
                for el, k in d.items():
                    y[el] = y.get(el, 0) - k
                if any(v < 0 for v in y.values()) or y.get("^N", 0):
                    continue
                y = {k: v for k, v in y.items() if v}
                if y and C.dbe_ok(y)[0] and C.oxygen_ok(y)[0]:
                    out.append((f"{name}: {C.format_formula(y)} {a}", r))
                    break
    return out


def _named_rivals(rows: list, level_row, mz_ion: float) -> list[tuple[str, dict]]:
    """The exotic rivals within ALIAS_MDA of the ion that the files' engine (`alternatives`) and
    the pooled levels (`competitors_left`) left open: (reading, ion counts)."""
    seen, out = set(), []
    texts = []
    for r in rows:
        alts = _s(r.get("alternatives"))
        if alts.startswith("["):
            try:
                for it in json.loads(alts):
                    if isinstance(it, dict) and it.get("formula") and it.get("adduct"):
                        texts.append((str(it["formula"]), str(it["adduct"])))
            except (ValueError, TypeError):
                pass
    if level_row is not None:
        left = _s(level_row.get("competitors_left")).split(" || ")[0]
        for it in left.split(";"):
            p = it.strip().split()
            if len(p) == 2:
                texts.append((p[0], p[1]))
    for nf, ad in texts:
        if (nf, ad) in seen:
            continue
        seen.add((nf, ad))
        try:
            m = C.ion_mz(nf, ad)
        except Exception:  # noqa: BLE001
            continue
        if abs(m - mz_ion) * 1e3 <= ALIAS_MDA:
            cnt = _counts(nf, ad)
            if cnt:
                out.append((f"{nf} {ad}", cnt))
    return out


def _partner(facts_rows, nf: str, ad: str, adducts, sign: str) -> str | None:
    """(e): an adduct partner of the same neutral in an episode spectrum: the text, or None."""
    try:
        m_self = C.ion_mz(nf, ad)
    except Exception:  # noqa: BLE001
        return None
    for a in adducts:
        if a == ad or (a.endswith("+") and sign != "+") or (a.endswith("-") and sign != "-"):
            continue
        if not _counts(nf, a):
            continue
        try:
            dm = C.ion_mz(nf, a) - m_self
        except Exception:  # noqa: BLE001
            continue
        if abs(dm) < 0.01:
            continue
        for sid, sp, f in facts_rows:
            if not f["h0"] > 0:
                continue
            h, _sn, m = sp.near(f["mz0"] + dm)
            if h > 0 and np.isfinite(sp.edge) and h >= PAIR_X * sp.edge:
                return f"{nf} {a} at {m:.4f} in {sid} ({h / sp.edge:.0f}x the edge)"
    return None


def _sideband(full: dict, efiles: list, ts_sorted, counts: dict, sigma: dict, orbitrap: bool) -> str | None:
    """The low-file hold's two clauses (batch/sideband.py) read on the episode rows: the
    isotopologue line of a brighter Assigned row in every episode file, or (Orbitrap-class)
    a satellite partner in every file that the per-spectrum test calls the row's artefact.
    A row with its own 13C line in EVERY episode file is an M0 of its own (the hold's
    own rule, `hold_low_file_promotions`). The text of the hold that applies, or None."""
    per, own = [], []
    for sid, r in efiles:
        f = full.get(sid)
        if f is None:
            return None
        sg = sigma.get(sid)
        tol = SB.ISO_PPM + (max(0.0, sg) if sg is not None and np.isfinite(sg) else 0.0)
        own.append(bool(SB._own_13c(f, r, counts, tol)))
        sat = (SB._satellite(f, r, ts_sorted, []) or SB._shoulder(f, r)) if orbitrap else None
        per.append((r, SB._isotopologue(f, r, tol), sat))
    if own and all(own):
        return None                            # an M0 of its own in every episode file
    if per and all(iso is not None for _r, iso, _sat in per):
        x = per[0][1]
        return f"{SB.HOLD_MARK} ({x['rule']} of {x['partner_mz']:.4f} {x['partner_reading']})"
    sats = [sat for _r, _iso, sat in per if sat is not None]
    if per and sats and all(iso is not None or sat is not None for _r, iso, sat in per):
        top = max(sats, key=lambda x: _num(x["partner_height"]))
        test = SB.series_satellite_test(ts_sorted, _num(per[0][0].get("mz")), top["partner_mz"], counts.get("C", 0))
        if test is not None and test["artefact"]:
            return f"{SB.SATELLITE_HOLD_MARK} (of {top['partner_mz']:.4f}: {test['why']})"
    return None


def _note(merged, i, msg):
    prev = _s(merged.at[i, "tier_reason"])
    merged.at[i, "tier_reason"] = f"{prev} | {msg}" if prev else msg


def judge(merged: pd.DataFrame, ledgers: dict, ts: pd.DataFrame | None, *, windows: dict | None = None,
          edges: dict | None = None, adducts=(), labelled: bool = False, levels: pd.DataFrame | None = None,
          tol_ppm: float = 6.0, exempt=frozenset(), mode: str | None = None, orbitrap: bool = True,
          max_files: int = MAX_FILES, log=print) -> tuple[pd.DataFrame, dict]:
    """Part B (module note) on every episodic merged reading with n_files <= `max_files`.

    merged: the merged ledger (mz, neutral_formula, adduct, tier, srcs, n_files,
    episode, tier_reason; mutated: tier, tier_reason, `episode_claim`). ledgers:
    {sample id: its per-file ledger, every peak}. ts: the batch series
    (sample_item_id, peak_id, mz, height[, signal_to_noise]). windows: {sample id:
    (mu, sigma) ppm}, the files' calibrated windows. edges: {sample id: noise edge
    cps}. adducts: the reagent profile's ion forms. labelled: the reagent carries
    the 15N label. levels: the pooled pairs (competitors_left). exempt: curated
    neutrals a failure never demotes. mode: MODES (None: MODE). Returns (table, summary)."""
    mode = MODE if mode is None else str(mode)
    if mode not in MODES:
        raise ValueError(f"episode evidence mode {mode!r}: one of {MODES}")
    out = {"judged": 0, "assigned": 0, "promoted": 0, "demoted": 0, "kept_assigned": 0, "untestable": 0,
           "candidate": 0, "mode": MODE if mode is None else str(mode)}
    if merged is None or not len(merged) or "episode" not in merged.columns:
        return empty_table(), out
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    windows, edges = windows or {}, edges or {}
    sel = merged["episode"].map(truthy)
    if "n_files" in merged.columns:
        sel &= pd.to_numeric(merged["n_files"], errors="coerce").fillna(99) <= max_files
    if "solvent_cluster_of" in merged.columns:
        sel &= merged["solvent_cluster_of"].map(_s) == ""
    idx = merged.index[sel]
    if "episode_claim" not in merged.columns:
        merged["episode_claim"] = pd.NA
    if not len(idx):
        return empty_table(), out
    lvl = {}
    if levels is not None and len(levels) and {"neutral_formula", "adduct"} <= set(levels.columns):
        lvl = {(str(n), str(a)): r for n, a, r in zip(levels["neutral_formula"], levels["adduct"],
                                                      levels.to_dict("records"))}
    pooled_facts = IG.pair_facts(levels) if levels is not None and len(levels) else {}
    sigma = {k: (_num(w[1]) if w else float("nan")) for k, w in windows.items()}
    full, frames = {}, {}
    ts_sorted = None
    if ts is not None and len(ts) and {"sample_item_id", "mz", "height"} <= set(ts.columns):
        ts_sorted = ts[["sample_item_id", "mz", "height"]].dropna(subset=["mz"]).sort_values("mz", kind="mergesort")
    for sid, led in (ledgers or {}).items():
        if led is not None and len(led) and {"mz", "height", "role", "tier", "peak_id"} <= set(led.columns):
            g = led.copy()
            g["_ion"] = [SB.ion_key(n, a) if _s(n) and _s(a) else "" for n, a in
                         zip(g["neutral_formula"], g["adduct"])]
            g["_mz"] = pd.to_numeric(g["mz"], errors="coerce")
            full[str(sid)] = g
    for sid, led in (ledgers or {}).items():
        if led is None or not len(led) or not {"mz", "role"} <= set(led.columns):
            continue
        f = led[led["role"].astype(str) == "M0"].copy()
        f["_ion"] = [SB.ion_key(n, a) for n, a in zip(f["neutral_formula"], f["adduct"])]
        f["_mz"] = pd.to_numeric(f["mz"], errors="coerce")
        frames[str(sid)] = f
    tsg = None
    if ts is not None and len(ts) and {"sample_item_id", "mz", "height"} <= set(ts.columns):
        tsg = ts
    spectra: dict = {}

    def spectrum(sid):
        if sid not in spectra:
            sub = tsg[tsg["sample_item_id"].astype(str) == sid] if tsg is not None else None
            e = edges.get(sid)
            if (e is None or not np.isfinite(_num(e))) and sub is not None and len(sub):
                e = float(pd.to_numeric(sub["height"], errors="coerce").quantile(SB.EDGE_Q))
            spectra[sid] = _Spectrum(sub, e) if sub is not None and len(sub) else None
        return spectra[sid]

    rows = []
    for i in idx:
        nf, ad = _s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"])
        counts = _counts(nf, ad)
        if not nf or not ad or not counts:
            continue
        sign = "+" if ad.endswith("+") else "-"
        key, m = SB.ion_key(nf, ad), _num(merged.at[i, "mz"])
        srcs = [x for x in _s(merged.at[i, "srcs"]).split(",") if x]
        efiles = []
        for sid in srcs:
            f = frames.get(sid)
            if f is None:
                continue
            s_ = f[(f["_ion"] == key) & ((f["_mz"] - m).abs() <= m * tol_ppm * 1e-6)]
            if not len(s_) or "episode" not in s_.columns:
                continue
            s_ = s_[s_["episode"].map(truthy)]
            if len(s_):
                efiles.append((sid, s_.loc[pd.to_numeric(s_["height"], errors="coerce").idxmax()]))
        if not efiles:
            continue
        out["judged"] += 1
        tier_in = _s(merged.at[i, "tier"])
        # (a) the mass in every episode file's calibrated window
        a_bad, a_out = [], []          # a_out: the MEASURED failures (a window exists, the mass outside it)
        for sid, r in efiles:
            w = windows.get(sid)
            e = _num(r.get("ppm_error"))
            if not w or not np.isfinite(_num(w[1])) or not np.isfinite(e):
                a_bad.append(f"{sid}: no calibrated window")
            elif abs(e - _num(w[0])) > K_SIGMA * _num(w[1]):
                a_bad.append(f"{sid}: {e:+.2f} ppm outside {_num(w[0]):+.2f} +- {K_SIGMA * _num(w[1]):.2f}")
                a_out.append(a_bad[-1])
        a_ok = not a_bad
        # the episode spectra's line facts
        facts_rows = []
        for sid, r in efiles:
            sp = spectrum(sid)
            if sp is not None:
                facts_rows.append((sid, sp, _file_facts(sp, r, counts, sign)))
        facts = [f for _sid, _sp, f in facts_rows]
        b_state, b_det = _isotopes(facts, counts) if facts else ("untestable", "no series rows")
        # (c) the rivals the isotope facts must exclude
        try:
            mz_ion = C.ion_mz(nf, ad)
        except Exception:  # noqa: BLE001
            mz_ion = m
        rivals = _alias_rivals(counts, sign, adducts, labelled) + _named_rivals(
            [r for _sid, r in efiles], lvl.get((nf, ad)), mz_ion)
        left, gone, favoured = [], [], []
        for name, rc in rivals:
            if rc == counts:
                continue
            why = _excluded(facts, counts, rc) if facts else None
            (gone if why else left).append(f"{name}" + (f" ({why})" if why else ""))
            if not why and facts:
                fav = _favours(facts, counts, rc)
                if fav:
                    favoured.append(f"{name} ({fav})")
        c_ok = not left
        # (d) no hold: the row's own notes, the pooled isotope-contradiction facts, and the low-file
        # hold's two clauses read on the episode rows themselves (a Candidate reading the hold never tested)
        tr = _s(merged.at[i, "tier_reason"])
        holds = [mk for mk in (SB.HOLD_MARK, SB.SATELLITE_HOLD_MARK, IG.NOTE_MARK) if mk + ":" in tr or
                 f"{mk} in " in tr]
        pf = pooled_facts.get((nf, ad))
        if pf and IG.NOTE_MARK not in holds:
            holds.append(f"{IG.NOTE_MARK} ({pf[0]})")
        sb = _sideband(full, efiles, ts_sorted, counts, sigma, orbitrap)
        if sb:
            holds.append(sb)
        d_ok = not holds
        # (e) an adduct partner in the same spectra, or a second episode file of the ion
        pair = _partner(facts_rows, nf, ad, adducts, sign)
        e_ok = bool(pair) or len(efiles) >= 2
        e_det = pair or (f"{len(efiles)} episode files read the ion" if len(efiles) >= 2 else
                         "one episode file, no adduct partner in its spectrum")
        radical = ad in ("[M]-.", "[M]+.") or bool(
            _s(merged.at[i, "ion_only_of"] if "ion_only_of" in merged.columns else ""))
        exotic = sorted(set(counts) - PROMOTABLE)
        elements_ok = not exotic and not radical
        all_ok = a_ok and b_state == "holds" and c_ok and d_ok and e_ok and elements_ok
        failed = ([f"(a) {'; '.join(a_bad)}"] if not a_ok else []) \
            + ([f"(b) {b_state}: {b_det}"] if b_state != "holds" else []) \
            + ([f"(c) rival left: {'; '.join(left[:3])}" + (f" (+{len(left) - 3})" if len(left) > 3 else "")]
               if not c_ok else []) \
            + ([f"(d) held: {', '.join(holds)}"] if not d_ok else []) \
            + ([f"(e) {e_det}"] if not e_ok else []) \
            + ([f"elements {''.join(exotic)} no isotope speaks for: never promoted"] if exotic else []) \
            + (["a radical / ion-only reading: never promoted"] if radical else [])
        files_txt = ",".join(sid for sid, _r in efiles)
        prior = _s(merged.at[i, "tier_reason"])
        measured, meas = False, []
        if all_ok:
            verdict, claim, tier_out = "assigned", ASSIGNED_CLAIM, "Assigned"
            msg = (f"{NOTE_MARK}: Assigned from a single episode ({len(efiles)} file(s): {files_txt}) -- (a) in the "
                   f"calibrated window; (b) {b_det}; (c) "
                   + (f"rivals excluded: {'; '.join(gone[:3])}" if gone else "no alias rival")
                   + f"; (e) {e_det}")
            out["assigned"] += 1
            out["promoted"] += int(tier_in != "Assigned")
        else:
            verdict = "untestable" if b_state == "untestable" else "candidate"
            claim = UNTESTABLE_CLAIM if b_state == "untestable" else "single episode, " + failed[0].split(":")[0]
            curated = nf in exempt
            # `measured` demotes only on a contradicting MEASUREMENT, never on a missing one: (a) a mass
            # outside an existing window; (b) an isotope line read against the reading; (c) a rival a
            # measured isotope fact favours; (d) a hold (each a measurement); (e) never -- the absence of an
            # adduct pair or a second file is no measurement against the reading
            meas = ([f"(a) {'; '.join(a_out)}"] if a_out else []) \
                + ([f"(b) {b_det}"] if b_state == "failed" else []) \
                + ([f"(c) favoured rival: {'; '.join(favoured[:3])}"] if favoured else []) \
                + ([f"(d) held: {', '.join(holds)}"] if not d_ok else [])
            measured = bool(meas)
            demote = {"strict": True, "measured": measured, "promote-only": False}[mode]
            tier_out = ("Candidate" if (demote and not curated) or tier_in != "Assigned" else tier_in)
            msg = (f"{NOTE_MARK}: not Assigned from its episode ({len(efiles)} file(s): {files_txt}) -- "
                   + "; ".join(failed)
                   + ("" if tier_out == tier_in else "; Candidate")
                   + ("; the claim: " + UNTESTABLE_CLAIM if b_state == "untestable" else ""))
            out["untestable"] += int(b_state == "untestable")
            out["candidate"] += 1
            out["demoted"] += int(tier_in == "Assigned" and tier_out != "Assigned")
            out["kept_assigned"] += int(tier_in == "Assigned" and tier_out == "Assigned")
        merged.at[i, "tier"] = tier_out
        merged.at[i, "episode_claim"] = claim
        _note(merged, i, msg)
        rows.append(dict(mz=m, neutral_formula=nf, adduct=ad, tier_in=tier_in, tier_out=tier_out, verdict=verdict,
                         episode_claim=claim, n_files=int(_num(merged.at[i, "n_files"]))
                         if "n_files" in merged.columns else len(srcs),
                         episode_files=files_txt, a_mass=("ok" if a_ok else "; ".join(a_bad)), b_isotopes=b_state,
                         b_detail=b_det, c_rivals=("ok" if c_ok else "left"),
                         c_detail="; ".join(left + gone), d_hold=", ".join(holds), e_second=e_ok, e_detail=e_det,
                         note=msg, measured_failure=bool(measured), mode=mode, prior_reason=prior,
                         measurement="; ".join(meas) if measured else ""))
    if out["judged"]:
        log(f"[episode] {out['judged']} episodic reading(s) in <= {max_files} files judged on their episode's "
            f"spectra: {out['assigned']} Assigned ({out['promoted']} promoted), {out['candidate']} not "
            f"({out['untestable']} isotopes untestable, {out['demoted']} demoted)")
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), out


# --------------------------------------------------------------------------- Part C
def _series_lines(ts: pd.DataFrame):
    """A line reader over the batch series: f(m, ppm) -> {sample id: the tallest height within ppm}."""
    t = ts[["sample_item_id", "mz", "height"]].dropna(subset=["mz"]).sort_values("mz", kind="mergesort")
    mz = t["mz"].to_numpy(dtype=float)
    sid = t["sample_item_id"].astype(str).to_numpy()
    h = pd.to_numeric(t["height"], errors="coerce").to_numpy(dtype=float)

    def line(m, ppm=CLUSTER_PPM):
        w = m * ppm * 1e-6
        lo, hi = np.searchsorted(mz, m - w, "left"), np.searchsorted(mz, m + w, "right")
        if hi <= lo:
            return pd.Series(dtype=float)
        return pd.Series(h[lo:hi]).groupby(sid[lo:hi]).max()
    return line


def _elevated(h: pd.Series, edge: pd.Series, n_spec: int, x_edge: float) -> set:
    """The spectra where a line stands out: >= x_edge x the spectrum's noise edge, and -- for a
    line present in more than half of the batch's spectra (a background) -- >= SOLVENT_LIFT x
    its median height there."""
    h = h[h > 0]
    if not len(h):
        return set()
    thr = x_edge * edge.reindex(h.index).to_numpy()
    if len(h) > 0.5 * n_spec:
        thr = np.maximum(thr, SOLVENT_LIFT * float(h.median()))
    return set(h.index[h.to_numpy() >= thr])


def _r(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or not (a.std() > 0 and b.std() > 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pair_test(k, p, s) -> dict:
    """The plume-solvent pair test over the episode spectra holding all three lines: the
    cluster `k`, the parent `p`, the solvent ion `s` (arrays). {n, r (k on p x s), r_solvent
    (k on s alone), cv_parent (std / mean of p), specific}: specific = n >= SPECIFIC_MIN_N,
    r >= TRACK_R, r - r_solvent >= SPECIFIC_GAIN and cv_parent >= PARENT_CV_MIN -- the parent's
    own variation, not the episode's, carries the line."""
    k, p, s = (np.asarray(x, dtype=float) for x in (k, p, s))
    r, rs = _r(k, p * s), _r(k, s)
    cv = float(p.std() / p.mean()) if len(p) and p.mean() > 0 else float("nan")
    spec = bool(len(k) >= SPECIFIC_MIN_N and np.isfinite(r) and r >= TRACK_R and np.isfinite(rs)
                and r - rs >= SPECIFIC_GAIN and np.isfinite(cv) and cv >= PARENT_CV_MIN)
    return {"n": int(len(k)), "r": r, "r_solvent": rs, "cv_parent": cv, "specific": spec}


def cluster_label(adduct: str, solvent: str) -> str | None:
    """The adduct of a persistent ion's cluster with `solvent`: '[M+H]+' -> '[M+(C5H9NO)H]+',
    '[M+(CH4N2O)H]+' -> '[M+(C5H9NO)(CH4N2O)H]+', '[M-H]-' -> '[M+(C2H6OS)-H]-'; None when the
    parent's adduct has no such form (the label's ion must be the parent's + the solvent)."""
    ad = _s(adduct)
    if ad.startswith("[M+"):
        lab = f"[M+({solvent}){ad[3:]}"
    elif ad.startswith("[M-"):
        lab = f"[M+({solvent}){ad[2:]}"
    else:
        return None
    return lab


def read_solvent_clusters(merged: pd.DataFrame, ts: pd.DataFrame | None, *, adducts=(),
                          log=print) -> tuple[pd.DataFrame, dict]:
    """Part C (module note): relabel the merged rows that are a persistent Assigned ion's
    cluster with a plume solvent bright in an episode. Mutates merged (neutral_formula,
    adduct, tier, alternatives, tier_reason, `solvent_cluster_of`); returns (table, summary)."""
    out = {"solvents_bright": [], "relabelled": 0, "pair_passed": 0, "capped": 0}
    if merged is None or not len(merged) or ts is None or not len(ts) or "episode" not in merged.columns:
        return empty_clusters(), out
    if not {"sample_item_id", "mz", "height"} <= set(ts.columns):
        return empty_clusters(), out
    if "solvent_cluster_of" not in merged.columns:
        merged["solvent_cluster_of"] = pd.NA
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    line = _series_lines(ts)
    n_spec = int(ts["sample_item_id"].nunique())
    edge = pd.to_numeric(ts["height"], errors="coerce").groupby(ts["sample_item_id"].astype(str)).quantile(SB.EDGE_Q)
    sign = "+" if any(str(a).endswith("+") for a in adducts) else "-"
    mzs = pd.to_numeric(merged["mz"], errors="coerce")
    rows = []
    for solv in SC.episode_solvents():
        ms = C.neutral_mass(solv.formula)
        ions = []
        for a in adducts:
            if not str(a).endswith(sign):
                continue
            try:
                ions.append(C.ion_mz(solv.formula, a))
            except Exception:  # noqa: BLE001
                continue
        if sign == "+":
            ions.append(2 * ms + C.M["H"] - C.M_E)          # the proton-bound dimer
        # the solvent's own ion: each of its forms where it stands out (`_elevated`: >= SOLVENT_X x the
        # edge and, for a form present in most spectra, >= SOLVENT_LIFT x its median -- which makes the
        # episode at most half of the batch's spectra by itself: no more than half a line's heights can
        # reach 5x its median -- per FORM: `bright`, the union over the forms, may exceed half)
        hs, bright = None, set()
        for m_ in ions:
            s_ = line(m_)
            if not len(s_):
                continue
            b_ = _elevated(s_, edge, n_spec, SOLVENT_X)
            if not b_:
                continue
            bright |= b_
            hs = s_ if hs is None else pd.concat([hs, s_], axis=1).max(axis=1)
        if hs is None or not bright:
            continue
        out["solvents_bright"].append({"solvent": solv.formula, "name": solv.name, "n_bright": len(bright)})
        # persistent Assigned parents
        par = merged[(merged["tier"].astype(str) == "Assigned") & (merged["solvent_cluster_of"].map(_s) == "")]
        for j, p in par.iterrows():
            pm = _num(p["mz"])
            tgt = pm + ms
            hit = merged.index[((mzs - tgt).abs() <= tgt * CLUSTER_PPM * 1e-6)
                               & (merged["solvent_cluster_of"].map(_s) == "")]
            hit = [k for k in hit if k != j]
            if not hit:
                continue
            ps = line(pm)
            if len(ps) < PERSIST_FRAC * n_spec:
                continue
            lab = cluster_label(p["adduct"], solv.formula)
            if lab is None:
                continue
            want = dict(_counts(p["neutral_formula"], p["adduct"]))
            for el, n in C.parse_formula(solv.formula).items():
                want[el] = want.get(el, 0) + n
            if _counts(p["neutral_formula"], lab) != {k: v for k, v in want.items() if v}:
                continue
            for k in hit:
                ks = line(_num(merged.at[k, "mz"]))
                ks = ks[ks > 0]
                # its PRESENCE is confined to the episode (a line seen elsewhere is no plume cluster)
                if not len(ks) or len(ks) > CLUSTER_MAX_FRAC * n_spec:
                    continue
                n_in = int(sum(1 for x in ks.index if x in bright))
                if n_in < CONFINED_FRAC * len(ks):
                    continue
                kel = ks.index
                # the pair test reads the episode's own spectra: parent x solvent there, never one spike
                j3 = pd.concat([ks.rename("k"), ps.rename("p"), hs.rename("s")], axis=1, join="inner").dropna()
                j3 = j3[(j3 > 0).all(axis=1) & j3.index.isin(list(bright))]
                if len(j3) < TRACK_MIN_N:
                    continue                   # too few episode spectra to read anything: nothing is read
                pt = pair_test(j3["k"].to_numpy(), j3["p"].to_numpy(), j3["s"].to_numpy())
                r, passed = pt["r"], pt["specific"]
                same_ion = _counts(merged.at[k, "neutral_formula"], merged.at[k, "adduct"]) == _counts(
                    p["neutral_formula"], lab)
                if not same_ion and not passed:
                    continue                   # another ion at the mass is re-read only on the specific test
                ptier = _s(p["tier"])
                tier_out = ptier if passed else ("Candidate" if ptier == "Assigned" else ptier)
                was = (_s(merged.at[k, "neutral_formula"]), _s(merged.at[k, "adduct"]))
                tier_in = _s(merged.at[k, "tier"])
                merged.at[k, "neutral_formula"] = p["neutral_formula"]
                merged.at[k, "adduct"] = lab
                merged.at[k, "tier"] = tier_out
                merged.at[k, "solvent_cluster_of"] = f"{p['neutral_formula']} {p['adduct']} + {solv.formula}"
                if "alternatives" in merged.columns and all(was):
                    prev = _s(merged.at[k, "alternatives"])
                    merged.at[k, "alternatives"] = f"{was[0]} {was[1]} (the covalent reading)" + (
                        f"; {prev}" if prev else "")
                _note(merged, k, (f"{CLUSTER_MARK}: the {solv.formula} ({solv.name}) cluster of the persistent "
                                  f"{p['neutral_formula']} {p['adduct']} (m/z {pm:.4f}, +{ms:.4f}): present in "
                                  f"{len(kel)} spectra, {n_in} of them in the solvent's episode; "
                                  + (f"tracks parent x solvent specifically (r {r:.2f} vs {pt['r_solvent']:.2f} on "
                                     f"the solvent alone, parent CV {pt['cv_parent']:.2f}, {pt['n']} episode "
                                     "spectra): the parent's tier" if passed else
                                     f"solvent cluster of {p['neutral_formula']} {p['adduct']}: {CO_OCCURS} (r "
                                     f"{r:.2f} on parent x solvent vs {pt['r_solvent']:.2f} on the solvent alone, "
                                     f"parent CV {pt['cv_parent']:.2f}, {pt['n']} episode spectra -- not a specific "
                                     "track): capped at Candidate")
                                  + (f"; was {was[0]} {was[1]}" if all(was) else "")))
                out["relabelled"] += 1
                out["pair_passed"] += int(passed)
                out["capped"] += int(not passed)
                rows.append(dict(mz=_num(merged.at[k, "mz"]), was_neutral=was[0], was_adduct=was[1],
                                 neutral_formula=p["neutral_formula"], adduct=lab, solvent=solv.formula,
                                 parent_mz=pm, parent_reading=f"{p['neutral_formula']} {p['adduct']}",
                                 parent_tier=ptier, tier_in=tier_in, tier_out=tier_out, n_spectra=len(kel),
                                 n_in_episode=n_in, n_track=len(j3), track_r=round(r, 4) if np.isfinite(r) else np.nan,
                                 r_solvent=(round(pt["r_solvent"], 4) if np.isfinite(pt["r_solvent"]) else np.nan),
                                 cv_parent=round(pt["cv_parent"], 4) if np.isfinite(pt["cv_parent"]) else np.nan,
                                 pair_test=passed))
    if out["relabelled"]:
        log(f"[episode] {out['relabelled']} merged row(s) read as a persistent ion's plume-solvent cluster "
            f"({out['pair_passed']} track parent x solvent specifically, {out['capped']} capped at Candidate: "
            f"{CO_OCCURS})")
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS), out


def cluster_rows(merged: pd.DataFrame) -> pd.Series:
    """The merged rows Part C read as a solvent cluster (a boolean mask)."""
    if merged is None or "solvent_cluster_of" not in merged.columns:
        return pd.Series(False, index=getattr(merged, "index", None))
    return merged["solvent_cluster_of"].map(_s) != ""
