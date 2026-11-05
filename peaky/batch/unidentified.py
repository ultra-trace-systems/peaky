"""Real but unidentified lines: every bright per-file line that no batch output
explains gets a row in ``tables/unidentified.csv`` with the reason
(docs/OUTPUTS.md, docs/MERGE.md step 7).

Two pieces:

* the **vote-drop recorder** (`DropLog`, `vote_drops`): the merge's own list of
  the per-file M0 rows that leave it with no merged row at their m/z -- the
  merged rows a merge gate removed after the vote (the reagent-cluster strip,
  the reagent-water rungs, the isotopologue strip, the element-signature
  removal) and the trace-first scope cut, one row per per-file M0 row
  ``{file, peak_id, mz, neutral_formula, adduct, tier_in, reason}``
  (``tables/vote_drops.csv``);
* the **unidentified table** (`build`): one row per m/z bin of the batch's
  unidentified per-file rows that holds a BRIGHT one, where

  - *bright* = height >= `BRIGHT_X` x the file's noise edge
    (batch_summary ``per_file[].noise_edge_cps``, the file's own run's
    ``cfg.noise_edge_cps``; ``height_gate_cps`` where a file has none), or a
    row its ledger flags ``episode`` (the hook an episode-aware selection sets;
    no ledger carries it today, so it reads False);
  - *unidentified* = a per-file row whose role at the end of the run is
    ``unexplained``; an ``iso_child`` / ``artifact`` whose parent is not
    explained (an M0 parent the merged ledger does not hold Assigned at its
    m/z, or a parent that is itself unexplained; a reagent parent explains it);
    or an M0 row the vote-drop recorder lists;

  and no merged row (any tier) sits at the bin's m/z (the merge window).
  ``why`` is the first of ``vote_dropped``, ``orphan_iso_child``,
  ``orphan_artifact``, ``all_blocked`` (a reading was tried and a gate other
  than the score / box / admission refused it), ``no_formula`` (no reading
  fitted: only the score floor, the grid box or the admission gate spoke)
  that one of the bin's rows has.

Diagnostics: nothing here changes a tier, a reading or a merged row.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from peaky.assignment import blocked as BL
from peaky.batch import iso_checks as IC
from peaky.chem import isotopes as ISO

__version__ = "0.1.0"

#: a line is bright at this many x its file's noise edge
BRIGHT_X = 5.0
#: a partner (orphan's parent, satellite) within this many Da of the line; brighter than it
PARTNER_DA = 0.0040
#: the isotopologue spacings a partner may sit BELOW the line at (Da)
PARTNER_SPACINGS = {"13C": ISO.D_13C_EXACT, "2x13C": 2 * ISO.D_13C_EXACT, "34S": ISO.D_34S,
                    "37Cl": ISO.D_37CL, "18O": ISO.D_18O, "15N": ISO.D_15N}
#: the position window (ppm) of an isotope line probe: rule C's (iso_checks.C_TOL_PPM)
LINE_TOL_PPM = IC.C_TOL_PPM
#: a 34S / 37Cl line is testable where ONE atom's line would reach this predicted S/N
#: (levels.lines.DET_SNR, the same number) -- with no server S/N, DET_X x the noise edge
LINE_MIN_SNR = 10.0
LINE_DET_X = 3.0
#: the gate classes that mean "no reading fitted" (no_formula); any other gate on a
#: tried reading means a reading fitted and was refused (all_blocked)
NO_FIT_GATES = ("no candidate in box", "no candidate scored", "not admitted", "score<min_raw", "no ppm",
                "pass4:score", "confidence Reject", "labeled:score", "released")

#: two picks of one file count as two resolved lines (a line the merge absorbed beside a merged
#: row's peak) only when they are at least this many HWHM apart by the batch's width model:
#: closer maxima of one line are a split centroid or a shoulder, not two ions
ABSORB_MIN_HWHM = 1.0

WHY_ORDER = ("vote_dropped", "absorbed_by_merge", "orphan_iso_child", "orphan_artifact", "all_blocked", "no_formula")

TABLE_COLUMNS = ("mz", "mz_corrected", "n_files_present", "n_files_selected", "h_max", "x_edge_max", "roles",
                 "c13_count", "s34_line", "cl37_line", "readings_tried", "partner_mz", "partner_role", "why",
                 "first_seen", "last_seen", "episode")
DROP_COLUMNS = ("file", "peak_id", "mz", "neutral_formula", "adduct", "tier_in", "reason")

#: the reason of a dropped M0 row whose merged row no gate is recorded as removing
REASON_UNRECORDED = "no merged row at its m/z"
REASON_SCOPE = "trace-first scope (not on a residual bin)"

_ART_PARENT = re.compile(r"(?:side lobe|ringing/sidelobe) of (?:m/z )?(\d+\.\d+)")
_BLOCKED_READING = re.compile(r"^(?P<neutral>\S+) (?P<adduct>\S+) \((?P<ion>[^)]*)\)")


def empty_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in TABLE_COLUMNS})


def empty_drops() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in DROP_COLUMNS})


def _txt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    return "" if s in ("nan", "<NA>", "None") else s


def _truthy(v) -> bool:
    return _txt(v).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- the vote-drop recorder
class DropLog:
    """The merged rows the merge's gates removed after the vote, in order:
    ``note(before, after, gate)`` records every row of ``before`` (by m/z,
    neutral, adduct) missing from ``after``."""

    def __init__(self):
        self.removed: list[dict] = []

    @staticmethod
    def _keys(frame) -> list:
        if frame is None or not len(frame):
            return []
        return list(zip(np.round(pd.to_numeric(frame["mz"], errors="coerce").to_numpy(float), 6),
                        frame["neutral_formula"].map(_txt), frame["adduct"].map(_txt)))

    def note(self, before, after, gate: str) -> int:
        kb, ka = self._keys(before), set(self._keys(after))
        n = 0
        for k in kb:
            if k not in ka:
                self.removed.append(dict(mz=float(k[0]), neutral_formula=k[1], adduct=k[2], gate=str(gate)))
                n += 1
        return n

    def clear(self) -> None:
        self.removed = []


def _adj(mz, off) -> np.ndarray:
    o = float(off) if off is not None and np.isfinite(float(off)) else 0.0
    return np.asarray(mz, dtype=float) * (1.0 - o / 1e6)


def _near(sorted_mz: np.ndarray, x: float, tol_ppm: float) -> int:
    """Index of the nearest value of ``sorted_mz`` within ``tol_ppm`` of x, else -1."""
    if not len(sorted_mz) or not np.isfinite(x):
        return -1
    j = int(np.searchsorted(sorted_mz, x))
    best, bd = -1, float("inf")
    for k in (j - 1, j):
        if 0 <= k < len(sorted_mz):
            d = abs(sorted_mz[k] - x)
            if d < bd:
                best, bd = k, d
    return best if bd <= x * tol_ppm * 1e-6 else -1


#: a removed merged row is its cluster's when their m/z agree to this many Da (the row's m/z IS the cluster mean)
CLUSTER_MZ_DA = 1e-5


def _cluster_index(jitter: pd.DataFrame | None, rem: list) -> tuple[dict, dict]:
    """From the vote's long form (`align`'s jitter: cluster, src, mz -- every per-file row of every
    cluster): {(src, m/z): cluster} and {cluster: the removal (``rem``) whose merged row it was}."""
    if jitter is None or not len(jitter) or not {"cluster", "src", "mz"} <= set(jitter.columns):
        return {}, {}
    j = jitter[["cluster", "src", "mz"]].copy()
    j["mz"] = pd.to_numeric(j["mz"], errors="coerce")
    where = {(str(s), round(float(m), 7)): int(c) for c, s, m in zip(j["cluster"], j["src"], j["mz"]) if np.isfinite(m)}
    means = j.groupby("cluster")["mz"].mean()
    rmz = np.array([r["mz"] for r in rem], dtype=float)
    o = np.argsort(rmz)
    rs = rmz[o]
    gone = {}
    for c, m in means.items():
        if not np.isfinite(m) or not len(rs):
            continue
        k = int(np.searchsorted(rs, m))
        for q in (k - 1, k):
            if 0 <= q < len(rs) and abs(rs[q] - m) <= CLUSTER_MZ_DA:
                gone[int(c)] = rem[int(o[q])]
                break
    return where, gone


def vote_drops(full_ledgers: dict, align_input: dict, merged: pd.DataFrame, log: DropLog | None, *,
               tol_ppm: float, offsets: dict | None = None, jitter: pd.DataFrame | None = None) -> pd.DataFrame:
    """The per-file M0 rows (``full_ledgers``) the merge dropped. With the vote's
    long form (``jitter``: align's cluster of every per-file row) a row is dropped
    when the merged row of ITS cluster is one a gate removed (``log``), reason =
    that gate and the removed reading; without it (or for a row the long form does
    not hold), when no row of ``merged`` lies within ``tol_ppm`` of its m/z, reason
    = the removal nearest within the window, else ``REASON_UNRECORDED``. A row the
    trace-first scope kept out of the vote (``align_input`` = the vote's input
    frames, indexed as the ledgers) is ``REASON_SCOPE``."""
    offsets = offsets or {}
    mm = np.sort(pd.to_numeric(merged["mz"], errors="coerce").dropna().to_numpy(float)) \
        if merged is not None and len(merged) else np.array([])
    rem = sorted((log.removed if log is not None else []), key=lambda r: r["mz"])
    rmz = np.array([r["mz"] for r in rem], dtype=float)
    where, gone = _cluster_index(jitter, rem)
    rows = []
    for sid in sorted(full_ledgers, key=str):
        led = full_ledgers[sid]
        if led is None or not len(led) or "role" not in led.columns:
            continue
        m0 = led[led["role"].astype(str) == "M0"]
        if not len(m0):
            continue
        inp = align_input.get(sid) if align_input else None
        voted = set(inp.index) if inp is not None else set(m0.index)
        mz = pd.to_numeric(m0["mz"], errors="coerce").to_numpy(float)
        adj = _adj(mz, offsets.get(sid))
        for i, x_raw, x in zip(m0.index, mz, adj):
            if not np.isfinite(x):
                continue
            c = where.get((str(sid), round(float(x_raw), 7))) if i in voted else None
            if c is not None:
                if c not in gone:
                    continue          # its cluster's merged row survived (wherever its mean sits)
                why = f"{gone[c]['gate']} ({gone[c]['neutral_formula']} {gone[c]['adduct']})"
            elif _near(mm, x_raw, tol_ppm) >= 0 or _near(mm, x, tol_ppm) >= 0:
                continue
            elif i not in voted:
                why = REASON_SCOPE
            else:
                k = _near(rmz, x, tol_ppm)
                if k < 0:
                    k = _near(rmz, x_raw, tol_ppm)
                why = (f"{rem[k]['gate']} ({rem[k]['neutral_formula']} {rem[k]['adduct']})" if k >= 0
                       else REASON_UNRECORDED)
            rows.append(dict(file=str(sid), peak_id=_txt(led.at[i, "peak_id"]) if "peak_id" in led.columns else "",
                             mz=float(x_raw), neutral_formula=_txt(led.at[i, "neutral_formula"]),
                             adduct=_txt(led.at[i, "adduct"]), tier_in=_txt(led.at[i, "tier"]), reason=why))
    if not rows:
        return empty_drops()
    return pd.DataFrame(rows, columns=list(DROP_COLUMNS))


# --------------------------------------------------------------------------- the unidentified table
def _edge(stats: dict | None) -> float:
    st = stats or {}
    for k in ("noise_edge_cps", "height_gate_cps"):
        v = st.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            return v
    return float("nan")


def _gate_kind(gate: str, reading: str) -> str:
    """'tried' when a reading fitted and a gate other than the no-fit ones refused it, else 'none'."""
    if not reading:
        return "none"
    cls = BL.gate_class(gate) if gate else ""
    return "none" if any(cls.startswith(g) for g in NO_FIT_GATES) else "tried"


def _reading_dict(reading: str, gate: str) -> dict:
    m = _BLOCKED_READING.match(reading or "")
    if m:
        return {"ion": m.group("ion") or None, "neutral": m.group("neutral"), "adduct": m.group("adduct"),
                "blocked_by": gate or None}
    return {"ion": None, "neutral": None, "adduct": None, "blocked_by": gate or None}


class _File:
    """One file's ledger as sorted arrays (m/z, height, area, S/N, role, peak_id)."""

    def __init__(self, sid, led, edge, offset):
        self.sid = sid
        self.led = led
        self.edge = edge
        self.offset = offset
        mz = pd.to_numeric(led["mz"], errors="coerce").to_numpy(float)
        o = np.argsort(mz, kind="mergesort")
        self.idx = led.index.to_numpy()[o]
        self.mz = mz[o]
        self.adj = _adj(self.mz, offset)
        self.h = pd.to_numeric(led["height"], errors="coerce").to_numpy(float)[o] if "height" in led.columns \
            else np.full(len(o), np.nan)
        self.a = pd.to_numeric(led["area"], errors="coerce").to_numpy(float)[o] if "area" in led.columns \
            else np.full(len(o), np.nan)
        self.snr = (pd.to_numeric(led["signal_to_noise"], errors="coerce").to_numpy(float)[o]
                    if "signal_to_noise" in led.columns else np.full(len(o), np.nan))
        self.role = led["role"].astype(str).to_numpy()[o]
        self.pos = {k: j for j, k in enumerate(self.idx)}

    def window(self, x, tol_da):
        return (int(np.searchsorted(self.mz, x - tol_da, "left")), int(np.searchsorted(self.mz, x + tol_da, "right")))

    def tallest(self, x, tol_da, exclude=-1):
        lo, hi = self.window(x, tol_da)
        best = -1
        for j in range(lo, hi):
            if j != exclude and (best < 0 or self.h[j] > self.h[best]):
                best = j
        return best


def _row_text(f: _File, j: int) -> str:
    i = f.idx[j]
    led = f.led
    r = f.role[j]
    if r == "M0":
        return f"M0 {_txt(led.at[i, 'tier'])} {_txt(led.at[i, 'neutral_formula'])} {_txt(led.at[i, 'adduct'])}".strip()
    if r == "iso_child":
        return (f"iso_child ({_txt(led.at[i, 'iso_label'])}) of {_txt(led.at[i, 'parent_neutral_formula'])} "
                f"{_txt(led.at[i, 'parent_adduct'])}").strip()
    return r


def _c13_count(f: _File, j: int) -> float:
    """The carbon count from the row's own 13C line (the per-file count of
    `levels.context.carbon_counts`: area ratio at +D13C within C_TOL_PPM, over
    the mass factor ((m+1)/m)^1.5 and R13C), where a line of that count is
    measurable: its predicted S/N >= iso_checks.C_MIN_SNR13 (with the row's S/N)
    and its height >= C_KDL x the edge. NaN when untestable."""
    x0, h0, a0, sn = f.mz[j], f.h[j], f.a[j], f.snr[j]
    if not (np.isfinite(x0) and np.isfinite(h0) and h0 > 0):
        return float("nan")
    t = x0 + IC.D13C
    k = f.tallest(t, IC.C_TOL_PPM * t * 1e-6, exclude=j)
    use_area = np.isfinite(a0) and a0 > 0
    v1 = (f.a[k] if use_area else f.h[k]) if k >= 0 else 0.0
    v1 = v1 if np.isfinite(v1) else 0.0
    v0 = a0 if use_area else h0
    wf = ((x0 + 1.0) / x0) ** 1.5
    c = (v1 / v0) / wf / IC.R13C
    n = max(c, 1.0)
    if np.isfinite(sn) and sn * n * IC.R13C < IC.C_MIN_SNR13:
        return float("nan")
    if np.isfinite(f.edge) and h0 * n * IC.R13C < IC.C_KDL * f.edge:
        return float("nan")
    return float(c)


def _line_state(f: _File, j: int, spacing: float, ratio: float) -> str:
    """'present' | 'absent' | 'untestable' for the line ``spacing`` Da above the
    row: present = a peak within LINE_TOL_PPM; absent = none, where ONE atom's
    line (``ratio`` x the row) would reach LINE_MIN_SNR (the row's S/N) and
    LINE_DET_X x the edge."""
    x0, h0, sn = f.mz[j], f.h[j], f.snr[j]
    if not (np.isfinite(x0) and np.isfinite(h0)):
        return "untestable"
    t = x0 + spacing
    if f.tallest(t, LINE_TOL_PPM * t * 1e-6, exclude=j) >= 0:
        return "present"
    snr_ok = (sn * ratio >= LINE_MIN_SNR) if np.isfinite(sn) else True
    edge_ok = (h0 * ratio >= LINE_DET_X * f.edge) if np.isfinite(f.edge) else False
    return "absent" if (snr_ok and edge_ok) else "untestable"


def _partner(f: _File, j: int) -> tuple[float, str]:
    """The brighter row within +-PARTNER_DA of the line, else the brightest
    brighter row one isotopologue spacing below it: (m/z, '<role text>' or
    '<spacing> parent: <role text>')."""
    x0, h0 = f.mz[j], f.h[j]
    k = f.tallest(x0, PARTNER_DA, exclude=j)
    if k >= 0 and f.h[k] > h0:
        return float(f.mz[k]), f"within {1e3 * (f.mz[k] - x0):+.2f} mDa: {_row_text(f, k)}"
    best, lab = -1, ""
    for name, d in PARTNER_SPACINGS.items():
        t = x0 - d
        k = f.tallest(t, LINE_TOL_PPM * t * 1e-6)
        if k >= 0 and f.h[k] > h0 and (best < 0 or f.h[k] > f.h[best]):
            best, lab = k, name
    if best >= 0:
        return float(f.mz[best]), f"{lab} parent: {_row_text(f, best)}"
    return float("nan"), ""


def _parent_of(f: _File, j: int) -> int:
    """The position of the row's parent in the file: an iso_child's
    parent_peak_id; an artifact's lobe / ringing parent named in its commentary."""
    i = f.idx[j]
    led = f.led
    if f.role[j] == "iso_child" and "parent_peak_id" in led.columns and "peak_id" in led.columns:
        pid = _txt(led.at[i, "parent_peak_id"])
        if pid:
            hit = led.index[led["peak_id"].astype(str) == pid]
            if len(hit):
                return f.pos.get(hit[0], -1)
        return -1
    if f.role[j] == "artifact":
        m = _ART_PARENT.search(_txt(led.at[i, "commentary"]) if "commentary" in led.columns else "")
        if m:
            x = float(m.group(1))
            k = f.tallest(x, 0.001, exclude=j)
            return k
    return -1


def build(full_ledgers: dict, per_stats, merged: pd.DataFrame, *, drops: pd.DataFrame | None = None,
          ts=None, tol_ppm: float, offsets: dict | None = None, axis_info: dict | None = None,
          bright_x: float = BRIGHT_X, resolution=None) -> pd.DataFrame:
    """The unidentified table (module note). ``per_stats``: the files' stats
    (each with ``sample_id`` and ``noise_edge_cps``) or {sid: stats};
    ``merged``: the final merged ledger; ``drops``: `vote_drops`; ``ts``: the
    batch series (sample_item_id, mz, height, datetime_utc) for first / last
    seen; ``axis_info``: batch_summary['mass_axis'] (``mz`` on the server's
    axis, ``mz_corrected`` the ledgers' when it applied); ``resolution``: the
    batch's width model (chem.resolution.Resolution or its dict) -- without one
    no line is read as absorbed by the merge (no separation test). Brightest
    first."""
    offsets = offsets or {}
    _rp = resolution
    if isinstance(_rp, dict):
        from peaky.chem.resolution import Resolution
        _rp = Resolution.from_dict(_rp) if _rp.get("coef") else None
    if isinstance(per_stats, dict):
        stats = {str(k): v for k, v in per_stats.items()}
    else:
        stats = {str(s.get("sample_id")): s for s in (per_stats or []) if isinstance(s, dict)}
    mm_raw = np.sort(pd.to_numeric(merged["mz"], errors="coerce").dropna().to_numpy(float)) \
        if merged is not None and len(merged) else np.array([])
    # the merged rows with their ION (assign_batch._ion_key, the vote's key) and text, m/z-sorted: a
    # per-file row inside a merged row's window is "absorbed" when it reads another ion and its own
    # file holds the merged ion's peak beside it (two picks in one spectrum: resolved lines)
    from peaky.batch.assign_batch import _ion_key, _s
    if merged is not None and len(merged):
        _mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy(float)
        _ok = np.isfinite(_mz)
        _o = np.argsort(_mz[_ok], kind="mergesort")
        _mm = merged[_ok].iloc[_o]
        M_mz = _mz[_ok][_o]
        _nf = _mm["neutral_formula"] if "neutral_formula" in _mm.columns else pd.Series([None] * len(_mm))
        _ad = _mm["adduct"] if "adduct" in _mm.columns else pd.Series([None] * len(_mm))
        M_ion = [_ion_key(_s(n), _s(a)) for n, a in zip(_nf, _ad)]
        M_txt = [f"{_s(t)} {_s(n)} {_s(a)}".strip() for t, n, a in zip(_mm["tier"], _nf, _ad)]
    else:
        M_mz, M_ion, M_txt = np.array([]), [], []

    def _hwhm(mz) -> float:
        if _rp is None:
            return float("nan")
        try:
            return float(_rp.fwhm(float(mz))) / 2.0
        except Exception:  # noqa: BLE001 -- a width model that cannot read the mass: no separation test
            return float("nan")

    def _merged_at(f, j):
        """(k, axis): the merged row (its index in M_mz) within the window of row j of file f -- on the
        raw m/z first, else on the offset-corrected one -- and the file's m/z array on that axis;
        (-1, None) when none."""
        k = _near(M_mz, f.mz[j], tol_ppm)
        if k >= 0:
            return k, f.mz
        k = _near(M_mz, f.adj[j], tol_ppm)
        return (k, f.adj) if k >= 0 else (-1, None)

    def _resolved(f, j):
        """(k, q): row j of file f sits beside merged row k as a RESOLVED line -- j within the window of
        k, and the same file holds another pick q (not an artifact row) nearer k, at least
        ABSORB_MIN_HWHM HWHM (the width model at j) from j: two picks of one spectrum that the width
        model separates; (-1, -1) when not."""
        k, ax = _merged_at(f, j)
        if k < 0:
            return -1, -1
        x = M_mz[k]
        hw = _hwhm(ax[j])
        if not (np.isfinite(hw) and hw > 0):
            return -1, -1
        lo = int(np.searchsorted(ax, x - x * tol_ppm * 1e-6, "left"))
        hi = int(np.searchsorted(ax, x + x * tol_ppm * 1e-6, "right"))
        for q in range(lo, hi):
            if q == j or f.role[q] == "artifact":
                continue
            if abs(ax[q] - x) < abs(ax[j] - x) and abs(ax[j] - ax[q]) >= ABSORB_MIN_HWHM * hw:
                return k, q
        return -1, -1
    assigned = merged[merged["tier"].astype(str) == "Assigned"] if merged is not None and len(merged) else None
    ma = np.sort(pd.to_numeric(assigned["mz"], errors="coerce").dropna().to_numpy(float)) \
        if assigned is not None and len(assigned) else np.array([])
    dropped = set()
    drop_reason = {}
    if drops is not None and len(drops):
        for r in drops.itertuples(index=False):
            dropped.add((str(r.file), str(r.peak_id)))
            drop_reason[(str(r.file), str(r.peak_id))] = str(r.reason)
    files, parts = {}, []
    for sid in sorted(full_ledgers, key=str):
        led = full_ledgers[sid]
        if led is None or not len(led) or not {"role", "mz"} <= set(led.columns):
            continue
        f = _File(str(sid), led, _edge(stats.get(str(sid))), offsets.get(sid))
        files[str(sid)] = f
        epi = (led["episode"].map(_truthy).to_numpy()[np.argsort(pd.to_numeric(led["mz"], errors="coerce")
                                                                  .to_numpy(float), kind="mergesort")]
               if "episode" in led.columns else np.zeros(len(f.mz), dtype=bool))
        pids = (led["peak_id"].astype(str).to_numpy()[np.argsort(pd.to_numeric(led["mz"], errors="coerce")
                                                                 .to_numpy(float), kind="mergesort")]
                if "peak_id" in led.columns else np.array([""] * len(f.mz)))
        for j in range(len(f.mz)):
            r = f.role[j]
            kind = None
            if r == "unexplained":
                kind = "unexplained"
            elif r in ("iso_child", "artifact"):
                p = _parent_of(f, j)
                if p < 0:
                    explained = False
                else:
                    pr = f.role[p]
                    if pr == "reagent":
                        explained = True
                    elif pr == "M0":
                        explained = _near(ma, f.mz[p], tol_ppm) >= 0 or _near(ma, f.adj[p], tol_ppm) >= 0
                    elif pr in ("iso_child", "artifact"):
                        explained = False
                    else:
                        explained = False
                if not explained:
                    kind = "orphan_iso_child" if r == "iso_child" else "orphan_artifact"
            if r == "M0" and (str(sid), str(pids[j])) not in dropped:
                # a vote loser of ANOTHER ion than its merged row: a candidate absorbed line (listed
                # only when some file resolves it from the merged row's peak; see the bins below)
                k, _ax = _merged_at(f, j)
                if k >= 0 and _ion_key(_s(led.at[f.idx[j], "neutral_formula"]),
                                       _s(led.at[f.idx[j], "adduct"])) != M_ion[k]:
                    kind = "absorbed"
            if (str(sid), str(pids[j])) in dropped and (r == "M0" or kind == "unexplained"):
                # an M0 row the merge dropped -- still M0, or released to unexplained by the
                # batch's per-file rewrite (an isotope child of an Assigned parent stays explained)
                kind = "vote_dropped"
            if kind is None or not np.isfinite(f.mz[j]):
                continue
            x = f.h[j] / f.edge if (np.isfinite(f.edge) and f.edge > 0 and np.isfinite(f.h[j])) else np.nan
            parts.append((float(f.adj[j]), str(sid), j, kind, x, bool(epi[j]), str(pids[j])))
    if not parts:
        return empty_table()

    def _bins(pp: list) -> list:
        pp = sorted(pp, key=lambda t: (t[0], t[1]))
        if not pp:
            return []
        ms_ = np.array([p_[0] for p_ in pp])
        cid_ = np.zeros(len(ms_), dtype=np.int64)
        if len(ms_) > 1:
            cid_[1:] = np.cumsum(np.diff(ms_) / ms_[:-1] * 1e6 > float(tol_ppm))
        return [[pp[k_] for k_ in np.where(cid_ == c_)[0]] for c_ in np.unique(cid_)]
    # the bins: the regular parts alone and the absorbed parts (vote losers of another ion) alone, each
    # by single linkage at the window -- an absorbed part never lengthens a regular chain (nor moves a
    # regular bin's median onto a merged row), so it cannot hide a regular line
    groups = (_bins([p_ for p_ in parts if p_[3] != "absorbed"])
              + _bins([p_ for p_ in parts if p_[3] == "absorbed"]))
    ts_mz = ts_t = None
    if ts is not None and len(ts) and {"mz", "datetime_utc"} <= set(ts.columns):
        o = np.argsort(pd.to_numeric(ts["mz"], errors="coerce").to_numpy(float), kind="mergesort")
        ts_mz = pd.to_numeric(ts["mz"], errors="coerce").to_numpy(float)[o]
        ts_t = pd.to_datetime(ts["datetime_utc"], utc=True, errors="coerce").to_numpy()[o]
    applied = bool((axis_info or {}).get("applied"))
    out = []
    n_sel = len(files)
    # each bin decided: held by a merged row (skipped), released as a line the merge absorbed (kept, its
    # members on the resolved side only), or a regular unidentified bin
    decided = []
    for grp in groups:
        grp = sorted(grp, key=lambda t: (t[0], t[1]))
        raw = np.array([files[g[1]].mz[g[2]] for g in grp])
        med_raw = float(np.median(raw))
        med_adj = float(np.median([g[0] for g in grp]))
        kinds = {g[3] for g in grp}
        held = _near(mm_raw, med_raw, tol_ppm) >= 0 or _near(mm_raw, med_adj, tol_ppm) >= 0
        absorbed_by = -1
        if held or kinds <= {"absorbed"}:
            # a merged row holds the line -- unless some file resolves a line from that merged row's
            # peak (two picks of one spectrum the width model separates): then the merge absorbed a line
            # of its own, and the row is built from the bin's members on the resolved side only
            # (beyond the midpoint of the resolving pair, on its side), not from the merged line's picks
            for g in grp:
                absorbed_by, q = _resolved(files[g[1]], g[2])
                if absorbed_by >= 0:
                    fg = files[g[1]]
                    mid = (fg.adj[g[2]] + fg.adj[q]) / 2.0
                    side = 1.0 if fg.adj[g[2]] > fg.adj[q] else -1.0
                    grp = [h_ for h_ in grp if (h_[0] - mid) * side > 0]
                    break
            if absorbed_by < 0 or not grp:
                continue
        decided.append([grp, absorbed_by])
    # one line the merge absorbed is one row: the released regular bin and the released bin of vote losers
    # beside the same merged row, within the window of each other, are joined
    rel = sorted((d_ for d_ in decided if d_[1] >= 0), key=lambda d_: (d_[1], float(np.median([g[0] for g in d_[0]]))))
    joined = []
    for d_ in rel:
        if joined and joined[-1][1] == d_[1] and _near(np.array([float(np.median([g[0] for g in joined[-1][0]]))]),
                                                       float(np.median([g[0] for g in d_[0]])), tol_ppm) >= 0:
            joined[-1][0] = joined[-1][0] + d_[0]
        else:
            joined.append([list(d_[0]), d_[1]])
    decided = [d_ for d_ in decided if d_[1] < 0] + joined
    for grp, absorbed_by in decided:
        grp = sorted(grp, key=lambda t: (t[0], t[1]))
        if not any((np.isfinite(g[4]) and g[4] >= bright_x) or g[5] for g in grp):
            continue
        raw = np.array([files[g[1]].mz[g[2]] for g in grp])
        med_raw = float(np.median(raw))
        med_adj = float(np.median([g[0] for g in grp]))
        kinds = {g[3] for g in grp}
        kinds = kinds - {"absorbed"} if absorbed_by < 0 else kinds
        bright = [g for g in grp if (np.isfinite(g[4]) and g[4] >= bright_x) or g[5]]
        top = max(grp, key=lambda g: (files[g[1]].h[g[2]] if np.isfinite(files[g[1]].h[g[2]]) else -1, g[1]))
        ft, jt = files[top[1]], top[2]
        # roles: the files' roles at the bin (every row within the window), one count per row
        roles, present = {}, 0
        for sid_, f in files.items():
            lo, hi = f.window(med_raw, med_raw * tol_ppm * 1e-6)
            if hi > lo:
                present += 1
        for g in grp:
            rr = files[g[1]].role[g[2]]
            roles[rr] = roles.get(rr, 0) + 1
        # readings tried, over the files, deduplicated
        tried, seen = [], set()
        why_tried = False
        for g in grp:
            f = files[g[1]]
            i = f.idx[g[2]]
            led = f.led
            if g[3] == "unexplained":
                rd = _txt(led.at[i, BL.READING]) if BL.READING in led.columns else ""
                gt = _txt(led.at[i, BL.GATE]) if BL.GATE in led.columns else ""
                if _gate_kind(gt, rd) == "tried":
                    why_tried = True
                d = _reading_dict(rd, gt)
            elif g[3] == "absorbed":
                k, _ax = _merged_at(f, g[2])
                dppm = (f.mz[g[2]] - M_mz[k]) / M_mz[k] * 1e6 if k >= 0 else float("nan")
                d = {"ion": _txt(led.at[i, "ion_formula"]) if "ion_formula" in led.columns else None,
                     "neutral": _txt(led.at[i, "neutral_formula"]), "adduct": _txt(led.at[i, "adduct"]),
                     "blocked_by": (f"absorbed by the merge: the vote kept {M_txt[k]} {dppm:+.1f} ppm away"
                                    if k >= 0 else "absorbed by the merge")}
            elif g[3] == "vote_dropped":
                d = {"ion": _txt(led.at[i, "ion_formula"]) if "ion_formula" in led.columns else None,
                     "neutral": _txt(led.at[i, "neutral_formula"]), "adduct": _txt(led.at[i, "adduct"]),
                     "blocked_by": drop_reason.get((g[1], g[6]), REASON_UNRECORDED)}
                tr = _txt(led.at[i, "tier_reason"]) if "tier_reason" in led.columns else ""
                if "isotope contradiction:" in tr:
                    d["blocked_by"] += " | " + tr[tr.index("isotope contradiction:"):].split(" | ")[0]
            else:
                p = _parent_of(f, g[2])
                pt = _row_text(f, p) if p >= 0 else "no parent row in the file"
                d = {"ion": None, "neutral": None, "adduct": None,
                     "blocked_by": f"{f.role[g[2]]} of a parent not Assigned in the merged ledger ({pt})"}
            key = (d.get("neutral"), d.get("adduct"), BL.gate_class(d.get("blocked_by") or "")
                   if d.get("blocked_by") else "")
            if key in seen:
                continue
            seen.add(key)
            tried.append({k: (v or None) for k, v in d.items()})
        tried.sort(key=lambda d: (d.get("neutral") is None, str(d.get("neutral")), str(d.get("adduct")),
                                  str(d.get("blocked_by"))))
        if "vote_dropped" in kinds:
            why = "vote_dropped"
        elif absorbed_by >= 0:
            why = "absorbed_by_merge"
        elif "orphan_iso_child" in kinds:
            why = "orphan_iso_child"
        elif "orphan_artifact" in kinds:
            why = "orphan_artifact"
        else:
            why = "all_blocked" if why_tried else "no_formula"
        # partner: an orphan's parent, else a brighter neighbour / isotopologue parent
        pmz, prole = float("nan"), ""
        orph = [g for g in grp if g[3].startswith("orphan")]
        if orph:
            go = max(orph, key=lambda g: files[g[1]].h[g[2]] if np.isfinite(files[g[1]].h[g[2]]) else -1)
            p = _parent_of(files[go[1]], go[2])
            if p >= 0:
                pmz, prole = float(files[go[1]].mz[p]), _row_text(files[go[1]], p)
        if absorbed_by >= 0 and why == "absorbed_by_merge":
            pmz, prole = float(M_mz[absorbed_by]), f"merged {M_txt[absorbed_by]}"
        if not prole:
            pmz, prole = _partner(ft, jt)
        cs = [c_ for c_ in (_c13_count(files[g[1]], g[2]) for g in bright) if np.isfinite(c_)]
        first = last = ""
        if ts_mz is not None:
            lo = int(np.searchsorted(ts_mz, med_adj - med_adj * tol_ppm * 1e-6, "left"))
            hi = int(np.searchsorted(ts_mz, med_adj + med_adj * tol_ppm * 1e-6, "right"))
            if hi > lo:
                tt = ts_t[lo:hi]
                tt = tt[~pd.isna(tt)]
                if len(tt):
                    first, last = str(pd.Timestamp(tt.min()).isoformat()), str(pd.Timestamp(tt.max()).isoformat())
        xs = [g[4] for g in grp if np.isfinite(g[4])]
        hs = [files[g[1]].h[g[2]] for g in grp if np.isfinite(files[g[1]].h[g[2]])]
        out.append({
            "mz": med_raw, "mz_corrected": med_raw if applied else np.nan,
            "n_files_present": int(present), "n_files_selected": int(n_sel),
            "h_max": float(max(hs)) if hs else np.nan, "x_edge_max": round(float(max(xs)), 2) if xs else np.nan,
            "roles": ";".join(f"{k}:{v}" for k, v in sorted(roles.items(), key=lambda kv: (-kv[1], kv[0]))),
            "c13_count": round(float(np.median(cs)), 1) if cs else np.nan,
            "s34_line": _line_state(ft, jt, ISO.D_34S, ISO.ISOTOPE_RATIO["34S"]),
            "cl37_line": _line_state(ft, jt, ISO.D_37CL, ISO.ISOTOPE_RATIO["37Cl"]),
            "readings_tried": json.dumps(tried, sort_keys=True),
            "partner_mz": round(pmz, 5) if np.isfinite(pmz) else np.nan, "partner_role": prole,
            "why": why, "first_seen": first, "last_seen": last,
            "episode": bool(any(g[5] for g in grp)),
            "_srcs": ",".join(sorted({g[1] for g in grp})),
        })
    if not out:
        return empty_table()
    t = pd.DataFrame(out)
    if applied:
        from peaky.batch.assign_batch import to_server_axis
        try:
            frame = ((axis_info or {}).get("wave") or {}).get("model") == "frame"
            t["mz"] = to_server_axis(t["mz_corrected"].to_numpy(float), axis_info,
                                     t["_srcs"].tolist() if frame else None)
        except Exception:  # noqa: BLE001 -- an axis record the inverse cannot read: the ledgers' axis, said so
            t["mz"] = t["mz_corrected"]
    t["mz"] = t["mz"].astype(float).round(5)
    t["mz_corrected"] = pd.to_numeric(t["mz_corrected"], errors="coerce").round(5)
    t = t.drop(columns="_srcs").sort_values(["x_edge_max", "mz"], ascending=[False, True], kind="mergesort",
                                             na_position="last")
    return t.reset_index(drop=True)[list(TABLE_COLUMNS)]


def summary(table: pd.DataFrame, drops: pd.DataFrame | None) -> dict:
    """batch_summary['unidentified']: the table's row count by `why` and the drop list's by reason class."""
    by = table["why"].value_counts().to_dict() if len(table) else {}
    dr = {}
    if drops is not None and len(drops):
        dr = drops["reason"].map(lambda s: str(s).split(" (")[0]).value_counts().to_dict()
    return {"rows": int(len(table)), "by_why": {k: int(by.get(k, 0)) for k in WHY_ORDER},
            "bright_x": BRIGHT_X, "vote_drops": int(len(drops)) if drops is not None else 0,
            "vote_drops_by_reason": {str(k): int(v) for k, v in dr.items()}}


#: the table is written gzip-compressed (unidentified.csv.gz) past this many bytes
GZIP_OVER_BYTES = 50 * 1024 * 1024


def write(table: pd.DataFrame, tables_dir: str) -> str:
    """Write ``table`` as tables/unidentified.csv; past GZIP_OVER_BYTES as
    tables/unidentified.csv.gz instead. Returns the path written."""
    import os
    path = os.path.join(tables_dir, "unidentified.csv")
    table.to_csv(path, index=False)
    if os.path.getsize(path) > GZIP_OVER_BYTES:
        gz = path + ".gz"
        table.to_csv(gz, index=False, compression="gzip")
        os.remove(path)
        return gz
    return path


def build_single(ledger: pd.DataFrame, noise_edge_cps, *, sample_id: str = "file",
                 tol_ppm: float = 6.0) -> pd.DataFrame:
    """The table of ONE sample's ledger (the single-sample workbook's sheet): the
    file is its own batch, its committed M0 rows its own merged ledger (an iso
    child / artifact is an orphan when its parent is not Assigned in it), no
    vote and no series."""
    led = ledger
    m0 = led[led["role"].astype(str) == "M0"] if len(led) else led
    merged = pd.DataFrame({"mz": pd.to_numeric(m0["mz"], errors="coerce").to_numpy(float) if len(m0) else [],
                           "neutral_formula": m0["neutral_formula"].to_numpy() if len(m0) else [],
                           "adduct": m0["adduct"].to_numpy() if len(m0) else [],
                           "tier": m0["tier"].astype(str).to_numpy() if len(m0) else []})
    return build({str(sample_id): led}, {str(sample_id): {"noise_edge_cps": noise_edge_cps}}, merged,
                 tol_ppm=tol_ppm)
