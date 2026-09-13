"""Two-panel mass-defect / mass-error QC figure -- the calibration-and-coverage
diagnostic that complements the GKA findings figure.

Where the GKA figure reads the MERGED (M0-only) ledger, this one is sourced from
a FULL per-sample ledger (every role, including the iso_child satellites and the
UNEXPLAINED residual) so it shows the whole spectrum at once:

  Panel (a) mass-defect (mz - round(mz)) vs m/z -- the classic Kendrick-free
    mass-defect map. Five tier/role categories, each its own colour + marker:
    Assigned·parent (M0), Assigned·iso-child, Candidate·parent, Candidate·iso-child,
    and the grey UNEXPLAINED cloud. The assigned points trace the CH/CHO band; the
    grey cloud is where the unexplained signal sits relative to it.

  Panel (b) ppm mass-error vs m/z -- Assigned + Candidate M0 only (the rows that
    carry a real ppm_error), coloured by tier, with a 0-line and a linear trend.
    A sloped or offset trend is a calibration-drift diagnostic (the same
    instrument-accuracy read the tier engine self-calibrates from).

`split_categories` is a pure data function (no plotting); `render_qc` lazily
imports matplotlib and writes a standalone PNG. Determinism: the figure is a pure
function of the ledger, and matplotlib stamps the fixed SOURCE_DATE_EPOCH
(pipeline.CONTENT_EPOCH), so a re-run yields a byte-identical PNG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.assignment import tiers as T

__version__ = "0.1.0"

INK = "#222222"
GREY = "#777777"
UNEXPL_COLOR = "#B4B2A9"

# Tier/role categories for panel (a). Order = legend order + draw order (the grey
# unexplained cloud is drawn FIRST, underneath, so the coloured assignments sit on
# top). (key, label, colour, marker, point size, zorder).
TIER_ASSIGNED = T.TIER_ASSIGNED        # "Assigned"
TIER_CANDIDATE = T.TIER_CANDIDATE      # "Candidate"

CATEGORIES: list[tuple[str, str, str, str, float, int]] = [
    ("unexplained",     "unexplained",          UNEXPL_COLOR, ".", 7,  1),
    ("assigned_parent", "Assigned · M0",        "#1D9E75",    "o", 16, 4),
    ("assigned_iso",    "Assigned · iso-child", "#9AD1BE",    "v", 11, 3),
    ("cand_parent",     "Candidate · M0",       "#E0A93B",    "s", 16, 4),
    ("cand_iso",        "Candidate · iso-child", "#EBD0A0",   "^", 11, 3),
]

# ppm-error panel: tier -> colour (Assigned + Candidate M0 only).
PPM_TIER_COLORS = {TIER_ASSIGNED: "#1D9E75", TIER_CANDIDATE: "#E0A93B"}


# ---------------------------------------------------------------------------
# data: split a FULL per-sample ledger into the five tier/role categories
# ---------------------------------------------------------------------------
def _tier_of_parent(ledger: pd.DataFrame) -> dict:
    """{peak_id -> tier} over the M0 owners (the tier lives only on M0 rows). Used
    to give each iso_child the tier of the parent it points at."""
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    return dict(zip(m0["peak_id"], m0.get("tier", pd.Series(index=m0.index)).astype(str)))


def _norm_tier(v) -> str:
    """A ledger 'tier' cell -> 'Assigned' | 'Candidate' | ''. Tolerates the legacy
    'Identified' spelling (pre-0.5.0 ledgers) so an old per-file ledger still maps
    its M0 owners onto the Assigned category."""
    s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
    if s == "Identified":          # legacy spelling, pre tier rename
        return TIER_ASSIGNED
    return s if s in (TIER_ASSIGNED, TIER_CANDIDATE) else ""


def mass_defect(mz) -> np.ndarray:
    """Mass defect = mz - round(mz), in [-0.5, 0.5)."""
    m = np.asarray(mz, float)
    return m - np.round(m)


def split_categories(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a FULL per-sample ledger (all roles) into the five panel-(a) categories.

    Each value is a frame with at least [mz, md] (md = mass defect). An iso_child
    inherits the tier of its M0 parent; a parent/child whose tier is neither
    Assigned nor Candidate (e.g. an old uncalibrated row) is dropped from the
    coloured sets but the unexplained cloud is unaffected. Returns a dict keyed by
    the CATEGORIES keys; missing categories map to an empty frame."""
    out: dict[str, pd.DataFrame] = {k: ledger.iloc[0:0] for k, *_ in CATEGORIES}
    if "mz" not in ledger.columns or "role" not in ledger.columns or not len(ledger):
        return out
    df = ledger.copy()
    df["md"] = mass_defect(df["mz"])
    parent_tier = _tier_of_parent(df)

    # unexplained (grey cloud) -- the residual peak-by-peak
    out["unexplained"] = df[df["role"] == L.ROLE_UNEXPLAINED]

    # M0 owners, split by tier
    m0 = df[df["role"] == L.ROLE_M0].copy()
    m0["_t"] = m0.get("tier", pd.Series(index=m0.index)).map(_norm_tier)
    out["assigned_parent"] = m0[m0["_t"] == TIER_ASSIGNED]
    out["cand_parent"] = m0[m0["_t"] == TIER_CANDIDATE]

    # iso_child satellites, split by the tier of the parent they point at
    iso = df[df["role"] == L.ROLE_ISO].copy()
    if len(iso) and "parent_peak_id" in iso.columns:
        iso["_t"] = iso["parent_peak_id"].map(
            lambda p: _norm_tier(parent_tier.get(p)))
        out["assigned_iso"] = iso[iso["_t"] == TIER_ASSIGNED]
        out["cand_iso"] = iso[iso["_t"] == TIER_CANDIDATE]
    return out


def ppm_points(ledger: pd.DataFrame) -> pd.DataFrame:
    """The panel-(b) points: Assigned + Candidate M0 rows that carry a finite mass
    error. Columns [mz, ppm_error, tier] (tier normalised). Prefers the CALIBRATED
    error `ppm_error_cal` (per-file offset removed; tiers.stamp_calibrated_ppm) when
    present, falling back to the raw `ppm_error`; the chosen basis is flagged in
    ``.attrs['calibrated']``. Empty when the ledger has no calibrated M0 rows."""
    cols = ["mz", "ppm_error", "tier"]
    if not all(c in ledger.columns for c in ("mz", "ppm_error", "role")):
        return pd.DataFrame(columns=cols)
    val = "ppm_error"
    if "ppm_error_cal" in ledger.columns \
            and pd.to_numeric(ledger["ppm_error_cal"], errors="coerce").notna().any():
        val = "ppm_error_cal"
    m0 = ledger[ledger["role"] == L.ROLE_M0].copy()
    m0["tier"] = m0.get("tier", pd.Series(index=m0.index)).map(_norm_tier)
    m0 = m0[m0["tier"].isin((TIER_ASSIGNED, TIER_CANDIDATE))]
    m0 = m0[pd.to_numeric(m0[val], errors="coerce").notna()]
    if not len(m0):
        out = pd.DataFrame(columns=cols)
        out.attrs["calibrated"] = (val == "ppm_error_cal")
        return out
    m0["ppm_error"] = pd.to_numeric(m0[val], errors="coerce")
    out = m0[cols].reset_index(drop=True)
    out.attrs["calibrated"] = (val == "ppm_error_cal")
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def render_qc(ledger: pd.DataFrame, path: str, *, title: str = "", dpi: int = 150
              ) -> str:
    """Render the two-panel mass-defect / mass-error QC figure to `path` (PNG).

    `ledger` is a FULL per-sample ledger (every role). Panel (a) is the mass-defect
    map over the five tier/role categories; panel (b) is the ppm mass-error of the
    Assigned + Candidate M0 rows with a 0-line and a linear trend. Empty categories
    simply do not draw; a ledger with no calibrated M0 rows still renders panel (a)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = split_categories(ledger)
    pts = ppm_points(ledger)

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(8.3, 8.6))
    # top kept low so the panel (a) title clears the suptitle + subtitle stacked
    # above it (they collided at top=0.91); bbox_inches="tight" at save captures
    # the right-hand legends and left y-labels.
    fig.subplots_adjust(left=0.095, right=0.78, top=0.875, bottom=0.07, hspace=0.30)

    # --- panel (a): mass defect vs m/z, by tier/role category ----------------
    n_drawn = 0
    for key, label, col, marker, size, z in CATEGORIES:
        g = cats.get(key)
        if g is None or not len(g):
            continue
        alpha = 0.40 if key == "unexplained" else 0.9
        edge = "none" if key == "unexplained" else "white"
        axa.scatter(g["mz"], g["md"], s=size, c=col, marker=marker, alpha=alpha,
                    linewidths=0.0 if key == "unexplained" else 0.35,
                    edgecolors=edge, zorder=z, label=f"{label}  (n={len(g)})")
        n_drawn += len(g)
    axa.axhline(0, color="0.8", lw=0.7, zorder=0)
    axa.set_xlabel("m/z", fontsize=9)
    axa.set_ylabel("mass defect  (m/z − round)", fontsize=9)
    axa.set_title("(a) mass defect vs m/z — by assignment tier / role",
                  loc="left", fontsize=10.5, color=INK)
    axa.grid(alpha=0.22)
    axa.tick_params(labelsize=8)
    if n_drawn:
        axa.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   framealpha=0.95, markerscale=1.3, borderaxespad=0.0)

    # --- panel (b): ppm mass error vs m/z, Assigned+Candidate M0 + trend ------
    axb.axhline(0, color="#444", lw=0.8, zorder=1)
    for tier in (TIER_ASSIGNED, TIER_CANDIDATE):
        g = pts[pts["tier"] == tier]
        if not len(g):
            continue
        axb.scatter(g["mz"], g["ppm_error"], s=15, c=PPM_TIER_COLORS[tier],
                    marker="o", alpha=0.85, linewidths=0.3, edgecolors="white",
                    zorder=3, label=f"{tier}  (n={len(g)})")
    trend_txt = None
    if len(pts) >= 2:
        x = pts["mz"].to_numpy(float)
        y = pts["ppm_error"].to_numpy(float)
        if np.ptp(x) > 0:                       # a fit needs spread in m/z
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            axb.plot(xs, slope * xs + intercept, color="#B00020", lw=1.2, ls="--",
                     zorder=4, label="linear trend")
            trend_txt = (f"trend: {slope * 1000:+.2f} mppm/Th · "
                         f"intercept {intercept:+.2f} ppm · median {np.median(y):+.2f}")
    calibrated = bool(pts.attrs.get("calibrated"))
    cal_mu = ledger.attrs.get("cal_mu")
    if calibrated and cal_mu is not None and trend_txt is not None:
        trend_txt += f" · offset removed {cal_mu:+.2f} ppm (raw median)"
    axb.set_xlabel("m/z", fontsize=9)
    axb.set_ylabel("mass error (ppm)" + (" — calibrated" if calibrated else ""),
                   fontsize=9)
    axb.set_title("(b) ppm mass error vs m/z — Assigned + Candidate "
                  + ("(per-file offset removed)" if calibrated
                     else "(calibration drift)"),
                  loc="left", fontsize=10.5, color=INK)
    axb.grid(alpha=0.22)
    axb.tick_params(labelsize=8)
    if trend_txt:
        axb.text(0.015, 0.04, trend_txt, transform=axb.transAxes, fontsize=7.3,
                 va="bottom", color="#555")
    if len(pts):
        axb.legend(fontsize=7.8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   framealpha=0.95, borderaxespad=0.0)

    fig.suptitle(title or "Mass-defect & mass-error QC", x=0.095, y=0.975,
                 ha="left", fontsize=12.5, weight="bold", color=INK)
    fig.text(0.095, 0.945,
             "from the brightest selected sample's full ledger (all roles)",
             ha="left", fontsize=7.8, color=GREY)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
