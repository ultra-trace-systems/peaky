"""Static Generalized Kendrick Analysis (GKA) figure -- the print counterpart of
the interactive rotating-GKA widget (`scripts/gka_widget.py`).

The widget lets you rotate the Kendrick base live; for the PDF report we freeze
the canonical **CH2** base into a publishable Kendrick mass-defect (KMD) plot and
call out the longest confirmed homologous series. A CH2 addition leaves both the
KMD and the DBE unchanged, so a CH2 homologous series is a HORIZONTAL row of
constant colour -- the rows you see ARE the GKA finding. A second compact panel
summarises the strongest series found under every repeat unit (the structure the
other rotations of the widget would reveal: oxidation, alkoxylate, contaminants).

`detect_series` is a pure data function (no plotting); `render_gka` lazily imports
matplotlib and writes a standalone PNG. Built from one committed ledger -- the
merged batch ledger or a single-file ledger both work (it reads neutral_formula,
and tier / below_assignability where present).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L
from peaky.assignment import series_gka as G

__version__ = "0.1.1"   # an F-monster the ledger did not refuse (a CF2 ladder, fluorine.py) is drawn

INK = "#222222"
GREY = "#777777"

# A family earns a panel (and a series earns a connector line) only when its
# LONGEST drawn ladder has MORE than this many members. A >3 floor keeps the
# real alkyl/oxidation/alkoxylate ladders while dropping incidental 2-rung
# pairings (e.g. the siloxane D3->D4 longest-2 panel) that were never a series.
PANEL_MIN_MEMBERS = 3        # longest series must be > PANEL_MIN_MEMBERS
ZOOM_PAD = 0.10              # fraction of the drawn span padded around each panel

# Repeat-unit families. Each is one small-multiple panel: `base` is the unit the
# panel's Kendrick axis flattens (so that family's ladders run horizontal);
# `units` are the repeat units folded into the family's series/summary counts.
# `element` marks a CONTAMINANT family defined by an ELEMENT (Si / F): every
# element-bearing peak is highlighted, but the panel still shows only when the
# element forms a real ladder (longest > PANEL_MIN_MEMBERS under its base unit) —
# a lone D3->D4 siloxane pair is no longer surfaced. element=None -> a homology
# (ladder) family; the same longest-ladder>3 rule governs both kinds.
# (label, base unit, units folded in, colour, element)
FAMILIES: list[tuple[str, str, list[str], str, str | None]] = [
    ("alkyl",       "CH2",     ["CH2"],                            "#1D9E75", None),
    ("oxidation",   "O",       ["O", "CO", "CO2", "H2O", "C2H2O"], "#378ADD", None),
    ("alkoxylate",  "C2H4O",   ["C2H4O", "C3H6O"],                 "#7F77DD", None),
    ("siloxane",    "C2H6OSi", ["C2H6OSi"],                        "#BA7517", "Si"),
    ("fluorinated", "CF2",     ["CF2"],                            "#D4537E", "F"),
]


@dataclass
class Series:
    unit: str
    members: list[str]          # neutral formulas, ascending mass
    masses: list[float]

    @property
    def length(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# data: detect homologous series in the assigned neutral formulas
# ---------------------------------------------------------------------------
def _is_f_monster(cnt: dict) -> bool:
    """Unconfirmed-fluorine 'monster': F>=4, not a PFCA CnHF(2n-1)O2, and no Cl/Br/S
    anchor -- ¹⁹F is monoisotopic so these are mass coincidences (the formulas
    cleanup.demote_unconfirmed_fluorine reads; it refuses those its CF2 ladder does not
    lift). Excluded from the GKA, unless the ledger shows the row was not refused
    (`_neutral_masses`), so the fluorinated panel shows REAL PFAS, not the demoted
    coincidences."""
    nF, nC = cnt.get("F", 0), cnt.get("C", 0)
    if nF < 4:
        return False
    is_pfca = (nC >= 2 and cnt.get("H", 0) == 1 and cnt.get("O", 0) == 2 and nF == 2 * nC - 1)
    anchored = cnt.get("Cl", 0) or cnt.get("Br", 0) or cnt.get("S", 0)
    return not (is_pfca or anchored)


def _neutral_masses(ledger: pd.DataFrame) -> dict[str, float]:
    """{neutral_formula -> neutral monoisotopic mass} over the assigned M0 set,
    one entry per distinct neutral (deduped across adduct channels). Excludes
    unconfirmed-fluorine 'monsters' (the demoted coincidence fits) so the GKA's
    fluorinated panel shows the real PFAS (the PFCAs), not ¹⁹F mass coincidences --
    except a monster the ledger did NOT refuse: one read off its CF2 ladder
    (fluorine.py) carries no below_assignability flag, and a full ladder may reach
    Assigned, which a refused row never does (the refusal forces Candidate; the
    merged ledger carries the tier, not the flag)."""
    if "role" in ledger.columns:
        led = ledger[ledger["role"] == "M0"]
    else:
        led = ledger
    forms = led["neutral_formula"].dropna().astype(str)
    not_refused: set[str] = set()
    if "tier" in led.columns:
        not_refused |= set(forms[led.loc[forms.index, "tier"].astype(str) == "Assigned"])
    if "below_assignability" in led.columns:
        not_refused |= set(forms[~led.loc[forms.index, "below_assignability"].map(L._truthy).astype(bool)])
    out: dict[str, float] = {}
    for f in forms.unique():
        if not f or f == "nan":
            continue
        try:
            if _is_f_monster(C.parse_formula(f)) and f not in not_refused:
                continue
            out[f] = C.neutral_mass(f)
        except Exception:
            pass
    return out


def detect_series(formula_mass: dict[str, float], *, units=None, min_len: int = 4
                  ) -> list[Series]:
    """All homologous chains (>= min_len members) under each repeat unit, longest
    first. Chains that are a subset of a longer chain under the same unit are
    already excluded by `find_homolog_series` (it walks maximal runs)."""
    units = units or [u for fam in FAMILIES for u in fam[2]]
    out: list[Series] = []
    for u in units:
        for chain in G.find_homolog_series(formula_mass, u, min_len=min_len):
            out.append(Series(unit=u, members=chain,
                              masses=[formula_mass[m] for m in chain]))
    out.sort(key=lambda s: s.length, reverse=True)
    return out


def _family_of(unit: str) -> tuple[str, str]:
    for label, base, units, col, element in FAMILIES:
        if unit in units:
            return label, col
    return unit, "#888888"


def family_summary(series: list[Series]) -> list[dict]:
    """Aggregate detected series into families: how many series and how many
    distinct member peaks each repeat-unit family explains."""
    rows = []
    for label, base, units, col, element in FAMILIES:
        ss = [s for s in series if s.unit in units]
        members = {m for s in ss for m in s.members}
        rows.append({"family": label, "color": col, "n_series": len(ss),
                     "n_members": len(members), "longest": max((s.length for s in ss), default=0)})
    return rows


def element_members(formula_mass: dict[str, float], element: str) -> dict[str, float]:
    """{formula -> mass} for every neutral that CONTAINS `element` (Si/F...)."""
    return {f: m for f, m in formula_mass.items()
            if C.parse_formula(f).get(element, 0) > 0}


def _longest_ladder(formula_mass: dict[str, float], fam: tuple) -> int:
    """Members in the longest ladder of `fam`'s base unit. For an element family
    the ladder is searched over the element-bearing neutrals only (the set the
    panel highlights); for an organic family over all neutrals."""
    label, base, units, col, element = fam
    pool = element_members(formula_mass, element) if element else formula_mass
    ladders = G.find_homolog_series(pool, base, min_len=2)
    return max((len(c) for c in ladders), default=0)


def present_families(formula_mass: dict[str, float], *,
                     min_members: int = PANEL_MIN_MEMBERS) -> list[tuple]:
    """The families worth a panel: a family is shown only if its LONGEST ladder
    has MORE than `min_members` members (default >3). One unified rule for
    organic and contaminant (element) families — an incidental 2-rung pairing
    (e.g. the siloxane D3->D4 longest-2 panel) is no longer plotted, while a
    real >=4-rung ladder still is. A scattered element-bearing set with no
    series is likewise dropped (user 2026-06-20: "if there is no series
    shouldn't plot in GKA")."""
    return [fam for fam in FAMILIES
            if _longest_ladder(formula_mass, fam) > min_members]


# ---------------------------------------------------------------------------
# KMD math (base CH2 by default)
# ---------------------------------------------------------------------------
def kmd(mass, base: str = "CH2") -> np.ndarray:
    """Kendrick mass defect of `mass` on `base`: KMD = round(KM) - KM, where
    KM = mass * nominal(base) / exact(base). CH2 homologs share one KMD value."""
    exact = G.unit_mass(base)
    nominal = round(exact)
    km = np.asarray(mass, float) * nominal / exact
    return np.round(km) - km


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _zoom_to(ax, xs: list[float], ys: list[float], *, pad: float = ZOOM_PAD) -> None:
    """Frame the panel on its DRAWN (colored) points, padded by `pad` of the span
    on each side, so panels stop spanning the full 50–750 Da when the family
    clusters in a narrow window. No-op when fewer than two distinct points exist
    (matplotlib's autoscale already gives a sensible view)."""
    if len(xs) < 2:
        return
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) * pad or 1.0
    dy = (y1 - y0) * pad or (abs(y0) * pad or 1e-3)
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def _panel(ax, mass: np.ndarray, fmass: dict, base: str, color: str, label: str,
           element: str | None = None, *, min_len: int, highlight_min_len: int,
           top_chains: int) -> tuple[int, int]:
    """Draw one family small-multiple over a grey KMD cloud at `base`, then ZOOM
    to the bounding box of the family's DRAWN (colored) points (~10% pad).

    Homology family (element=None): connect the longest ladders OF THE BASE UNIT —
    every highlighted ladder runs horizontal (only the base unit flattens, so we
    never draw a tilted line). Returns (#ladders, #peaks).
    Contaminant family (element set): highlight EVERY element-bearing peak and
    connect its ladders. Returns (#ladders, #element-bearing peaks).
    Connector lines are drawn only for ladders with > PANEL_MIN_MEMBERS members."""
    y = kmd(mass, base)
    ax.scatter(mass, y, s=5, c="#CBC9C0", alpha=0.45, linewidths=0, zorder=1)
    ax.set_title(f"{label} · base {base}", fontsize=9.5, loc="left", color=color)
    drawn_x: list[float] = []        # colored points -> the zoom bounding box
    drawn_y: list[float] = []

    if element:
        em = element_members(fmass, element)
        if em:
            ex = np.array(list(em.values()))
            ey = kmd(ex, base)
            ax.scatter(ex, ey, s=20, c=color, alpha=0.9,
                       linewidths=0.4, edgecolors="white", zorder=3)
            drawn_x.extend(ex.tolist()); drawn_y.extend(np.atleast_1d(ey).tolist())
        ladders = G.find_homolog_series(em, base, min_len=2)
        for chain in (ladders if top_chains is None else ladders[:top_chains]):
            if len(chain) <= PANEL_MIN_MEMBERS:      # connect only real series (>3)
                continue
            xs = np.array([em[m] for m in chain])
            ax.plot(xs, kmd(xs, base), color=color, lw=1.0, alpha=0.85, zorder=2)
        longest = max((len(c) for c in ladders), default=0)
        note = f"{len(em)} {element}-bearing · {len(ladders)} ladder(s)"
        if longest:
            note += f" · longest {longest}"
        ax.text(0.015, 0.965, note, transform=ax.transAxes, fontsize=7, va="top", color="#444")
        ax.tick_params(labelsize=7.5); ax.grid(alpha=0.22)
        _zoom_to(ax, drawn_x, drawn_y)
        return len(ladders), len(em)

    # connect+mark every series with > PANEL_MIN_MEMBERS members (the new uniform
    # connector rule), longest first. `highlight_min_len` is unused now that the
    # >3 gate governs which series get a connector; kept in the signature for
    # back-compat with callers that still pass it.
    _drawable = detect_series(fmass, units=[base], min_len=PANEL_MIN_MEMBERS + 1)
    for s in (_drawable if top_chains is None else _drawable[:top_chains]):
        xs = np.array(s.masses)
        ys = kmd(xs, base)
        ax.plot(xs, ys, color=color, lw=1.0, alpha=0.9, zorder=2,
                marker="o", ms=3.0, mfc=color, mec="white", mew=0.35)
        drawn_x.extend(xs.tolist()); drawn_y.extend(np.atleast_1d(ys).tolist())
    alls = detect_series(fmass, units=[base], min_len=min_len)
    npk = len({m for s in alls for m in s.members})
    longest = max((s.length for s in alls), default=0)
    _zoom_to(ax, drawn_x, drawn_y)
    if alls:
        # be honest that only the top `top_chains` series get connector lines (the
        # rest of the counted series remain as the grey/colored KMD dots) — else a
        # "44 series" subtitle on a panel showing ~10 ladders reads as a bug.
        shown = (f"{len(alls)} series" if (top_chains is None or len(alls) <= top_chains)
                 else f"top {min(len(alls), top_chains)} of {len(alls)} series shown")
        ax.text(0.015, 0.965, f"{shown} · {npk} peaks · longest {longest}",
                transform=ax.transAxes, fontsize=7, va="top", color="#444")
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.22)
    return len(alls), npk


def render_gka(ledger: pd.DataFrame, path: str, *, min_len: int = 4,
               highlight_min_len: int = 5, top_chains: int = 10,
               title: str = "", dpi: int = 150) -> str:
    """Render the GKA findings figure to `path` (PNG): a small-multiple grid of
    Kendrick mass-defect plots, one per repeat-unit family that is ACTUALLY found
    (families with no series are dropped), each rotated to its own base so that
    family's homologous series flatten into horizontal ladders. The static
    counterpart of the rotating-GKA widget. A final cell rolls up the per-family
    peak counts.
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fmass = _neutral_masses(ledger)
    mass = np.array(list(fmass.values()))

    # one unified rule: a family earns a panel only if its longest ladder has
    # > PANEL_MIN_MEMBERS members (>3). Drops incidental 2-rung pairings (e.g. the
    # siloxane D3->D4 longest-2 panel) while keeping real >=4-rung ladders.
    panels = present_families(fmass)

    ncols = 2
    ncells = len(panels) + 1                              # + family-rollup cell
    nrows = max(1, math.ceil(ncells / ncols))
    H = 1.6 + 3.0 * nrows
    fig = plt.figure(figsize=(8.3, H))
    gs = GridSpec(nrows, ncols, hspace=0.42, wspace=0.24,
                  left=0.085, right=0.97, top=1 - 1.05 / H, bottom=0.42 / H)
    cells = [gs[i // ncols, i % ncols] for i in range(ncells)]

    rollup = []
    for cell, (label, base, units, col, element) in zip(cells, panels):
        ax = fig.add_subplot(cell)
        nser, npk = _panel(ax, mass, fmass, base, col, label, element,
                           min_len=min_len, highlight_min_len=highlight_min_len,
                           top_chains=top_chains)
        ax.set_xlabel("neutral mass (Da)", fontsize=8)
        ax.set_ylabel(f"KMD ({base})", fontsize=8)
        rollup.append((label, col, nser, npk))

    # rollup cell: per-family base-unit peak counts (consistent with the panels)
    ax2 = fig.add_subplot(cells[len(panels)])
    yy = np.arange(len(rollup))[::-1]
    ax2.barh(yy, [r[3] for r in rollup], color=[r[1] for r in rollup], alpha=0.85)
    ax2.set_yticks(yy); ax2.set_yticklabels([r[0] for r in rollup], fontsize=8)
    xmax = max((r[3] for r in rollup), default=1)
    for yi, r in zip(yy, rollup):
        ax2.text(r[3] + xmax * 0.02, yi, f"{r[2]}×", va="center", fontsize=7, color="#555")
    ax2.set_xlim(0, xmax * 1.28)
    ax2.set_xlabel("highlighted peaks (ladder members / element-bearing)", fontsize=8)
    ax2.set_title("Family rollup (peaks · #ladders)", fontsize=9.5, loc="left")
    ax2.tick_params(labelsize=7.5)

    sub = ("Each panel flattens its family's homologous series into horizontal ladders "
           "and zooms to its drawn span; a family is shown only if its longest ladder "
           "has > 3 members (Si/F highlighted by element)")
    fig.text(0.085, 1 - 0.40 / H, title or "GKA homologous-series findings",
             fontsize=12.5, weight="bold", ha="left", color=INK)
    fig.text(0.085, 1 - 0.62 / H, sub, fontsize=7.6, ha="left", color=GREY)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
