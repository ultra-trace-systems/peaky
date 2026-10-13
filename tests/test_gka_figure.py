"""Offline test for gka_figure.py — series detection, KMD math, and a render
smoke test from a synthetic ledger. Run: python3 tests/test_gka_figure.py"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import gka_figure as GF  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# a clean CH2 ladder + a separate O-addition ladder, plus a lone peak
CH2_CHAIN = ["C5H10O2", "C6H12O2", "C7H14O2", "C8H16O2", "C9H18O2"]
O_CHAIN = ["C10H16O2", "C10H16O3", "C10H16O4", "C10H16O5"]
# Si: D3->D4 IS a 2-rung C2H6OSi ladder (short series) -> siloxane shown.
# F-monsters (assorted high-F, no anchor, not PFCA) -> EXCLUDED from the GKA.
# PFCAs (real F, a CF2 ladder) -> KEPT -> fluorinated shown.
SI = ["C6H18O3Si3", "C8H24O4Si4", "C5H11NO3Si"]
FL = ["C3H2F6O", "C12H12F12", "C14H12F8"]            # F-monsters -> excluded
PFCA = ["C2HF3O2", "C3HF5O2", "C4HF7O2"]             # real PFAS, CF2 ladder -> kept
LEDGER = pd.DataFrame(
    [dict(role="M0", neutral_formula=f) for f in CH2_CHAIN + O_CHAIN + SI + FL + PFCA]
    + [dict(role="iso_child", neutral_formula="C5H10O2")]   # must be ignored (not M0)
)

fmass = GF._neutral_masses(LEDGER)
check("_neutral_masses: drops non-M0 + F-monsters / dedups",
      len(fmass) == len(set(CH2_CHAIN + O_CHAIN + SI + PFCA))
      and not any(f in fmass for f in FL), len(fmass))

# an F-monster the ledger did NOT refuse (read off its CF2 ladder: no below flag,
# or Assigned, which a refused row never is) is drawn; a refused one is not
LADDERED = pd.DataFrame([
    dict(role="M0", neutral_formula="C6H5F7O2", tier="Candidate", below_assignability=False),   # lifted
    dict(role="M0", neutral_formula="C7H5F9O2", tier="Assigned", below_assignability=False),    # full ladder
    dict(role="M0", neutral_formula="C12H12F12", tier="Candidate", below_assignability=True),   # refused
    dict(role="M0", neutral_formula="C14H12F8", tier="Candidate", below_assignability="True"),  # refused (csv)
])
_lm = GF._neutral_masses(LADDERED)
check("_neutral_masses: draws an F-monster the ledger did not refuse, drops a refused one",
      set(_lm) == {"C6H5F7O2", "C7H5F9O2"}, sorted(_lm))
# the merged ledger carries the tier, not the flag: only an Assigned F-monster is drawn
_mm = GF._neutral_masses(LADDERED.drop(columns=["role", "below_assignability"]))
check("_neutral_masses: on a merged ledger an Assigned F-monster is drawn, a Candidate one is not",
      set(_mm) == {"C7H5F9O2"}, sorted(_mm))

# --- series detection ------------------------------------------------------
ch2 = [s for s in GF.detect_series(fmass, units=["CH2"], min_len=3)]
check("detect_series: finds the CH2 ladder", len(ch2) == 1 and ch2[0].length == 5,
      [(s.unit, s.length) for s in ch2])
check("detect_series: members ascend in mass",
      ch2 and ch2[0].masses == sorted(ch2[0].masses), ch2[0].masses if ch2 else None)

oser = [s for s in GF.detect_series(fmass, units=["O"], min_len=3)]
check("detect_series: finds the O ladder", len(oser) == 1 and oser[0].length == 4,
      [(s.unit, s.length) for s in oser])

check("detect_series: min_len filters short chains",
      GF.detect_series(fmass, units=["CH2"], min_len=6) == [],
      GF.detect_series(fmass, units=["CH2"], min_len=6))

# --- KMD math --------------------------------------------------------------
masses = np.array([fmass[f] for f in CH2_CHAIN])
k = GF.kmd(masses, "CH2")
check("kmd: a CH2 ladder is one horizontal row (constant KMD)",
      float(k.max() - k.min()) < 1e-6, float(k.max() - k.min()))
# O additions move the CH2-KMD (not a CH2 homology) -> not constant
ko = GF.kmd(np.array([fmass[f] for f in O_CHAIN]), "CH2")
check("kmd: an O ladder is NOT flat under the CH2 base",
      float(ko.max() - ko.min()) > 1e-3, float(ko.max() - ko.min()))

# --- family summary --------------------------------------------------------
fam = GF.family_summary(GF.detect_series(fmass, min_len=3))
byname = {f["family"]: f for f in fam}
check("family_summary: alkyl family present", byname["alkyl"]["n_series"] >= 1, byname["alkyl"])
check("family_summary: oxidation family present", byname["oxidation"]["n_series"] >= 1,
      byname["oxidation"])
check("family_summary: every FAMILY represented",
      set(byname) == {lab for lab, *_ in GF.FAMILIES}, set(byname))

# --- panel-show rule: a family shows ONLY if its longest ladder has > 3 members --
check("element_members: Si collected; F-monsters excluded, real PFCAs kept",
      set(GF.element_members(fmass, "Si")) == set(SI)
      and set(GF.element_members(fmass, "F")) == set(PFCA), GF.element_members(fmass, "F"))
check("detect_series: Si has only a short (2-rung) C2H6OSi ladder (not > 3)",
      GF.detect_series(fmass, units=["C2H6OSi"], min_len=4) == []
      and len(__import__("peaky.assignment.series_gka", fromlist=["find_homolog_series"])
              .find_homolog_series(GF.element_members(fmass, "Si"), "C2H6OSi", min_len=2)) >= 1)
shown = [f[0] for f in GF.present_families(fmass)]
check("present_families: siloxane DROPPED (longest C2H6OSi ladder is 2, not > 3)",
      "siloxane" not in shown, shown)
check("present_families: fluorinated DROPPED (longest PFCA CF2 ladder is 3, not > 3)",
      "fluorinated" not in shown, shown)
check("present_families: alkyl/oxidation SHOW (their ladders are > 3)",
      "alkyl" in shown and "oxidation" in shown, shown)
check("_longest_ladder: reports the longest base-unit ladder per family",
      GF._longest_ladder(fmass, GF.FAMILIES[0]) == 5            # alkyl CH2 chain
      and GF._longest_ladder(fmass, GF.FAMILIES[3]) == 2,       # siloxane D3->D4
      [GF._longest_ladder(fmass, f) for f in GF.FAMILIES])

# --- per-panel zoom: frame the drawn span padded ~10%, not the full mass axis --
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

_fig, _ax = plt.subplots()
GF._zoom_to(_ax, [100.0, 200.0], [-0.1, 0.1])
_x0, _x1 = _ax.get_xlim()
check("_zoom_to: x range is the drawn span padded ~10% each side",
      abs(_x0 - 90.0) < 1e-9 and abs(_x1 - 210.0) < 1e-9, (_x0, _x1))
plt.close(_fig)

_fig, _ax = plt.subplots()
_ax.set_xlim(50, 750)                              # a full-range axis to override
GF._zoom_to(_ax, [300.0], [0.0])                   # a single drawn point -> no-op
check("_zoom_to: no-op with < 2 drawn points", _ax.get_xlim() == (50.0, 750.0),
      _ax.get_xlim())
plt.close(_fig)

# a rendered alkyl panel must zoom to its CH2-chain span (~70-156 Da), not 50-750
_fig, _ax = plt.subplots()
_mass = np.array(list(fmass.values()))
GF._panel(_ax, _mass, fmass, "CH2", "#1D9E75", "alkyl", None,
          min_len=4, highlight_min_len=5, top_chains=10)
_x0, _x1 = _ax.get_xlim()
_chain = [fmass[f] for f in CH2_CHAIN]
check("_panel: alkyl panel is zoomed to its drawn CH2 ladder span",
      _x0 > 50 and _x1 < 750 and _x0 < min(_chain) and _x1 > max(_chain),
      (_x0, _x1, min(_chain), max(_chain)))
plt.close(_fig)

# --- render smoke test -----------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    png = f"{d}/gka.png"
    out = GF.render_gka(LEDGER, png, title="synthetic")
    check("render_gka: PNG created", os.path.exists(out) and os.path.getsize(out) > 5000,
          os.path.getsize(out) if os.path.exists(out) else "missing")
    check("render_gka: it is a PNG", open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n")

    # a merged-style ledger (no 'role' column) must also render
    merged = pd.DataFrame({"neutral_formula": CH2_CHAIN + O_CHAIN, "mz": range(9)})
    out2 = GF.render_gka(merged, f"{d}/gka2.png", title="merged")
    check("render_gka: works on a role-less merged ledger", os.path.exists(out2))

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
