"""Known answer, offline: malathion is found and NAMED from the pinned EPACONS
snapshot alone, with peaky's hard-coded organophosphate dictionaries removed.

What the run has to do without a list of P/S formulas in code: the CHON grid
cannot reach C10H19O6PS2 (P is off the grid, S2 above its cap); pass 7
(certified-neutral discovery) must certify the neutral from three ion channels
converging on one core mass and open the P/S box for it; the active list's name
then lifts the pinned reading to 3c through a NAMED context-list entry
(`reflist:<id> = Malathion`), and the run's `lists` record names the snapshot
it read (id, version, sha256, pinned). The same test with the list off is the
control: found (pass 7) but unnamed, so never 3c.

The synthetic spectrum carries each malathion ion with its full predicted
isotope envelope (the evidence scale's own-isotope check rejects a reading
whose predicted minor lines are absent) and a CHO backbone of 26 neutrals with
13C lines, so the file calibrates and the per-file evidence stage levels it
(an Orbitrap-class width model, R = 140 000 at m/z 200).
"""
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import reflists as RL
from peaky.assignment.passes import config as PCfg
from peaky.assignment.passes import directors as D
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.chem import profiles as P
from peaky.lists.importers import epacons as E

MALATHION = "C10H19O6PS2"
ION = {"[M+H]+": "C10H20O6PS2+", "[M+NH4]+": "C10H23NO6PS2+", "[M+(CH4N2O)H]+": "C11H24N2O7PS2+"}
HEIGHT = {"[M+H]+": 3.5e5, "[M+NH4]+": 5.5e4, "[M+(CH4N2O)H]+": 2.5e5}
D13C = 1.0033548
BACKBONE = ["C6H10O", "C7H12O", "C8H14O", "C9H16O", "C10H16O", "C10H16O2", "C10H14O2", "C9H14O2", "C8H12O2",
            "C7H10O2", "C11H18O2", "C12H20O2", "C13H22O2", "C10H18O3", "C11H20O3", "C12H22O3", "C9H16O3",
            "C8H14O3", "C13H24O4", "C14H26O4", "C15H28O4", "C10H20O4", "C11H22O4", "C12H24O4", "C6H12O2", "C7H14O2"]


def spectrum() -> pd.DataFrame:
    rows = []
    for add_, h in HEIGHT.items():
        mz = C.ion_mz(MALATHION, add_)
        rows.append((mz, h))
        for shift, rel, _lab in ISO.isotope_pattern(ION[add_], min_rel=0.0005, merge_da=0.0005):
            rows.append((mz + shift, h * rel))
    for i, f in enumerate(BACKBONE):
        n_c = C.parse_formula(f)["C"]
        for add_, frac in (("[M+H]+", 1.0), ("[M+(CH4N2O)H]+", 0.3)):
            mz, h = C.ion_mz(f, add_), (2e5 - i * 3e3) * frac
            rows.append((mz, h))
            rows.append((mz + D13C, h * 0.0107 * (n_c + (1 if "CH4N2O" in add_ else 0))))
    return pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(rows))],
                         "mz": [r[0] for r in rows], "height": [r[1] for r in rows]})


@pytest.fixture
def no_p_dictionaries(monkeypatch):
    orig = D._known_species

    def without(polarity="negative", context=None):
        reg = dict(orig(polarity, context))
        reg.pop("organophosphate", None)
        reg.pop("organothiophosphate", None)
        return reg

    monkeypatch.setattr(D, "_known_species", without)
    RL.clear_catalog_cache()
    yield
    RL.clear_catalog_cache()


def run(include):
    cfg = PCfg.PassConfig(height_cutoff_cps=100.0)
    P.apply_side_channels(cfg, P.UR)
    lists, _tags, rec = RL.activate("known-answer", "uronium", record=True, include=include,
                                    fields=("context", "reagent label"))
    out = A.run("malathion-ka" + ("-list" if include else "-nolist"), "uronium", cfg=cfg, peaks=spectrum(),
                adducts=list(P.UR.adducts), use_cache=False, log=lambda *_: None, resolving_power=140000,
                reflists_active=lists, reflists_context=rec, reagent_profile="Ur")
    led = out["ledger"].copy()
    led["mz"] = pd.to_numeric(led["mz"], errors="coerce")
    return out, led


def row(led, adduct):
    mz = C.ion_mz(MALATHION, adduct)
    r = led[(led["mz"] - mz).abs() < 0.003]
    assert len(r) == 1, r
    return r.iloc[0]


def test_malathion_is_certified_named_and_levelled_3c_from_the_snapshot_alone(no_p_dictionaries):
    out, led = run(include=E.LIST_ID)
    for add_ in ("[M+H]+", "[M+NH4]+", "[M+(CH4N2O)H]+"):
        r = row(led, add_)
        assert r["role"] == "M0" and r["neutral_formula"] == MALATHION and r["adduct"] == add_
        assert r["tier"] == "Assigned" and str(r["method"]).startswith("certified:"), (add_, r["method"])
        assert f"reflist:{E.LIST_ID} = Malathion" in str(r["context"]), r["context"]
        assert "MODE CONTRADICTS" not in str(r["evidence"])   # a `both`-polarity list contradicts no run
    # the protonated and the ammonium reading: split pinned + the named entry -> 3c, claim identified
    for add_ in ("[M+H]+", "[M+NH4]+"):
        r = row(led, add_)
        assert r["evidence_level"] == "3c" and r["claim"] == "identified", (add_, r["evidence"])
    # the urea cluster's split stays open offline (no time series to settle its NH4 alias): ion only
    r = row(led, "[M+(CH4N2O)H]+")
    assert r["evidence_level"] in ("4b", "4a", "3c"), r["evidence"]
    # the run says which list it read
    rec = {x["id"]: x for x in out["lists"]}
    me = rec[E.LIST_ID]
    assert me["pinned"] is True and me["version"] == E.VERSION and len(me["sha256"]) == 64
    assert me["how_activated"] == "explicit (--lists)" and me["roles"] == ["name"]
    assert out["reflists_active"] == [("contaminants_keller2008", "2008.2"), (E.LIST_ID, E.VERSION)]


def test_without_the_list_malathion_is_found_but_unnamed(no_p_dictionaries):
    out, led = run(include=None)
    r = row(led, "[M+H]+")
    assert r["tier"] == "Assigned" and r["neutral_formula"] == MALATHION
    assert str(r["method"]).startswith("certified:")
    assert "reflist:" not in str(r["context"]) and r["evidence_level"] != "3c"
    assert [x["id"] for x in out["lists"]] == ["contaminants_keller2008"]
