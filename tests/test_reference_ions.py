"""chem/reference_ions.py -- the external mass yardstick."""
import csv

import numpy as np
import pytest

from peaky.chem import chemistry as C
from peaky.chem import reference_ions as RI
from peaky.paths import pkg_data


def test_the_nitrate_core_is_thirty_ions_with_eleven_anchors():
    t = RI.nitrate()
    assert len(t) == 30
    assert int(t["anchor"].sum()) == 11
    assert list(t.columns) == RI.COLUMNS
    assert (t["reagent"] == "NO3").all()
    assert set(t["channel"]) <= set(C.ADDUCT_SHIFTS)


def test_masses_come_from_chem_ion_mz():
    t = RI.nitrate()
    for _, r in t.iterrows():
        assert abs(r["mz"] - C.ion_mz(r["neutral"], r["channel"])) < 1e-9
    hno3 = t[(t["neutral"] == "HNO3") & (t["channel"] == "[M-H]-")].iloc[0]
    assert abs(hno3["mz"] - 61.98837) < 1e-4


def test_the_15n_variant_moves_only_the_cluster_channel():
    a, b = RI.nitrate(), RI.nitrate(label_15n=True)
    assert (b["reagent"] == "NO3_15N").all()
    assert "[M+^NO3]-" in set(b["channel"]) and "[M+NO3]-" not in set(b["channel"])
    ka = a.set_index(["neutral", "identity"])["mz"]
    kb = b.set_index(["neutral", "identity"])["mz"]
    d = (kb - ka).dropna()
    clus = a.set_index(["neutral", "identity"])["channel"] == "[M+NO3]-"
    assert np.allclose(d[clus], 0.9970349, atol=1e-6)      # 15N - 14N
    assert np.allclose(d[~clus], 0.0)


def test_bromide_carries_a_heavy_twin_for_every_br_adduct():
    t = RI.bromide()
    light = t[(t["channel"] == "[M+Br]-")]
    heavy = t[(t["channel"] == "[M+81Br]-")]
    assert len(light) == len(heavy) >= 8
    for _, r in light.iterrows():
        h = heavy[heavy["twin_key"] == r["twin_key"]]
        assert len(h) == 1
        assert abs(h.iloc[0]["mz"] - r["mz"] - 1.9979521) < 1e-6
        assert h.iloc[0]["iso"] == "81Br" and r["iso"] == ""
        assert not h.iloc[0]["anchor"]                  # twins never calibrate on their own
    assert (t["reagent"] == "Br").all()


def test_get_resolves_aliases_and_refuses_unknown_reagents():
    assert len(RI.get("nitrate")) == 30
    assert RI.get("^NO3")["reagent"].iloc[0] == "NO3_15N"
    assert RI.get("Br-")["reagent"].iloc[0] == "Br"
    with pytest.raises(KeyError):
        RI.get("I")
    assert RI.available() == ["NO3", "NO3_15N", "Br"]


def test_the_shipped_csv_carries_no_site_or_instrument_columns():
    with open(pkg_data("reference_ions", "nitrate.csv"), newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ["neutral", "channel", "identity", "grade", "anchor", "blend_flag"]


def test_a_composed_reagent_name_unions_both_channels_tables():
    # `profiles.compose` names a mixed inlet 'Br+NO3'. That name matched no table,
    # so a mixed-reagent TOF -- the instrument the wave exists for -- ran with no
    # calibrants at all and silently skipped mass-qc.
    no3, br = RI.get("NO3"), RI.get("Br")
    both = RI.get("Br+NO3")
    assert RI.get("NO3+Br").equals(both)                 # order is not information
    # the union, minus the ions certain in BOTH tables (HNO3 [M-H]- and one more)
    shared = 2
    assert len(both) == len(no3) + len(br) - shared == 53
    assert both["mz"].is_monotonic_increasing
    # an ion kept once, at its STRONGER grading: HNO3 [M-H]- is a nitrate anchor
    # and a B-grade bystander for bromide
    hno3 = both[(both["neutral"] == "HNO3") & (both["channel"] == "[M-H]-")]
    assert len(hno3) == 1 and hno3.iloc[0]["grade"] == "A" and bool(hno3.iloc[0]["anchor"])
    # the 79/81Br pairs share a twin_key on purpose -- dedup must not collapse them
    assert (both["iso"] == "81Br").sum() == (br["iso"] == "81Br").sum() == 12
    assert both.duplicated(subset=["neutral", "channel", "iso"]).sum() == 0


def test_a_composed_name_still_refuses_when_no_part_is_known():
    with pytest.raises(KeyError):
        RI.get("I+EasyIC")
    # one known part is enough, and gives exactly that part
    assert RI.get("NO3+I").equals(RI.get("NO3"))
