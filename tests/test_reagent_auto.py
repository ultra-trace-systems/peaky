"""`--reagent auto` (the CLI default) reads the reagent from the server's own
ionization mechanisms and STOPS when none of them is a reagent's diagnostic adduct.

Pinned here, offline and on synthetic tables:
  * polarity comes from the mechanisms' own charge (standard or legacy spelling),
    then from a `polarity` column's words -- never from a batch or sample name;
  * no diagnostic adduct -> ValueError naming the mechanisms seen (no guess of the
    first registered profile of the polarity).
"""
import pandas as pd
import pytest

from peaky import cli
from peaky.chem import profiles as P
from peaky.io import io_mascope as IO

# a batch name full of hyphens (as dated, instrument-coded names are), and one carrying a "+"
HYPHENS = "Instrument-A uronium run - zone-2 - zone-3"
PLUS = "PTR H3O+ run"


def _table(mechs, *, batch=HYPHENS, polarity=None, n=12):
    """A peak table shaped like a batch time series: `mechs` fill the first rows of
    `ionization_mechanism`, the rest are unmatched peaks."""
    d = pd.DataFrame({
        "sample_item_id": [f"s{i % 3}" for i in range(n)],
        "sample_item_name": [f"file-{i % 3}-2021-03-04-10-00" for i in range(n)],
        "sample_batch_name": batch,
        "datetime_utc": pd.Timestamp("2021-03-04"),
        "mz": [100.0 + 7.3 * i for i in range(n)],
        "height": 1e4,
        "ionization_mechanism": (list(mechs) + [None] * n)[:n],
    })
    if polarity is not None:
        d["polarity"] = polarity
    return d


# --------------------------------------------------------------------------- #
# polarity: the mechanisms' own charge, then a polarity column, never a name
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechs, sign", [
    (["[M+H]+"], "+"),
    (["+H+"], "+"),                     # legacy protonation
    (["-H+"], "-"),                     # legacy DEprotonation: the last char is not the charge
    (["+Br-", "-H+"], "-"),
    (["+"], "+"),                       # legacy bare molecular cation = [M]+.
    (["[M]-."], "-"),
    (["[M+(CH4N2O)H]+", "+H+"], "+"),
])
def test_polarity_is_read_from_the_mechanisms_charge(mechs, sign):
    assert P._detect_polarity(_table(mechs)) == sign


def test_both_polarities_stamped_is_no_polarity():
    assert P._detect_polarity(_table(["[M+H]+", "[M-H]-"])) is None


def test_the_polarity_column_is_read_when_no_mechanism_says():
    assert P._detect_polarity(_table([], polarity="Positive")) == "+"
    assert P._detect_polarity(_table([], polarity="negative")) == "-"
    assert P._detect_polarity(_table([], polarity="-")) == "-"
    # the mechanisms outrank the column
    assert P._detect_polarity(_table(["-H+"], polarity="positive")) == "-"


@pytest.mark.parametrize("batch", [HYPHENS, PLUS, "Iodide CIMS 2021-03-04"])
def test_the_batch_and_sample_names_are_never_read(batch):
    t = _table([], batch=batch)
    assert P._detect_polarity(t) is None
    with pytest.raises(ValueError, match="could not auto-detect reagent"):
        P.resolve("auto", t)


# --------------------------------------------------------------------------- #
# diagnostic adducts still resolve; anything less raises, naming what it saw
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechs, name", [
    (["+(CH4N2O)H+", "+H+"], "Ur"),
    (["[M+CH4N2O+H]+", "[M+H]+"], "Ur"),
    (["+NO3-", "+^NO3-", "-H+"], "NO3+NO3_15N"),
    (["+Br-", "+NO3-", "-H+"], "Br+NO3"),
    (["[M+I]-", "[M-H]-"], "I"),
    (["+"], "EasyIC"),                  # a weak signature, believed when alone
])
def test_a_diagnostic_adduct_resolves_whatever_the_name(mechs, name):
    assert P.resolve("auto", _table(mechs, batch=HYPHENS)).name == name


@pytest.mark.parametrize("mechs, polarity, seen, pol_word", [
    (["[M+H]+", "[M+Na]+"], None, "[M+H]+, [M+Na]+", "positive"),
    (["+H+"], None, "[M+H]+", "positive"),
    (["-H+"], None, "[M-H]-", "negative"),
    ([], "positive", "none", "positive"),
    ([], None, "none", "unknown"),
])
def test_no_diagnostic_adduct_raises_instead_of_guessing(mechs, polarity, seen, pol_word):
    with pytest.raises(ValueError) as e:
        P.resolve("auto", _table(mechs, polarity=polarity))
    msg = str(e.value)
    assert f"mechanisms seen: {seen};" in msg
    assert f"polarity {pol_word}" in msg
    assert "--reagent" in msg and "Ur" in msg and "Br" in msg
    # the CLI boundary maps 'not found' / 'no peaks' texts to a stale-id hint
    assert cli._friendly_server_error(e.value) is None


def test_a_generic_signature_cannot_fire_on_a_table_with_no_matches(monkeypatch):
    """detect_adducts' [M-H]- default is not a match: a profile declaring [M-H]- as
    its signature must not be picked for a table the server matched nothing in."""
    monkeypatch.setattr(P, "PROFILES", dict(P.PROFILES))
    monkeypatch.setattr(P, "_BY_ALIAS", dict(P._BY_ALIAS))
    P.register(P.ReagentProfile(
        name="Deprot", label="deprotonation only", polarity="-", adducts=["[M-H]-"],
        normaliser="tic", reagent_ion_re=None, ranges="C0-10 H0-20",
        detect_adduct="[M-H]-"))
    with pytest.raises(ValueError, match="mechanisms seen: none"):
        P.resolve("auto", _table([]))
    assert P.resolve("auto", _table(["-H+"])).name == "Deprot"


def test_recognised_adducts_has_no_default():
    assert IO.recognised_adducts(_table([])) == []
    assert IO.recognised_adducts(pd.DataFrame({"mz": [1.0]})) == []
    assert IO.detect_adducts(_table([])) == ["[M-H]-"]          # unchanged
    assert IO.recognised_adducts(_table(["+Br-", "-H+", "+Br-"])) == ["[M+Br]-", "[M-H]-"]
