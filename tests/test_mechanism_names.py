"""Guard the server-mechanism -> mascope-notation normalisation in the local
scoring dispatch.

The server names deprotonation '-H+' (remove a proton) with polarity '-', but it
yields an ANION. `mascope_tools.parse_ionization` reads the trailing sign as the
net charge, so an un-normalised '-H+' scores as a +1 cation and matches nothing —
which silently dropped the entire [M-H]- channel in a full batch (486 strong
peaks left unexplained). `_mechanism_names` must normalise the trailing sign to
the mechanism's polarity ('-H+' -> '-H-') while leaving already-consistent
adduct names ('+Br-', '+NH4+') untouched.
"""

import pandas as pd

from peaky import io_mascope as IO


class _FakeIonization:
    def __init__(self, df):
        self._df = df

    def list(self):
        return self._df


class _FakeClient:
    def __init__(self, df):
        self.ionization = _FakeIonization(df)


def _client():
    df = pd.DataFrame(
        [
            # name, polarity, id  — mirrors a live server's mechanism table
            ("-H+", "-", "dep"),  # deprotonation: anion despite trailing '+'
            ("+Br-", "-", "br"),
            ("+CO3-", "-", "co3"),
            ("+NH4+", "+", "nh4"),
            ("+H+", "+", "prot"),
            ("+^NO3-", "-", "no315n"),
        ],
        columns=[
            "ionization_mechanism",
            "ionization_mechanism_polarity",
            "ionization_mechanism_id",
        ],
    )
    return _FakeClient(df)


def test_deprotonation_sign_normalised_to_polarity():
    # '-H+' (the bug) must become '-H-' so parse_ionization charges it -1
    assert IO._mechanism_names(_client(), ["dep"]) == ["-H-"]


def test_consistent_adducts_unchanged():
    c = _client()
    assert IO._mechanism_names(c, ["br"]) == ["+Br-"]
    assert IO._mechanism_names(c, ["co3"]) == ["+CO3-"]
    assert IO._mechanism_names(c, ["nh4"]) == ["+NH4+"]
    assert IO._mechanism_names(c, ["prot"]) == ["+H+"]
    assert IO._mechanism_names(c, ["no315n"]) == ["+^NO3-"]


def test_unknown_id_skipped_and_empty():
    assert IO._mechanism_names(_client(), ["does-not-exist"]) == []
    assert IO._mechanism_names(_client(), None) == []


def test_polarity_sign_tolerant():
    assert IO._polarity_sign("negative") == "-"
    assert IO._polarity_sign("POS") == "+"
    assert IO._polarity_sign("-1") == "-"
    assert IO._polarity_sign("?") is None


def test_local_scoring_default_on_and_opt_out(monkeypatch):
    # default (unset) -> local scoring on
    monkeypatch.delenv("PEAKY_LOCAL_SCORING", raising=False)
    assert IO._local_scoring_enabled() is True
    # explicit truthy stays on
    for v in ("1", "true", "yes", "on"):
        monkeypatch.setenv("PEAKY_LOCAL_SCORING", v)
        assert IO._local_scoring_enabled() is True
    # escape hatch back to the server path
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("PEAKY_LOCAL_SCORING", v)
        assert IO._local_scoring_enabled() is False
