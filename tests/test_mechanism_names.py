"""Guard the server-mechanism -> scorer-spelling conversion in the local scoring
dispatch.

A server before Mascope 1.10 stores a mechanism in the legacy
'<operation><moiety><moiety charge>' spelling, whose trailing sign is the added
or removed species' charge rather than the ion's: deprotonation is '-H+' (a
proton removed) although it yields an ANION. A server from 1.10 on stores the
standard adduct notation, '[M-H]-'. The scorer (`mascope_tools`) reads both by
their grammar since the release this branch pins, so `_mechanism_names` hands
it the standard spelling of whatever the row stores - and no longer flips a
trailing sign to the row's polarity, which under that grammar would turn the
deprotonation into a hydride abstraction ('-H-' is '[M-H]+').
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


def _client(rows):
    return _FakeClient(pd.DataFrame(rows, columns=[
        "ionization_mechanism",
        "ionization_mechanism_polarity",
        "ionization_mechanism_id",
    ]))


#: a mechanism table as a server before Mascope 1.10 stores it
LEGACY = [
    ("-H+", "-", "dep"),  # deprotonation: anion despite trailing '+'
    ("+Br-", "-", "br"),
    ("+CO3-", "-", "co3"),
    ("+NH4+", "+", "nh4"),
    ("+H+", "+", "prot"),
    ("+^NO3-", "-", "no315n"),
    ("+(CH4N2O)H+", "+", "uro"),
    ("+", "+", "et"),
]

#: the same table on a server from 1.10 on
STANDARD = [
    ("[M-H]-", "-", "dep"),
    ("[M+Br]-", "-", "br"),
    ("[M+CO3]-", "-", "co3"),
    ("[M+NH4]+", "+", "nh4"),
    ("[M+H]+", "+", "prot"),
    ("[M+^NO3]-", "-", "no315n"),
    ("[M+CH4N2O+H]+", "+", "uro"),
    ("[M]+.", "+", "et"),
]

#: what the scorer is handed for each id, from either table
EXPECTED = {
    "dep": "[M-H]-", "br": "[M+Br]-", "co3": "[M+CO3]-", "nh4": "[M+NH4]+",
    "prot": "[M+H]+", "no315n": "[M+^NO3]-", "uro": "[M+CH4N2O+H]+", "et": "[M]+.",
}


def test_legacy_rows_are_handed_over_in_the_standard_notation():
    c = _client(LEGACY)
    for mech_id, spelling in EXPECTED.items():
        assert IO._mechanism_names(c, [mech_id]) == [spelling], mech_id


def test_standard_rows_are_handed_over_unchanged():
    c = _client(STANDARD)
    for mech_id, spelling in EXPECTED.items():
        assert IO._mechanism_names(c, [mech_id]) == [spelling], mech_id


def test_deprotonation_is_the_anion_not_the_hydride_cation():
    # the old rewrite made '-H-' of the deprotonation row; the library now
    # reads '-H-' as [M-H]+, so the row must reach the scorer as [M-H]-
    assert IO._mechanism_names(_client(LEGACY), ["dep"]) == ["[M-H]-"]
    assert IO._mechanism_names(_client([("-H-", "+", "hyd")]), ["hyd"]) == ["[M-H]+"]


def test_polarity_column_does_not_override_the_spelling():
    # a row stored with a polarity that contradicts what it says is read as it
    # says, the way the server reads it (logged, not rewritten)
    assert IO._mechanism_names(_client([("-H-", "-", "odd")]), ["odd"]) == ["[M-H]+"]


def test_unreadable_row_is_skipped():
    c = _client([("not a mechanism", "-", "bad"), ("+Br-", "-", "br")])
    assert IO._mechanism_names(c, ["bad", "br"]) == ["[M+Br]-"]


def test_order_and_local_tokens():
    c = _client(LEGACY)
    assert IO._mechanism_names(c, ["br", "dep"]) == ["[M+Br]-", "[M-H]-"]
    # a local token is appended, and not twice when the server also names it
    assert IO._mechanism_names(c, ["prot", "local:[M-H]+"]) == ["[M+H]+", "[M-H]+"]
    assert IO._mechanism_names(_client([("[M-H]+", "+", "hyd")]),
                               ["hyd", "local:[M-H]+"]) == ["[M-H]+"]


def test_unknown_id_skipped_and_empty():
    assert IO._mechanism_names(_client(LEGACY), ["does-not-exist"]) == []
    assert IO._mechanism_names(_client(LEGACY), None) == []


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
