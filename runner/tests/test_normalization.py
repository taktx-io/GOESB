import pytest

from oesb_runner.normalization import get_normalizer, normalize
from oesb_runner.normalization.oesb_nl_v1 import number_to_dutch_words


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "nul"),
        (1, "een"),
        (13, "dertien"),
        (20, "twintig"),
        (21, "eenentwintig"),
        (22, "tweeëntwintig"),
        (23, "drieëntwintig"),
        (24, "vierentwintig"),
        (100, "honderd"),
        (165, "honderdvijfenzestig"),
        (200, "tweehonderd"),
        (1000, "duizend"),
        (2024, "tweeduizendvierentwintig"),
        (999_999, "negenhonderdnegenennegentigduizendnegenhonderdnegenennegentig"),
    ],
)
def test_number_to_dutch_words(n, expected):
    assert number_to_dutch_words(n) == expected


def test_number_to_dutch_words_rejects_out_of_range():
    with pytest.raises(ValueError):
        number_to_dutch_words(1_000_000)


def test_normalize_lowercase_and_punctuation():
    out = normalize("goesb-nl-v1", "Hallo, wereld!", expand_numbers=False)
    assert out == "hallo wereld"


def test_normalize_expands_numbers():
    out = normalize("goesb-nl-v1", "Ik heb 24 appels.", remove_punctuation=False)
    assert "vierentwintig" in out
    assert "24" not in out


def test_normalize_disable_all_options_is_identity_modulo_case():
    out = normalize(
        "goesb-nl-v1", "Hallo, 3 Wereld!",
        lowercase=False, remove_punctuation=False, expand_numbers=False,
    )
    assert out == "Hallo, 3 Wereld!"


def test_unknown_ruleset_raises():
    with pytest.raises(ValueError):
        get_normalizer("does-not-exist")
