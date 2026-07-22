import pytest

from oesb_runner.normalization import normalize
from oesb_runner.normalization.oesb_en_v1 import number_to_english_words


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "zero"),
        (1, "one"),
        (13, "thirteen"),
        (20, "twenty"),
        (24, "twenty four"),
        (100, "one hundred"),
        (165, "one hundred sixty five"),
        (2024, "two thousand twenty four"),
    ],
)
def test_number_to_english_words(n, expected):
    assert number_to_english_words(n) == expected


def test_number_to_english_words_rejects_out_of_range():
    with pytest.raises(ValueError):
        number_to_english_words(1_000_000)


def test_normalize_lowercase_and_punctuation():
    out = normalize("goesb-en-v1", "Hello, World!", expand_numbers=False)
    assert out == "hello world"


def test_normalize_expands_numbers():
    out = normalize("goesb-en-v1", "I have 24 apples.", remove_punctuation=False)
    assert "twenty four" in out
    assert "24" not in out
