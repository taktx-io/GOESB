"""German (de) normalization ruleset `goesb-de-v1`.

Mirrors oesb_en_v1's structure exactly; only the language-specific number
expansion differs. Number expansion covers plain integers 0-999999 written
with digits. German joins number words into single compounds without
spaces (e.g. "einundzwanzig", "zweihundertdrei") — this is not a bug in the
whitespace-cleanup step below, it's how German number words are actually
written.
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
_TEENS = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
          "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_TENS = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig",
         6: "sechzig", 7: "siebzig", 8: "achtzig", 9: "neunzig"}

_DIGITS_RE = re.compile(r"\d+")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _below_100(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    tens_digit, unit_digit = divmod(n, 10)
    tens_word = _TENS[tens_digit]
    if unit_digit == 0:
        return tens_word
    unit_word = "ein" if unit_digit == 1 else _UNITS[unit_digit]  # "einundzwanzig", not "einsundzwanzig"
    return f"{unit_word}und{tens_word}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    prefix = "hundert" if hundreds == 1 else f"{_UNITS[hundreds]}hundert"
    return prefix if rest == 0 else prefix + _below_100(rest)


def number_to_german_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to German number words."""
    if n < 0:
        return "minus " + number_to_german_words(-n)
    if n > 999_999:
        raise ValueError("goesb-de-v1 number expansion supports 0..999999")
    if n == 0:
        return "null"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = "tausend" if thousands == 1 else f"{_below_1000(thousands)}tausend"
    return prefix if rest == 0 else prefix + _below_1000(rest)


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_german_words(int(m.group())), text)


@register("goesb-de-v1")
def normalize(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    expand_numbers: bool = True,
) -> str:
    if lowercase:
        text = text.lower()
    if expand_numbers:
        text = _expand_numbers(text)
    if remove_punctuation:
        text = _NON_WORD_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
