"""English (en) normalization ruleset `goesb-en-v1`.

Mirrors oesb_nl_v1's structure exactly; only the language-specific number
expansion differs, per the plugin boundary (docs/02-architecture.md §3).
Number expansion covers plain integers 0-999999 written with digits.
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
         6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

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
    return tens_word if unit_digit == 0 else f"{tens_word} {_UNITS[unit_digit]}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    prefix = f"{_UNITS[hundreds]} hundred"
    return prefix if rest == 0 else f"{prefix} {_below_100(rest)}"


def number_to_english_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to English number words."""
    if n < 0:
        return "minus " + number_to_english_words(-n)
    if n > 999_999:
        raise ValueError("goesb-en-v1 number expansion supports 0..999999")
    if n == 0:
        return "zero"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = f"{_below_1000(thousands)} thousand"
    return prefix if rest == 0 else f"{prefix} {_below_1000(rest)}"


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_english_words(int(m.group())), text)


@register("goesb-en-v1")
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
