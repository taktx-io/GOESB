"""French (fr) normalization ruleset `goesb-fr-v1`.

Mirrors oesb_en_v1's structure exactly; only the language-specific number
expansion differs. Number expansion covers plain integers 0-999999 written
with digits, including the base-20 remnants in 70-99 (soixante-dix,
quatre-vingt(s), quatre-vingt-dix) that make French numerals irregular
compared to the other Romance-language rulesets in this package.
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
_TEENS = ["dix", "onze", "douze", "treize", "quatorze", "quinze",
          "seize", "dix-sept", "dix-huit", "dix-neuf"]
_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}

_DIGITS_RE = re.compile(r"\d+")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _below_100(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 70:
        tens_digit, unit_digit = divmod(n, 10)
        tens_word = _TENS[tens_digit]
        if unit_digit == 0:
            return tens_word
        if unit_digit == 1:
            return f"{tens_word} et un"
        return f"{tens_word}-{_UNITS[unit_digit]}"
    if n < 80:
        rest = n - 60  # 10..19
        return "soixante et onze" if rest == 11 else f"soixante-{_TEENS[rest - 10]}"
    if n < 90:
        rest = n - 80  # 0..9
        return "quatre-vingts" if rest == 0 else f"quatre-vingt-{_UNITS[rest]}"
    rest = n - 80  # 10..19
    return "quatre-vingt-onze" if rest == 11 else f"quatre-vingt-{_TEENS[rest - 10]}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        prefix = "cent"
    else:
        prefix = f"{_UNITS[hundreds]} cent" + ("s" if rest == 0 else "")
    return prefix if rest == 0 else f"{prefix} {_below_100(rest)}"


def number_to_french_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to French number words."""
    if n < 0:
        return "moins " + number_to_french_words(-n)
    if n > 999_999:
        raise ValueError("goesb-fr-v1 number expansion supports 0..999999")
    if n == 0:
        return "zéro"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = "mille" if thousands == 1 else f"{_below_1000(thousands)} mille"
    return prefix if rest == 0 else f"{prefix} {_below_1000(rest)}"


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_french_words(int(m.group())), text)


@register("goesb-fr-v1")
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
