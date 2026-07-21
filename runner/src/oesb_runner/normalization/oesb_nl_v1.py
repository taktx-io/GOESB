"""Dutch (nl) normalization ruleset `oesb-nl-v1`.

The only place Dutch-specific text handling lives (docs/02-architecture.md
§3 Profile). WER/CER alignment itself stays language-agnostic; this module
just prepares reference/hypothesis text identically before scoring.

Number expansion covers plain integers 0-999999 written with digits (e.g.
"32" -> "tweeëndertig"); decimals, ordinals, and dates are a known
limitation, out of scope for M1.
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["nul", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"]
_TEENS = ["tien", "elf", "twaalf", "dertien", "veertien", "vijftien",
          "zestien", "zeventien", "achttien", "negentien"]
_TENS = {2: "twintig", 3: "dertig", 4: "veertig", 5: "vijftig",
         6: "zestig", 7: "zeventig", 8: "tachtig", 9: "negentig"}
# Dutch spelling requires a trema when "een"/"en" would otherwise create a
# vowel clash with the preceding unit word (e.g. "twee" + "en" -> "tweeën").
_UNIT_EN_JOIN = {"twee": "tweeën", "drie": "drieën"}

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
    unit_word = _UNITS[unit_digit]
    if unit_word in _UNIT_EN_JOIN:
        # e.g. "twee" -> "tweeën" already fuses the "en" joiner with a trema.
        return f"{_UNIT_EN_JOIN[unit_word]}{tens_word}"
    return f"{unit_word}en{tens_word}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    prefix = "honderd" if hundreds == 1 else f"{_UNITS[hundreds]}honderd"
    return prefix if rest == 0 else prefix + _below_100(rest)


def number_to_dutch_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to Dutch number words."""
    if n < 0:
        return "min " + number_to_dutch_words(-n)
    if n > 999_999:
        raise ValueError("oesb-nl-v1 number expansion supports 0..999999")
    if n == 0:
        return "nul"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = "duizend" if thousands == 1 else _below_1000(thousands) + "duizend"
    return prefix if rest == 0 else prefix + _below_1000(rest)


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_dutch_words(int(m.group())), text)


@register("oesb-nl-v1")
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
