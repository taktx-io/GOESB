"""Portuguese (pt) normalization ruleset `goesb-pt-v1`.

Mirrors oesb_en_v1's structure exactly; only the language-specific number
expansion differs. Number expansion covers plain integers 0-999999 written
with digits, using the Brazilian Portuguese forms consistent with FLEURS'
pt_br audio (e.g. "dezesseis" not the European "dezasseis").
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]
_TEENS = ["dez", "onze", "doze", "treze", "catorze", "quinze",
          "dezesseis", "dezessete", "dezoito", "dezenove"]
_TENS = {2: "vinte", 3: "trinta", 4: "quarenta", 5: "cinquenta",
         6: "sessenta", 7: "setenta", 8: "oitenta", 9: "noventa"}
_HUNDREDS = {2: "duzentos", 3: "trezentos", 4: "quatrocentos", 5: "quinhentos",
             6: "seiscentos", 7: "setecentos", 8: "oitocentos", 9: "novecentos"}

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
    return tens_word if unit_digit == 0 else f"{tens_word} e {_UNITS[unit_digit]}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        prefix = "cem" if rest == 0 else "cento"
    else:
        prefix = _HUNDREDS[hundreds]
    return prefix if rest == 0 else f"{prefix} e {_below_100(rest)}"


def number_to_portuguese_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to Portuguese number words."""
    if n < 0:
        return "menos " + number_to_portuguese_words(-n)
    if n > 999_999:
        raise ValueError("goesb-pt-v1 number expansion supports 0..999999")
    if n == 0:
        return "zero"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = "mil" if thousands == 1 else f"{_below_1000(thousands)} mil"
    return prefix if rest == 0 else f"{prefix} e {_below_1000(rest)}"


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_portuguese_words(int(m.group())), text)


@register("goesb-pt-v1")
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
