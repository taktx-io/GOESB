"""Spanish (es) normalization ruleset `goesb-es-v1`.

Mirrors oesb_en_v1's structure exactly; only the language-specific number
expansion differs. Number expansion covers plain integers 0-999999 written
with digits, using the common (masculine) reading form — gender agreement
with a following noun ("un"/"una") isn't modeled, matching the simplicity
level of the existing en/nl rulesets.
"""
from __future__ import annotations

import re

from . import register

_UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
_TEENS = ["diez", "once", "doce", "trece", "catorce", "quince",
          "dieciséis", "diecisiete", "dieciocho", "diecinueve"]
_TWENTIES = ["veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
             "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta",
         6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_HUNDREDS = {2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos",
             6: "seiscientos", 7: "setecientos", 8: "ochocientos", 9: "novecientos"}

_DIGITS_RE = re.compile(r"\d+")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _below_100(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 30:
        return _TWENTIES[n - 20]
    tens_digit, unit_digit = divmod(n, 10)
    tens_word = _TENS[tens_digit]
    return tens_word if unit_digit == 0 else f"{tens_word} y {_UNITS[unit_digit]}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        prefix = "cien" if rest == 0 else "ciento"
    else:
        prefix = _HUNDREDS[hundreds]
    return prefix if rest == 0 else f"{prefix} {_below_100(rest)}"


def number_to_spanish_words(n: int) -> str:
    """Expand a non-negative integer (<= 999999) to Spanish number words."""
    if n < 0:
        return "menos " + number_to_spanish_words(-n)
    if n > 999_999:
        raise ValueError("goesb-es-v1 number expansion supports 0..999999")
    if n == 0:
        return "cero"
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    prefix = "mil" if thousands == 1 else f"{_below_1000(thousands)} mil"
    return prefix if rest == 0 else f"{prefix} {_below_1000(rest)}"


def _expand_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: number_to_spanish_words(int(m.group())), text)


@register("goesb-es-v1")
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
