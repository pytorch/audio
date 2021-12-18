# *****************************************************************************
# Copyright (c) 2017 Keith Ito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# *****************************************************************************
"""
Modified from https://github.com/keithito/tacotron
"""

import re

import inflect


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(text: str) -> str:
    return re.sub(_comma_number_re, lambda m: m.group(1).replace(",", ""), text)


def _expand_pounds(text: str) -> str:
    return re.sub(_pounds_re, r"\1 pounds", text)


def _expand_dollars_repl_fn(m):
    """The replacement function for expanding dollars."""
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    if len(parts) > 1 and parts[1]:
        if len(parts[1]) == 1:
            # handle the case where we have one digit after the decimal point
            cents = int(parts[1]) * 10
        else:
            cents = int(parts[1])
    else:
        cents = 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_dollars(text: str) -> str:
    return re.sub(_dollars_re, _expand_dollars_repl_fn, text)


def _expand_decimal_point(text: str) -> str:
    return re.sub(
        _decimal_number_re, lambda m: m.group(1).replace(".", " point "), text
    )


def _expand_ordinal(text: str) -> str:
    return re.sub(_ordinal_re, lambda m: _inflect.number_to_words(m.group(0)), text)


def _expand_number_repl_fn(m):
    """The replacement function for expanding number."""
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def _expand_number(text: str) -> str:
    return re.sub(_number_re, _expand_number_repl_fn, text)


def normalize_numbers(text: str) -> str:
    text = _remove_commas(text)
    text = _expand_pounds(text)
    text = _expand_dollars(text)
    text = _expand_decimal_point(text)
    text = _expand_ordinal(text)
    text = _expand_number(text)
    return text
