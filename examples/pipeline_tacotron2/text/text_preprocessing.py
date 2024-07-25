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
from typing import List, Optional, Union

from torchaudio.datasets import CMUDict
from unidecode import unidecode

from .numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "abcdefghijklmnopqrstuvwxyz"

symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)
_phonemizer = None


available_symbol_set = {"english_characters", "english_phonemes"}
available_phonemizers = {"DeepPhonemizer"}


def get_symbol_list(symbol_list: str = "english_characters", cmudict_root: Optional[str] = "./") -> List[str]:
    if symbol_list == "english_characters":
        return [_pad] + list(_special) + list(_punctuation) + list(_letters)
    elif symbol_list == "english_phonemes":
        return [_pad] + list(_special) + list(_punctuation) + CMUDict(cmudict_root).symbols
    else:
        raise ValueError(
            f"The `symbol_list` {symbol_list} is not supported."
            f"Supported `symbol_list` includes {available_symbol_set}."
        )


def word_to_phonemes(sent: str, phonemizer: str, checkpoint: str) -> List[str]:
    if phonemizer == "DeepPhonemizer":
        from dp.phonemizer import Phonemizer

        global _phonemizer
        _other_symbols = "".join(list(_special) + list(_punctuation))
        _phone_symbols_re = r"(\[[A-Z]+?\]|" + "[" + _other_symbols + "])"  # [\[([A-Z]+?)\]|[-!'(),.:;? ]]

        if _phonemizer is None:
            # using a global variable so that we don't have to relode checkpoint
            # everytime this function is called
            _phonemizer = Phonemizer.from_checkpoint(checkpoint)

        # Example:
        # sent = "hello world!"
        # '[HH][AH][L][OW] [W][ER][L][D]!'
        sent = _phonemizer(sent, lang="en_us")

        # ['[HH]', '[AH]', '[L]', '[OW]', ' ', '[W]', '[ER]', '[L]', '[D]', '!']
        ret = re.findall(_phone_symbols_re, sent)

        # ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D', '!']
        ret = [r.replace("[", "").replace("]", "") for r in ret]

        return ret
    else:
        raise ValueError(
            f"The `phonemizer` {phonemizer} is not supported. " "Supported `symbol_list` includes `'DeepPhonemizer'`."
        )


def text_to_sequence(
    sent: str,
    symbol_list: Union[str, List[str]] = "english_characters",
    phonemizer: Optional[str] = "DeepPhonemizer",
    checkpoint: Optional[str] = "./en_us_cmudict_forward.pt",
    cmudict_root: Optional[str] = "./",
) -> List[int]:
    r"""Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
        sent (str): The input sentence to convert to a sequence.
        symbol_list (str or List of string, optional): When the input is a string, available options include
            "english_characters" and "english_phonemes". When the input is a list of string, ``symbol_list`` will
            directly be used as the symbol to encode. (Default: "english_characters")
        phonemizer (str or None, optional): The phonemizer to use. Only used when ``symbol_list`` is "english_phonemes".
            Available options include "DeepPhonemizer". (Default: "DeepPhonemizer")
        checkpoint (str or None, optional): The path to the checkpoint of the phonemizer. Only used when
            ``symbol_list`` is "english_phonemes". (Default: "./en_us_cmudict_forward.pt")
        cmudict_root (str or None, optional): The path to the directory where the CMUDict dataset is found or
            downloaded. Only used when ``symbol_list`` is "english_phonemes". (Default: "./")

    Returns:
        List of integers corresponding to the symbols in the sentence.

    Examples:
        >>> text_to_sequence("hello world!", "english_characters")
        [19, 16, 23, 23, 26, 11, 34, 26, 29, 23, 15, 2]
        >>> text_to_sequence("hello world!", "english_phonemes")
        [54, 20, 65, 69, 11, 92, 44, 65, 38, 2]
    """
    if symbol_list == "english_phonemes":
        if any(param is None for param in [phonemizer, checkpoint, cmudict_root]):
            raise ValueError(
                "When `symbol_list` is 'english_phonemes', "
                "all of `phonemizer`, `checkpoint`, and `cmudict_root` must be provided."
            )

    sent = unidecode(sent)  # convert to ascii
    sent = sent.lower()  # lower case
    sent = normalize_numbers(sent)  # expand numbers
    for regex, replacement in _abbreviations:  # expand abbreviations
        sent = re.sub(regex, replacement, sent)
    sent = re.sub(_whitespace_re, " ", sent)  # collapse whitespace

    if isinstance(symbol_list, list):
        symbols = symbol_list
    elif isinstance(symbol_list, str):
        symbols = get_symbol_list(symbol_list, cmudict_root=cmudict_root)
        if symbol_list == "english_phonemes":
            sent = word_to_phonemes(sent, phonemizer=phonemizer, checkpoint=checkpoint)

    _symbol_to_id = {s: i for i, s in enumerate(symbols)}

    return [_symbol_to_id[s] for s in sent if s in _symbol_to_id]
