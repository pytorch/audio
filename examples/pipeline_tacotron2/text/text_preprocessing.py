"""Modified from https://github.com/keithito/tacotron"""

from typing import List
import re
from unidecode import unidecode

from .numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'abcdefghijklmnopqrstuvwxyz'

symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def text_to_sequence(sent: str) -> List[int]:
    r'''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      Args:
        sent (str): The input sentence to convert to a sequence.

      Returns:
        List of integers corresponding to the symbols in the sentence.
    '''
    sent = unidecode(sent)  # convert to ascii
    sent = sent.lower()  # lower case
    sent = normalize_numbers(sent)  # expand numbers
    for regex, replacement in _abbreviations:  # expand abbreviations
        sent = re.sub(regex, replacement, sent)
    sent = re.sub(_whitespace_re, ' ', sent)  # collapse whitespace

    return [_symbol_to_id[s] for s in sent if s in _symbol_to_id]
