from __future__ import absolute_import, division, print_function, unicode_literals
import random
import torchaudio

TEST_PREFIX = ['fbank', 'spec', 'resample']


def generate_rand_boolean():
    # Generates a random boolean ('true', 'false')
    return 'true' if random.randint(0, 1) else 'false'


def generate_rand_window_type():
    # Generates a random window type
    return torchaudio.compliance.kaldi.WINDOWS[random.randint(0, len(torchaudio.compliance.kaldi.WINDOWS) - 1)]


def parse(token):
    # converts an arg extracted from filepath to its corresponding python type
    if token == 'true':
        return True
    if token == 'false':
        return False
    if token in torchaudio.compliance.kaldi.WINDOWS or token in TEST_PREFIX:
        return token
    if '.' in token:
        return float(token)
    return int(token)
