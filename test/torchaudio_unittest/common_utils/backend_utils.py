import unittest

import torchaudio


def set_audio_backend(backend):
    """Allow additional backend value, 'default'"""
    backends = torchaudio.list_audio_backends()
    if backend == 'soundfile':
        be = 'soundfile'
    elif backend == 'default':
        if 'sox_io' in backends:
            be = 'sox_io'
        elif 'soundfile' in backends:
            be = 'soundfile'
        else:
            raise unittest.SkipTest('No default backend available')
    else:
        be = backend

    torchaudio.set_audio_backend(be)
