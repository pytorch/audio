import unittest

import torchaudio


def set_audio_backend(backend):
    """Allow additional backend value, 'default'"""
    backends = torchaudio.list_audio_backends()
    if backend == 'soundfile-new':
        be = 'soundfile'
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    elif backend == 'default':
        if 'sox_io' in backends:
            be = 'sox_io'
        elif 'soundfile' in backends:
            be = 'soundfile'
            torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True
        else:
            raise unittest.SkipTest('No default backend available')
    else:
        be = backend

    torchaudio.set_audio_backend(be)
