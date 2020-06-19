import unittest

import torchaudio

from .import data_utils


BACKENDS = torchaudio.list_audio_backends()


def _filter_backends_with_mp3(backends):
    # Filter out backends that do not support mp3
    test_filepath = data_utils.get_asset_path('steam-train-whistle-daniel_simon.mp3')

    def supports_mp3(backend):
        torchaudio.set_audio_backend(backend)
        try:
            torchaudio.load(test_filepath)
            return True
        except (RuntimeError, ImportError):
            return False

    return [backend for backend in backends if supports_mp3(backend)]


BACKENDS_MP3 = _filter_backends_with_mp3(BACKENDS)


def set_audio_backend(backend):
    """Allow additional backend value, 'default'"""
    if backend == 'default':
        if 'sox' in BACKENDS:
            be = 'sox'
        elif 'soundfile' in BACKENDS:
            be = 'soundfile'
        else:
            raise unittest.SkipTest('No default backend available')
    else:
        be = backend

    torchaudio.set_audio_backend(be)
