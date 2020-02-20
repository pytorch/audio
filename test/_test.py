import os
import unittest
from contextlib import contextmanager

import common_utils
import torchaudio

BACKENDS = torchaudio._backend._audio_backends


@contextmanager
def AudioBackendScope(new_backend):
    previous_backend = torchaudio.get_audio_backend()
    try:
        # unittest.skipIf(not new_backend, "No backend supporting this test.")
        # unittest.skipIf(new_backend not in BACKENDS, new_backend + " is not available.")
        torchaudio.set_audio_backend(new_backend)
        yield
    finally:
        torchaudio.set_audio_backend(previous_backend)


def get_backends_with_mp3(backends):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()
    test_filepath = os.path.join(
        test_dirpath, "assets", "steam-train-whistle-daniel_simon.mp3"
    )

    backends_mp3 = []

    for b in backends:
        torchaudio.load(test_filepath)
        try:
            with AudioBackendScope(b):
                waveform, sample_rate = torchaudio.load(test_filepath)
            backends_mp3.append(b)
        except:
            pass

    return backends_mp3


BACKENDS_MP3 = get_backends_with_mp3(BACKENDS)
FIRST_BACKEND_MP3 = BACKENDS_MP3[0] if BACKENDS_MP3 else None
