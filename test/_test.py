import os
from contextlib import contextmanager

import common_utils
import torchaudio

BACKENDS = torchaudio._backend._audio_backends


@contextmanager
def AudioBackendScope(new_backend):
    previous_backend = torchaudio.get_audio_backend()
    try:
        torchaudio.set_audio_backend(new_backend)
        yield
    finally:
        torchaudio.set_audio_backend(previous_backend)


def get_backends_with_mp3(backends):
    test_dirpath, _ = common_utils.create_temp_assets_dir()
    test_filepath = os.path.join(
        test_dirpath, "assets", "steam-train-whistle-daniel_simon.mp3"
    )

    backends_mp3 = []

    for backend in backends:
        try:
            with AudioBackendScope(backend):
                torchaudio.load(test_filepath)
            backends_mp3.append(backend)
        except RuntimeError:
            pass

    return backends_mp3


BACKENDS_MP3 = get_backends_with_mp3(BACKENDS)
