import os
import tempfile
from contextlib import contextmanager
from shutil import copytree

import torch
import torchaudio

_TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BACKENDS = torchaudio._backend._audio_backends


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, 'assets', *paths)


def create_temp_assets_dir():
    """
    Creates a temporary directory and moves all files from test/assets there.
    Returns a Tuple[string, TemporaryDirectory] which is the folder path
    and object.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    copytree(os.path.join(_TEST_DIR_PATH, "assets"),
             os.path.join(tmp_dir.name, "assets"))
    return tmp_dir.name, tmp_dir


def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32):
    """ Generates random tensors given a seed and size
    https://en.wikipedia.org/wiki/Linear_congruential_generator
    X_{n + 1} = (a * X_n + c) % m
    Using Borland C/C++ values

    The tensor will have values between [0,1)
    Inputs:
        seed (int): an int
        size (Tuple[int]): the size of the output tensor
        a (int): the multiplier constant to the generator
        c (int): the additive constant to the generator
        m (int): the modulus constant to the generator
    """
    num_elements = 1
    for s in size:
        num_elements *= s

    arr = [(a * seed + c) % m]
    for i in range(num_elements - 1):
        arr.append((a * arr[i] + c) % m)

    return torch.tensor(arr).float().view(size) / m


def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low


@contextmanager
def AudioBackendScope(new_backend):
    previous_backend = torchaudio.get_audio_backend()
    try:
        torchaudio.set_audio_backend(new_backend)
        yield
    finally:
        torchaudio.set_audio_backend(previous_backend)


def filter_backends_with_mp3(backends):
    # Filter out backends that do not support mp3
    test_filepath = get_asset_path('steam-train-whistle-daniel_simon.mp3')

    def supports_mp3(backend):
        try:
            with AudioBackendScope(backend):
                torchaudio.load(test_filepath)
            return True
        except (RuntimeError, ImportError):
            return False

    return [backend for backend in backends if supports_mp3(backend)]


BACKENDS_MP3 = filter_backends_with_mp3(BACKENDS)


_IS_SOX_INITIALIZED = False


def initialize_sox():
    """Initialize sox backend only if it has not yet."""
    global _IS_SOX_INITIALIZED
    if not _IS_SOX_INITIALIZED:
        torchaudio.initialize_sox()
        _IS_SOX_INITIALIZED = True
