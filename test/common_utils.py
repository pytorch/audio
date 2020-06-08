import os
import tempfile
import unittest
from typing import Type, Iterable, Union
from contextlib import contextmanager
from shutil import copytree

import torch
from torch.testing._internal.common_utils import TestCase
import torchaudio

_TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BACKENDS = torchaudio._backend._BACKENDS


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


class TestBaseMixin:
    dtype = None
    device = None


_SKIP_IF_NO_CUDA = unittest.skipIf(not torch.cuda.is_available(), reason='CUDA not available')


def define_test_suite(testbase: Type[TestBaseMixin], dtype: str, device: str):
    if dtype not in ['float32', 'float64']:
        raise NotImplementedError(f'Unexpected dtype: {dtype}')
    if device not in ['cpu', 'cuda']:
        raise NotImplementedError(f'Unexpected device: {device}')

    name = f'Test{testbase.__name__}_{device.upper()}_{dtype.capitalize()}'
    attrs = {'dtype': getattr(torch, dtype), 'device': torch.device(device)}
    testsuite = type(name, (testbase, TestCase), attrs)

    if device == 'cuda':
        testsuite = _SKIP_IF_NO_CUDA(testsuite)
    return testsuite


def define_test_suites(
        scope: dict,
        testbases: Iterable[Type[TestBaseMixin]],
        dtypes: Iterable[str] = ('float32', 'float64'),
        devices: Iterable[str] = ('cpu', 'cuda'),
):
    for suite in testbases:
        for device in devices:
            for dtype in dtypes:
                t = define_test_suite(suite, dtype, device)
                scope[t.__name__] = t


def get_whitenoise(
    *,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    seed: int = 0,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
):
    """Generate pseudo audio data with whitenoise

    Args:
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        n_channels: Number of channels
        seed: Seed value used for random number generation.
            Note that this function does not modify global random generator state.
        dtype: Torch dtype
        device: device
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    shape = [n_channels, sample_rate * duration]
    # According to the doc, folking rng on all CUDA devices is slow when there are many CUDA devices,
    # so we only folk on CPU, generate values and move the data to the given device
    with torch.random.fork_rng([]):
        torch.random.manual_seed(seed)
        tensor = torch.randn(shape, dtype=dtype, device='cpu')
    tensor /= 2.0
    tensor.clamp_(-1.0, 1.0)
    return tensor.to(device=device)


def get_sinusoid(
    *,
    frequency: float = 300,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
):
    """Generate pseudo audio data with sine wave.

    Args:
        frequency: Frequency of sine wave
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        n_channels: Number of channels
        dtype: Torch dtype
        device: device

    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    pie2 = 2 * 3.141592653589793
    end = pie2 * frequency * duration
    theta = torch.linspace(0, end, sample_rate * duration, dtype=dtype, device=device)
    return torch.sin(theta, out=None).repeat([n_channels, 1])
