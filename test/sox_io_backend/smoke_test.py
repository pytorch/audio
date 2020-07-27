import itertools
import unittest

from torchaudio.utils import sox_utils
from torchaudio.backend import sox_io_backend
from torchaudio._internal.module_utils import is_module_available
from parameterized import parameterized

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    skipIfNoExtension,
    get_wav_data,
)
from .common import name_func


skipIfNoMP3 = unittest.skipIf(
    not is_module_available('torchaudio._torchaudio') or
    'mp3' not in sox_utils.list_read_formats() or
    'mp3' not in sox_utils.list_write_formats(),
    '"sox_io" backend does not support MP3')


@skipIfNoExtension
class SmokeTest(TempDirMixin, TorchaudioTestCase):
    """Run smoke test on various audio format

    The purpose of this test suite is to verify that sox_io_backend functionalities do not exhibit
    abnormal behaviors.

    This test suite should be able to run without any additional tools (such as sox command),
    however without such tools, the correctness of each function cannot be verified.
    """
    def run_smoke_test(self, ext, sample_rate, num_channels, *, compression=None, dtype='float32'):
        duration = 1
        num_frames = sample_rate * duration
        path = self.get_temp_path(f'test.{ext}')
        original = get_wav_data(dtype, num_channels, normalize=False, num_frames=num_frames)

        # 1. run save
        sox_io_backend.save(path, original, sample_rate, compression=compression)
        # 2. run info
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels
        # 3. run load
        loaded, sr = sox_io_backend.load(path, normalize=False)
        assert sr == sample_rate
        assert loaded.shape[0] == num_channels

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=name_func)
    def test_wav(self, dtype, sample_rate, num_channels):
        """Run smoke test on wav format"""
        self.run_smoke_test('wav', sample_rate, num_channels, dtype=dtype)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-4.2, -0.2, 0, 0.2, 96, 128, 160, 192, 224, 256, 320],
    )))
    @skipIfNoMP3
    def test_mp3(self, sample_rate, num_channels, bit_rate):
        """Run smoke test on mp3 format"""
        self.run_smoke_test('mp3', sample_rate, num_channels, compression=bit_rate)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-1, 0, 1, 2, 3, 3.6, 5, 10],
    )))
    def test_vorbis(self, sample_rate, num_channels, quality_level):
        """Run smoke test on vorbis format"""
        self.run_smoke_test('vorbis', sample_rate, num_channels, compression=quality_level)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=name_func)
    def test_flac(self, sample_rate, num_channels, compression_level):
        """Run smoke test on flac format"""
        self.run_smoke_test('flac', sample_rate, num_channels, compression=compression_level)
