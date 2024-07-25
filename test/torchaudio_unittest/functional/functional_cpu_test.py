import unittest

import torch
import torchaudio.functional as F
from parameterized import parameterized
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoSox, TorchaudioTestCase

from .functional_impl import Functional, FunctionalCPUOnly


class TestFunctionalFloat32(Functional, FunctionalCPUOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")

    @unittest.expectedFailure
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


class TestFunctionalFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")


@skipIfNoSox
class TestApplyCodec(TorchaudioTestCase):
    def _smoke_test(self, format, compression, check_num_frames):
        """
        The purpose of this test suite is to verify that apply_codec functionalities do not exhibit
        abnormal behaviors.
        """
        sample_rate = 8000
        num_frames = 3 * sample_rate
        num_channels = 2
        waveform = torch.rand(num_channels, num_frames)

        augmented = F.apply_codec(waveform, sample_rate, format, True, compression)
        assert augmented.dtype == waveform.dtype
        assert augmented.shape[0] == num_channels
        if check_num_frames:
            assert augmented.shape[1] == num_frames

    def test_wave(self):
        self._smoke_test("wav", compression=None, check_num_frames=True)

    @parameterized.expand([(96,), (128,), (160,), (192,), (224,), (256,), (320,)])
    def test_mp3(self, compression):
        self._smoke_test("mp3", compression, check_num_frames=False)

    @parameterized.expand([(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)])
    def test_flac(self, compression):
        self._smoke_test("flac", compression, check_num_frames=False)

    @parameterized.expand([(-1,), (0,), (1,), (2,), (3,), (3.6,), (5,), (10,)])
    def test_vorbis(self, compression):
        self._smoke_test("vorbis", compression, check_num_frames=False)
