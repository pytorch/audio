"""Test suites for numerical compatibility with librosa"""
import os
import unittest
from distutils.version import StrictVersion
from parameterized import parameterized

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio._internal.module_utils import is_module_available

LIBROSA_AVAILABLE = is_module_available('librosa')

if LIBROSA_AVAILABLE:
    import numpy as np
    import librosa
    import scipy

import pytest

from torchaudio_unittest import common_utils

@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TestFunctional(common_utils.TorchaudioTestCase):
    """Test suite for functions in `functional` module."""
    @parameterized.expand([
        (0.5,), (1.01,), (1.3,)
    ])
    def test_phase_vocoder(self, rate):
        torch.random.manual_seed(48)
        complex_specgrams = torch.randn(2, 1025, 400, dtype=torch.cdouble)
        hop_length = 256

        # Due to cummulative sum, numerical error in using torch.float32 will
        # result in bottom right values of the stretched sectrogram to not
        # match with librosa.

        phase_advance = torch.linspace(0, np.pi * hop_length, complex_specgrams.shape[-2], dtype=torch.double)[..., None]

        complex_specgrams_stretch = F.phase_vocoder(complex_specgrams, rate=rate, phase_advance=phase_advance)

        # == Test shape
        expected_size = list(complex_specgrams.size())
        expected_size[-1] = int(np.ceil(expected_size[-1] / rate))

        assert complex_specgrams.dim() == complex_specgrams_stretch.dim()
        assert complex_specgrams_stretch.size() == torch.Size(expected_size)

        # == Test values
        index = [0] + [slice(None)] * 2
        mono_complex_specgram = complex_specgrams[index].numpy()
        expected_complex_stretch = librosa.phase_vocoder(mono_complex_specgram,
                                                        rate=rate,
                                                        hop_length=hop_length)

        self.assertEqual(complex_specgrams_stretch[index], torch.from_numpy(expected_complex_stretch))
