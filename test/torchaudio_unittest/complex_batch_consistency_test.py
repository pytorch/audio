"""Test numerical consistency among single input and batched input."""
import unittest
import itertools
from parameterized import parameterized

import torch
import torchaudio
import torchaudio.functional as F

from torchaudio_unittest import common_utils

class TestTransforms(common_utils.TorchaudioTestCase):
    backend = 'default'

    def test_batch_TimeStretch(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        kwargs = {
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'window': torch.hann_window(2048),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': True,
            'onesided': True,
        }
        rate = 2

        complex_specgrams = torch.stft(waveform, **kwargs)
        complex_specgrams = torch.view_as_complex(complex_specgrams)

        # Single then transform then batch
        expected = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams).repeat(3, 1, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams.repeat(3, 1, 1, 1, 1))

        self.assertEqual(computed, expected, atol=1e-5, rtol=1e-5)
