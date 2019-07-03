import os

import torch
import torchaudio
import unittest
import test.common_utils


class TestFunctional(unittest.TestCase):
    # size (2,20)
    test_data = torch.tensor([
        [45.4243, 81.9316, 19.1100, 32.4998, 45.3313, 68.8204, 42.0782, 19.7222,
         76.8721, 69.9104, 27.7188, 86.3579, 30.3251, 92.0308, 70.0568, 74.8940,
         94.3127, 82.9875, 88.8303, 96.3460],
        [59.4262, 91.0040, 74.7672, 79.8533, 46.7943, 13.6757, 85.5145, 33.0060,
         88.5102, 25.6912, 57.9501, 33.3326, 71.5654, 90.0321, 81.8218, 91.6907,
         87.9834, 16.4177, 62.4474, 0.2146]
    ]).float()

    def _test_istft_helper(self, sound, kwargs):
        stft = torch.stft(sound, **kwargs)
        estimate = torchaudio.functional.istft(stft, length=sound.size(1), **kwargs)

        # trim sound for case when constructed signal is shorter than original
        sound = sound[:, :estimate.size(1)]

        self.assertTrue(sound.shape == estimate.shape, (sound.shape, estimate.shape))
        self.assertTrue(torch.allclose(sound, estimate, atol=1e-4))

    def test_istft(self):
        kwargs1 = {
            'n_fft': 12,
            'hop_length': 4,
            'win_length': 12,
            'window': torch.hann_window(12),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': True,
            'onesided': True,
        }

        kwargs2 = {
            'n_fft': 12,
            'hop_length': 2,
            'win_length': 8,
            'window': torch.hann_window(8),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': False,
            'onesided': False,
        }

        kwargs3 = {
            'n_fft': 15,
            'hop_length': 3,
            'win_length': 11,
            'window': torch.hamming_window(11),
            'center': True,
            'pad_mode': 'constant',
            'normalized': True,
            'onesided': False,
        }

        kwargs4 = {
            'n_fft': 5,
            'hop_length': 2,
            'win_length': 5,
            'window': torch.hamming_window(5),
            'center': False,
            'pad_mode': 'constant',
            'normalized': False,
            'onesided': True,
        }

        kwargs5 = {
            'n_fft': 3,
            'hop_length': 2,
            'win_length': 3,
            'window': torch.hamming_window(3),
            'center': False,
            'pad_mode': 'reflect',
            'normalized': False,
            'onesided': False,
        }

        self._test_istft_helper(self.test_data, kwargs1)
        self._test_istft_helper(self.test_data, kwargs2)
        self._test_istft_helper(self.test_data, kwargs3)
        self._test_istft_helper(self.test_data, kwargs4)
        self._test_istft_helper(self.test_data, kwargs5)


if __name__ == '__main__':
    unittest.main()
