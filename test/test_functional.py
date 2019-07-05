import os

import torch
import torchaudio
import unittest
import test.common_utils


class TestFunctional(unittest.TestCase):
    data_sizes = (2, 20)
    number_of_trials = 10

    def _test_istft_helper(self, sound, kwargs):
        stft = torch.stft(sound, **kwargs)
        estimate = torchaudio.functional.istft(stft, length=sound.size(1), **kwargs)

        # trim sound for case when constructed signal is shorter than original
        sound = sound[:, :estimate.size(1)]

        self.assertTrue(sound.shape == estimate.shape, (sound.shape, estimate.shape))
        self.assertTrue(torch.allclose(sound, estimate, atol=1e-4))

    def test_istft1(self):
        # hann_window, centered, normalized, onesided
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

        for i in range(self.number_of_trials):
            test_data = torch.rand(self.data_sizes)
            self._test_istft_helper(test_data, kwargs1)

    def test_istft2(self):
        # hann_window, centered, not normalized, not onesided
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

        for i in range(self.number_of_trials):
            test_data = torch.rand(self.data_sizes)
            self._test_istft_helper(test_data, kwargs2)

    def test_istft3(self):
        # hamming_window, centered, normalized, not onesided
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

        for i in range(self.number_of_trials):
            test_data = torch.rand(self.data_sizes)
            self._test_istft_helper(test_data, kwargs3)

    def test_istft4(self):
        # hamming_window, not centered, not normalized, onesided
        # window same size as n_fft
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

        for i in range(self.number_of_trials):
            test_data = torch.rand(self.data_sizes)
            self._test_istft_helper(test_data, kwargs4)

    def test_istft5(self):
        # hamming_window, not centered, not normalized, not onesided
        # window same size as n_fft
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

        for i in range(self.number_of_trials):
            test_data = torch.rand(self.data_sizes)
            self._test_istft_helper(test_data, kwargs5)


if __name__ == '__main__':
    unittest.main()
