import math

import torch
import torchaudio
import unittest
import test.common_utils


class TestFunctional(unittest.TestCase):
    data_sizes = [(2, 20), (3, 15), (4, 10)]
    number_of_trials = 100

    def _compare_estimate(self, sound, estimate, atol=1e-6, rtol=1e-8):
        # trim sound for case when constructed signal is shorter than original
        sound = sound[..., :estimate.size(-1)]

        self.assertTrue(sound.shape == estimate.shape, (sound.shape, estimate.shape))
        self.assertTrue(torch.allclose(sound, estimate, atol=atol, rtol=rtol))

    def _test_istft_is_inverse_of_stft(self, kwargs):
        # generates a random sound signal for each tril and then does the stft/istft
        # operation to check whether we can reconstruct signal
        for data_size in self.data_sizes:
            for i in range(self.number_of_trials):
                sound = test.common_utils.random_float_tensor(i, data_size)

                stft = torch.stft(sound, **kwargs)
                estimate = torchaudio.functional.istft(stft, length=sound.size(1), **kwargs)

                self._compare_estimate(sound, estimate)

    def test_istft_is_inverse_of_stft1(self):
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

        self._test_istft_is_inverse_of_stft(kwargs1)

    def test_istft_is_inverse_of_stft2(self):
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

        self._test_istft_is_inverse_of_stft(kwargs2)

    def test_istft_is_inverse_of_stft3(self):
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

        self._test_istft_is_inverse_of_stft(kwargs3)

    def test_istft_is_inverse_of_stft4(self):
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

        self._test_istft_is_inverse_of_stft(kwargs4)

    def test_istft_is_inverse_of_stft5(self):
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

        self._test_istft_is_inverse_of_stft(kwargs5)

    def test_istft_of_ones(self):
        # stft = torch.stft(torch.ones(4), 4)
        stft = torch.tensor([
            [[4., 0.], [4., 0.], [4., 0.], [4., 0.], [4., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ])

        estimate = torchaudio.functional.istft(stft, n_fft=4, length=4)
        self._compare_estimate(torch.ones(4), estimate)

    def test_istft_of_zeros(self):
        # stft = torch.stft(torch.zeros(4), 4)
        stft = torch.zeros((3, 5, 2))

        estimate = torchaudio.functional.istft(stft, n_fft=4, length=4)
        self._compare_estimate(torch.zeros(4), estimate)

    def test_istft_requires_overlap_windows(self):
        # the window is size 1 but it hops 20 so there is a gap which throw an error
        stft = torch.zeros((3, 5, 2))
        self.assertRaises(AssertionError, torchaudio.functional.istft, stft, n_fft=4,
                          hop_length=20, win_length=1, window=torch.ones(1))

    def test_istft_requires_nola(self):
        stft = torch.zeros((3, 5, 2))
        kwargs_ok = {
            'n_fft': 4,
            'win_length': 4,
            'window': torch.ones(4),
        }

        kwargs_not_ok = {
            'n_fft': 4,
            'win_length': 4,
            'window': torch.zeros(4),
        }

        # A window of ones meets NOLA but a window of zeros does not. This should
        # throw an error.
        torchaudio.functional.istft(stft, **kwargs_ok)
        self.assertRaises(AssertionError, torchaudio.functional.istft, stft, **kwargs_not_ok)

    def test_istft_requires_non_empty(self):
        self.assertRaises(AssertionError, torchaudio.functional.istft, torch.zeros((3, 0, 2)), 2)
        self.assertRaises(AssertionError, torchaudio.functional.istft, torch.zeros((0, 3, 2)), 2)

    def _test_istft_of_sine(self, amplitude, L, n):
        # stft of amplitude*sin(2*pi/L*n*x) with the hop length and window size equaling L
        x = torch.arange(2 * L + 1, dtype=torch.get_default_dtype())
        sound = amplitude * torch.sin(2 * math.pi / L * x * n)
        # stft = torch.stft(sound, L, hop_length=L, win_length=L,
        #                   window=torch.ones(L), center=False, normalized=False)
        stft = torch.zeros((L // 2 + 1, 2, 2))
        stft_largest_val = (amplitude * L) / 2.0
        if n < stft.size(0):
            stft[n, :, 1] = -stft_largest_val

        if 0 <= L - n < stft.size(0):
            # symmetric about L // 2
            stft[L - n, :, 1] = stft_largest_val

        estimate = torchaudio.functional.istft(stft, L, hop_length=L, win_length=L,
                                               window=torch.ones(L), center=False, normalized=False)
        # There is a larger error due to the scaling of amplitude
        self._compare_estimate(sound, estimate, atol=1e-3)

    def test_istft_of_sine(self):
        self._test_istft_of_sine(amplitude=123, L=5, n=1)
        self._test_istft_of_sine(amplitude=150, L=5, n=2)
        self._test_istft_of_sine(amplitude=111, L=5, n=3)
        self._test_istft_of_sine(amplitude=160, L=7, n=4)
        self._test_istft_of_sine(amplitude=145, L=8, n=5)
        self._test_istft_of_sine(amplitude=80, L=9, n=6)
        self._test_istft_of_sine(amplitude=99, L=10, n=7)


if __name__ == '__main__':
    unittest.main()
