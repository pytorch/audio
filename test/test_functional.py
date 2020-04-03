import math
import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pytest

import common_utils
from common_utils import AudioBackendScope, BACKENDS


class TestFunctional(unittest.TestCase):
    data_sizes = [(2, 20), (3, 15), (4, 10)]
    number_of_trials = 100
    specgram = torch.tensor([1., 2., 3., 4.])

    test_dirpath, test_dir = common_utils.create_temp_assets_dir()

    test_filepath = os.path.join(test_dirpath, 'assets',
                                 'steam-train-whistle-daniel_simon.wav')
    waveform_train, sr_train = torchaudio.load(test_filepath)

    def _test_compute_deltas(self, specgram, expected, win_length=3, atol=1e-6, rtol=1e-8):
        computed = F.compute_deltas(specgram, win_length=win_length)
        self.assertTrue(computed.shape == expected.shape, (computed.shape, expected.shape))
        torch.testing.assert_allclose(computed, expected, atol=atol, rtol=rtol)

    def test_compute_deltas_onechannel(self):
        specgram = self.specgram.unsqueeze(0).unsqueeze(0)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5]]])
        self._test_compute_deltas(specgram, expected)

    def test_compute_deltas_twochannel(self):
        specgram = self.specgram.repeat(1, 2, 1)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]])
        self._test_compute_deltas(specgram, expected)

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

                sound = common_utils.random_float_tensor(i, data_size)

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

    def _test_linearity_of_istft(self, data_size, kwargs, atol=1e-6, rtol=1e-8):
        for i in range(self.number_of_trials):
            tensor1 = common_utils.random_float_tensor(i, data_size)
            tensor2 = common_utils.random_float_tensor(i * 2, data_size)
            a, b = torch.rand(2)
            istft1 = torchaudio.functional.istft(tensor1, **kwargs)
            istft2 = torchaudio.functional.istft(tensor2, **kwargs)
            istft = a * istft1 + b * istft2
            estimate = torchaudio.functional.istft(a * tensor1 + b * tensor2, **kwargs)
            self._compare_estimate(istft, estimate, atol, rtol)

    def test_linearity_of_istft1(self):
        # hann_window, centered, normalized, onesided
        kwargs1 = {
            'n_fft': 12,
            'window': torch.hann_window(12),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': True,
            'onesided': True,
        }
        data_size = (2, 7, 7, 2)
        self._test_linearity_of_istft(data_size, kwargs1)

    def test_linearity_of_istft2(self):
        # hann_window, centered, not normalized, not onesided
        kwargs2 = {
            'n_fft': 12,
            'window': torch.hann_window(12),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': False,
            'onesided': False,
        }
        data_size = (2, 12, 7, 2)
        self._test_linearity_of_istft(data_size, kwargs2)

    def test_linearity_of_istft3(self):
        # hamming_window, centered, normalized, not onesided
        kwargs3 = {
            'n_fft': 12,
            'window': torch.hamming_window(12),
            'center': True,
            'pad_mode': 'constant',
            'normalized': True,
            'onesided': False,
        }
        data_size = (2, 12, 7, 2)
        self._test_linearity_of_istft(data_size, kwargs3)

    def test_linearity_of_istft4(self):
        # hamming_window, not centered, not normalized, onesided
        kwargs4 = {
            'n_fft': 12,
            'window': torch.hamming_window(12),
            'center': False,
            'pad_mode': 'constant',
            'normalized': False,
            'onesided': True,
        }
        data_size = (2, 7, 3, 2)
        self._test_linearity_of_istft(data_size, kwargs4, atol=1e-5, rtol=1e-8)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_gain(self):
        waveform_gain = F.gain(self.waveform_train, 3)
        self.assertTrue(waveform_gain.abs().max().item(), 1.)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("gain", [3])
        sox_gain_waveform = E.sox_build_flow_effects()[0]

        self.assertTrue(torch.allclose(waveform_gain, sox_gain_waveform, atol=1e-04))

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_dither(self):
        waveform_dithered = F.dither(self.waveform_train)
        waveform_dithered_noiseshaped = F.dither(self.waveform_train, noise_shaping=True)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(self.test_filepath)
        E.append_effect_to_chain("dither", [])
        sox_dither_waveform = E.sox_build_flow_effects()[0]

        self.assertTrue(torch.allclose(waveform_dithered, sox_dither_waveform, atol=1e-04))
        E.clear_chain()

        E.append_effect_to_chain("dither", ["-s"])
        sox_dither_waveform_ns = E.sox_build_flow_effects()[0]

        self.assertTrue(torch.allclose(waveform_dithered_noiseshaped, sox_dither_waveform_ns, atol=1e-02))

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_vctk_transform_pipeline(self):
        test_filepath_vctk = os.path.join(self.test_dirpath, "assets/VCTK-Corpus/wav48/p224/", "p224_002.wav")
        wf_vctk, sr_vctk = torchaudio.load(test_filepath_vctk)

        # rate
        sample = T.Resample(sr_vctk, 16000, resampling_method='sinc_interpolation')
        wf_vctk = sample(wf_vctk)
        # dither
        wf_vctk = F.dither(wf_vctk, noise_shaping=True)

        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(test_filepath_vctk)
        E.append_effect_to_chain("gain", ["-h"])
        E.append_effect_to_chain("channels", [1])
        E.append_effect_to_chain("rate", [16000])
        E.append_effect_to_chain("gain", ["-rh"])
        E.append_effect_to_chain("dither", ["-s"])
        wf_vctk_sox = E.sox_build_flow_effects()[0]

        self.assertTrue(torch.allclose(wf_vctk, wf_vctk_sox, rtol=1e-03, atol=1e-03))

    def test_pitch(self):
        test_dirpath, test_dir = common_utils.create_temp_assets_dir()
        test_filepath_100 = os.path.join(test_dirpath, 'assets', "100Hz_44100Hz_16bit_05sec.wav")
        test_filepath_440 = os.path.join(test_dirpath, 'assets', "440Hz_44100Hz_16bit_05sec.wav")

        # Files from https://www.mediacollege.com/audio/tone/download/
        tests = [
            (test_filepath_100, 100),
            (test_filepath_440, 440),
        ]

        for filename, freq_ref in tests:
            waveform, sample_rate = torchaudio.load(filename)

            freq = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)

            threshold = 1
            s = ((freq - freq_ref).abs() > threshold).sum()
            self.assertFalse(s)

    def test_DB_to_amplitude(self):
        # Make some noise
        x = torch.rand(1000)
        spectrogram = torchaudio.transforms.Spectrogram()
        spec = spectrogram(x)

        amin = 1e-10
        ref = 1.0
        db_multiplier = math.log10(max(amin, ref))

        # Waveform amplitude -> DB -> amplitude
        multiplier = 20.
        power = 0.5

        db = F.amplitude_to_DB(torch.abs(x), multiplier, amin, db_multiplier, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, power)

        self.assertTrue(torch.allclose(torch.abs(x), x2, atol=5e-5))

        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, power)

        self.assertTrue(torch.allclose(spec, x2, atol=5e-5))

        # Waveform power -> DB -> power
        multiplier = 10.
        power = 1.

        db = F.amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, power)

        self.assertTrue(torch.allclose(torch.abs(x), x2, atol=5e-5))

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, power)

        self.assertTrue(torch.allclose(spec, x2, atol=5e-5))


@pytest.mark.parametrize('complex_tensor', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1025, 400, 2)
])
@pytest.mark.parametrize('power', [1, 2, 0.7])
def test_complex_norm(complex_tensor, power):
    expected_norm_tensor = complex_tensor.pow(2).sum(-1).pow(power / 2)
    norm_tensor = F.complex_norm(complex_tensor, power)

    assert torch.allclose(expected_norm_tensor, norm_tensor, atol=1e-5)


@pytest.mark.parametrize('specgram', [
    torch.randn(2, 1025, 400),
    torch.randn(1, 201, 100)
])
@pytest.mark.parametrize('mask_param', [100])
@pytest.mark.parametrize('mask_value', [0., 30.])
@pytest.mark.parametrize('axis', [1, 2])
def test_mask_along_axis(specgram, mask_param, mask_value, axis):

    mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis)

    other_axis = 1 if axis == 2 else 2

    masked_columns = (mask_specgram == mask_value).sum(other_axis)
    num_masked_columns = (masked_columns == mask_specgram.size(other_axis)).sum()
    num_masked_columns /= mask_specgram.size(0)

    assert mask_specgram.size() == specgram.size()
    assert num_masked_columns < mask_param


@pytest.mark.parametrize('specgrams', [
    torch.randn(4, 2, 1025, 400),
])
@pytest.mark.parametrize('mask_param', [100])
@pytest.mark.parametrize('mask_value', [0., 30.])
@pytest.mark.parametrize('axis', [2, 3])
def test_mask_along_axis_iid(specgrams, mask_param, mask_value, axis):

    mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

    other_axis = 2 if axis == 3 else 3

    masked_columns = (mask_specgrams == mask_value).sum(other_axis)
    num_masked_columns = (masked_columns == mask_specgrams.size(other_axis)).sum(-1)

    assert mask_specgrams.size() == specgrams.size()
    assert (num_masked_columns < mask_param).sum() == num_masked_columns.numel()


if __name__ == '__main__':
    unittest.main()
