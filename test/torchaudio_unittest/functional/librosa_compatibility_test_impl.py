import unittest
from distutils.version import StrictVersion

import torch
from parameterized import param

import torchaudio.functional as F
from torchaudio._internal.module_utils import is_module_available

LIBROSA_AVAILABLE = is_module_available('librosa')

if LIBROSA_AVAILABLE:
    import numpy as np
    import librosa


from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    nested_params,
    get_whitenoise,
    get_spectrogram,
)


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class Functional(TestBaseMixin):
    """Test suite for functions in `functional` module."""
    dtype = torch.float64

    @nested_params([0, 0.99])
    def test_griffinlim(self, momentum):
        # FFT params
        n_fft = 400
        win_length = n_fft
        hop_length = n_fft // 4
        window = torch.hann_window(win_length)
        power = 1
        # GriffinLim params
        n_iter = 8

        waveform = get_whitenoise(device=self.device, dtype=self.dtype)
        specgram = get_spectrogram(
            waveform, n_fft=n_fft, hop_length=hop_length, power=power,
            win_length=win_length, window=window)

        result = F.griffinlim(
            specgram,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            n_iter=n_iter,
            momentum=momentum,
            length=waveform.size(1),
            rand_init=False)
        expected = librosa.griffinlim(
            specgram[0].cpu().numpy(),
            n_iter=n_iter,
            hop_length=hop_length,
            momentum=momentum,
            init=None,
            length=waveform.size(1))[None, ...]
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-5, rtol=1e-07)

    @nested_params(
        [
            param(),
            param(n_mels=128, sample_rate=44100),
            param(n_mels=128, fmin=2000.0, fmax=5000.0),
            param(n_mels=56, fmin=100.0, fmax=9000.0),
            param(n_mels=56, fmin=800.0, fmax=900.0),
            param(n_mels=56, fmin=1900.0, fmax=900.0),
            param(n_mels=10, fmin=1900.0, fmax=900.0),
        ],
        [param(norm=n) for n in [None, 'slaney']],
        [param(mel_scale=s) for s in ['htk', 'slaney']],
    )
    def test_create_fb(self, n_mels=40, sample_rate=22050, n_fft=2048,
                       fmin=0.0, fmax=8000.0, norm=None, mel_scale="htk"):
        if (norm == "slaney" and StrictVersion(librosa.__version__) < StrictVersion("0.7.2")):
            self.skipTest('Test is known to fail with older versions of librosa.')
        if self.device != 'cpu':
            self.skipTest('No need to run this test on CUDA')

        expected = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmax=fmax,
            fmin=fmin,
            htk=mel_scale == "htk",
            norm=norm).T
        result = F.create_fb_matrix(
            sample_rate=sample_rate,
            n_mels=n_mels,
            f_max=fmax,
            f_min=fmin,
            n_freqs=(n_fft // 2 + 1),
            norm=norm,
            mel_scale=mel_scale)
        self.assertEqual(result, torch.from_numpy(expected), atol=7e-5, rtol=1.3e-6)

    def test_amplitude_to_DB_power(self):
        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0
        multiplier = 10.0

        spec = get_spectrogram(get_whitenoise(device=self.device, dtype=self.dtype), power=2)
        result = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        expected = librosa.core.power_to_db(spec[0].cpu().numpy())[None, ...]
        self.assertEqual(result, torch.from_numpy(expected))

    def test_amplitude_to_DB(self):
        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0
        multiplier = 20.0

        spec = get_spectrogram(get_whitenoise(device=self.device, dtype=self.dtype), power=1)
        result = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        expected = librosa.core.amplitude_to_db(spec[0].cpu().numpy())[None, ...]
        self.assertEqual(result, torch.from_numpy(expected))


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class FunctionalComplex(TestBaseMixin):
    @nested_params(
        [0.5, 1.01, 1.3],
        [True, False],
    )
    def test_phase_vocoder(self, rate, test_pseudo_complex):
        hop_length = 256
        num_freq = 1025
        num_frames = 400
        torch.random.manual_seed(42)

        # Due to cummulative sum, numerical error in using torch.float32 will
        # result in bottom right values of the stretched sectrogram to not
        # match with librosa.
        spec = torch.randn(num_freq, num_frames, device=self.device, dtype=torch.complex128)
        phase_advance = torch.linspace(
            0,
            np.pi * hop_length,
            num_freq,
            device=self.device,
            dtype=torch.float64)[..., None]

        stretched = F.phase_vocoder(
            torch.view_as_real(spec) if test_pseudo_complex else spec,
            rate=rate, phase_advance=phase_advance)

        expected_stretched = librosa.phase_vocoder(
            spec.cpu().numpy(),
            rate=rate,
            hop_length=hop_length)

        self.assertEqual(
            torch.view_as_complex(stretched) if test_pseudo_complex else stretched,
            torch.from_numpy(expected_stretched))
