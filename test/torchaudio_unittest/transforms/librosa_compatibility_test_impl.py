import unittest

import torch
import torchaudio.transforms as T
from torchaudio._internal.module_utils import is_module_available
from parameterized import param, parameterized

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_sinusoid,
    get_spectrogram,
    nested_params,
)

LIBROSA_AVAILABLE = is_module_available('librosa')

if LIBROSA_AVAILABLE:
    import librosa


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TransformsTestBase(TestBaseMixin):
    @parameterized.expand([
        param(n_fft=400, hop_length=200, power=2.0),
        param(n_fft=600, hop_length=100, power=2.0),
        param(n_fft=400, hop_length=200, power=3.0),
        param(n_fft=200, hop_length=50, power=2.0),
    ])
    def test_Spectrogram(self, n_fft, hop_length, power):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(),
            n_fft=n_fft, hop_length=hop_length, power=power)[0]

        result = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=power,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    def test_Spectrogram_complex(self):
        n_fft = 400
        hop_length = 200
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(),
            n_fft=n_fft, hop_length=hop_length, power=1)[0]

        result = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, return_complex=True,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result.abs(), torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    @nested_params(
        [
            param(n_fft=400, hop_length=200, n_mels=64),
            param(n_fft=600, hop_length=100, n_mels=128),
            param(n_fft=200, hop_length=50, n_mels=32),
        ],
        [param(norm=norm) for norm in [None, 'slaney']],
        [param(mel_scale=mel_scale) for mel_scale in ['htk', 'slaney']],
    )
    def test_MelSpectrogram(self, n_fft, hop_length, n_mels, norm, mel_scale):
        sample_rate = 16000
        waveform = get_sinusoid(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, norm=norm,
            htk=mel_scale == "htk")
        result = T.MelSpectrogram(
            sample_rate=sample_rate, window_fn=torch.hann_window,
            hop_length=hop_length, n_mels=n_mels,
            n_fft=n_fft, norm=norm, mel_scale=mel_scale,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    def test_magnitude_to_db(self):
        spectrogram = get_spectrogram(
            get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB('magnitude', 80.).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.amplitude_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    def test_power_to_db(self):
        spectrogram = get_spectrogram(
            get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB('power', 80.).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.power_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    @nested_params([
        param(n_fft=400, hop_length=200, n_mels=64, n_mfcc=40),
        param(n_fft=600, hop_length=100, n_mels=128, n_mfcc=20),
        param(n_fft=200, hop_length=50, n_mels=32, n_mfcc=25),
    ])
    def test_mfcc(self, n_fft, hop_length, n_mels, n_mfcc):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)
        result = T.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, norm='ortho',
            melkwargs={'hop_length': hop_length, 'n_fft': n_fft, 'n_mels': n_mels},
        ).to(self.device, self.dtype)(waveform)[0]

        melspec = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(), sr=sample_rate, n_fft=n_fft,
            win_length=n_fft, hop_length=hop_length,
            n_mels=n_mels, htk=True, norm=None)
        expected = librosa.feature.mfcc(
            S=librosa.core.spectrum.power_to_db(melspec),
            n_mfcc=n_mfcc, dct_type=2, norm='ortho')
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    @parameterized.expand([
        param(n_fft=400, hop_length=200),
        param(n_fft=600, hop_length=100),
        param(n_fft=200, hop_length=50),
    ])
    def test_spectral_centroid(self, n_fft, hop_length):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)

        result = T.SpectralCentroid(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        ).to(self.device, self.dtype)(waveform)
        expected = librosa.feature.spectral_centroid(
            y=waveform[0].cpu().numpy(), sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)
