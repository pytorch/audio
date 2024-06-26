import unittest

import torch
import torchaudio.transforms as T
from parameterized import param, parameterized
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import get_sinusoid, get_spectrogram, get_whitenoise, nested_params, TestBaseMixin

LIBROSA_AVAILABLE = is_module_available("librosa")

if LIBROSA_AVAILABLE:
    import librosa


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TransformsTestBase(TestBaseMixin):
    @parameterized.expand(
        [
            param(n_fft=400, hop_length=200, power=2.0),
            param(n_fft=600, hop_length=100, power=2.0),
            param(n_fft=400, hop_length=200, power=3.0),
            param(n_fft=200, hop_length=50, power=2.0),
        ]
    )
    def test_Spectrogram(self, n_fft, hop_length, power):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate,
            n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(), n_fft=n_fft, hop_length=hop_length, power=power, pad_mode="reflect"
        )[0]

        result = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power,).to(self.device, self.dtype)(
            waveform
        )[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=1e-4, rtol=1e-4)

    def test_Spectrogram_complex(self):
        n_fft = 400
        hop_length = 200
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate,
            n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(), n_fft=n_fft, hop_length=hop_length, power=1, pad_mode="reflect"
        )[0]

        result = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None, return_complex=True,).to(
            self.device, self.dtype
        )(waveform)[0]
        self.assertEqual(result.abs(), torch.from_numpy(expected), atol=1e-4, rtol=1e-4)

    @nested_params(
        [
            param(n_fft=400, hop_length=200, n_mels=64),
            param(n_fft=600, hop_length=100, n_mels=128),
            param(n_fft=200, hop_length=50, n_mels=32),
        ],
        [param(norm=norm) for norm in [None, "slaney"]],
        [param(mel_scale=mel_scale) for mel_scale in ["htk", "slaney"]],
    )
    def test_MelSpectrogram(self, n_fft, hop_length, n_mels, norm, mel_scale):
        sample_rate = 16000
        waveform = get_sinusoid(
            sample_rate=sample_rate,
            n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm=norm,
            htk=mel_scale == "htk",
            pad_mode="reflect",
        )
        result = T.MelSpectrogram(
            sample_rate=sample_rate,
            window_fn=torch.hann_window,
            hop_length=hop_length,
            n_mels=n_mels,
            n_fft=n_fft,
            norm=norm,
            mel_scale=mel_scale,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)
    
    def test_magnitude_to_db(self):
        spectrogram = get_spectrogram(get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB("magnitude", 80.0).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.amplitude_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    def test_power_to_db(self):
        spectrogram = get_spectrogram(get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB("power", 80.0).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.power_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    @nested_params(
        [
            param(n_fft=400, hop_length=200, n_mels=64, n_mfcc=40),
            param(n_fft=600, hop_length=100, n_mels=128, n_mfcc=20),
            param(n_fft=200, hop_length=50, n_mels=32, n_mfcc=25),
        ]
    )
    def test_mfcc(self, n_fft, hop_length, n_mels, n_mfcc):
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)
        result = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            norm="ortho",
            melkwargs={"hop_length": hop_length, "n_fft": n_fft, "n_mels": n_mels},
        ).to(self.device, self.dtype)(waveform)[0]

        melspec = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            htk=True,
            norm=None,
            pad_mode="reflect",
        )
        expected = librosa.feature.mfcc(
            S=librosa.core.spectrum.power_to_db(melspec), n_mfcc=n_mfcc, dct_type=2, norm="ortho"
        )
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    @parameterized.expand(
        [
            param(n_fft=400, hop_length=200),
            param(n_fft=600, hop_length=100),
            param(n_fft=200, hop_length=50),
        ]
    )
    def test_spectral_centroid(self, n_fft, hop_length):
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)

        result = T.SpectralCentroid(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,).to(
            self.device, self.dtype
        )(waveform)
        expected = librosa.feature.spectral_centroid(
            y=waveform[0].cpu().numpy(), sr=sample_rate, n_fft=n_fft, hop_length=hop_length, pad_mode="reflect"
        )
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    @nested_params(
        [
            param(sample_rate=1000, hop_length=100, n_bins=36, bins_per_octave=12, gamma=2., atol=0.3, rtol=0.3),
            param(sample_rate=1000, hop_length=10, n_bins=3, bins_per_octave=1, gamma=4., atol=0.2, rtol=0.2),
            param(sample_rate=500, hop_length=50, n_bins=16, bins_per_octave=8, gamma=6., atol=0.2, rtol=0.2),
            param(sample_rate=250, hop_length=25, n_bins=4, bins_per_octave=4, gamma=8., atol=1e-7, rtol=1e-7),
        ],
    )
    def test_VQT(self, sample_rate, hop_length, n_bins, bins_per_octave, gamma, atol, rtol):
        """
        Differences in resampling, which occurs n_bins/bins_per_octave - 1 times, between torch and librosa
        lead to diverging VQTs. This is likely as close as it can get.
        """
        f_min = 32.703
        waveform = get_whitenoise(sample_rate=sample_rate, dtype=self.dtype).to(self.device)

        expected = librosa.core.constantq.vqt(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate,
            hop_length=hop_length,
            fmin=f_min,
            n_bins=n_bins,
            gamma=gamma,
            bins_per_octave=bins_per_octave,
            sparsity=0.,                        # torchaudio VQT implemeted with sparsity 0
            res_type="sinc_best",               # torchaudio resampling roughly equivalent to sinc_best
        )
        result = T.VQT(
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            n_bins=n_bins,
            gamma=gamma,
            bins_per_octave=bins_per_octave,
            dtype=self.dtype,
        ).to(self.device)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=atol, rtol=rtol)
    
    @nested_params(
        [
            param(sample_rate=1000, hop_length=100, n_bins=36, bins_per_octave=12, atol=0.3, rtol=0.3),
            param(sample_rate=1000, hop_length=10, n_bins=3, bins_per_octave=1, atol=0.2, rtol=0.2),
            param(sample_rate=500, hop_length=50, n_bins=16, bins_per_octave=8, atol=0.2, rtol=0.2),
            param(sample_rate=250, hop_length=25, n_bins=4, bins_per_octave=4, atol=1e-7, rtol=1e-7),
        ],
    )
    def test_CQT(self, sample_rate, hop_length, n_bins, bins_per_octave, atol, rtol):
        """
        Differences in resampling, which occurs n_bins/bins_per_octave - 1 times, between torch and librosa
        lead to diverging CQTs. This is likely as close as it can get.
        """
        f_min = 32.703
        waveform = get_whitenoise(sample_rate=sample_rate, duration=2, dtype=self.dtype).to(self.device)

        expected = librosa.cqt(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate,
            hop_length=hop_length,
            fmin=f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            sparsity=0.,                        # torchaudio CQT implemeted with sparsity 0
            res_type="sinc_best",               # torchaudio resampling roughly equivalent to sinc_best
        )
        result = T.CQT(
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            dtype=self.dtype,
        ).to(self.device)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=atol, rtol=rtol)
    
    @nested_params(
        [
            param(sample_rate=1000, hop_length=100, n_bins=36, bins_per_octave=12, atol=0.02, rtol=0.02),
            param(sample_rate=1000, hop_length=10, n_bins=3, bins_per_octave=1, atol=0.01, rtol=0.01),
            param(sample_rate=500, hop_length=50, n_bins=16, bins_per_octave=8, atol=0.01, rtol=0.01),
            param(sample_rate=250, hop_length=25, n_bins=4, bins_per_octave=4, atol=1e-7, rtol=1e-7),
        ],
    )
    def test_InverseCQT(self, sample_rate, hop_length, n_bins, bins_per_octave, atol, rtol):
        """
        Differences in resampling, which occurs n_bins/bins_per_octave - 1 times, between torch and librosa
        lead to diverging iCQTs. This is likely as close as it can get.
        """
        f_min = 32.703
        waveform = get_whitenoise(sample_rate=sample_rate, duration=4, dtype=self.dtype).to(self.device)
        
        cqt = T.CQT(
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            dtype=self.dtype,
        ).to(self.device)(waveform)
        
        expected = librosa.core.icqt(
            C=cqt[0].cpu().numpy(),
            sr=sample_rate,
            hop_length=hop_length,
            fmin=f_min,
            bins_per_octave=bins_per_octave,
            sparsity=0.,                        # torchaudio iCQT implemeted with sparsity 0
            res_type="sinc_best",               # torchaudio resampling roughly equivalent to sinc_best
        )
        result = T.InverseCQT(
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            dtype=self.dtype,
        ).to(self.device)(cqt)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=atol, rtol=rtol)
