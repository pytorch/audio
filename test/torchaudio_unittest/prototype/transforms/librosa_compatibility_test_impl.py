import unittest

import torch
import torchaudio.prototype.transforms as T
from parameterized import param
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import get_sinusoid, nested_params, TestBaseMixin

LIBROSA_AVAILABLE = is_module_available("librosa")

if LIBROSA_AVAILABLE:
    import librosa


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TransformsTestBase(TestBaseMixin):
    @nested_params(
        [
            param(n_fft=400, hop_length=200, n_chroma=13),
            param(n_fft=600, hop_length=100, n_chroma=24),
            param(n_fft=200, hop_length=50, n_chroma=12),
        ],
    )
    def test_chroma_spectrogram(self, n_fft, hop_length, n_chroma):
        sample_rate = 16000
        waveform = get_sinusoid(
            sample_rate=sample_rate,
            n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.feature.chroma_stft(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma,
            norm=None,
            pad_mode="reflect",
            tuning=0.0,
        )
        result = T.ChromaSpectrogram(
            sample_rate=sample_rate,
            window_fn=torch.hann_window,
            hop_length=hop_length,
            n_chroma=n_chroma,
            n_fft=n_fft,
            tuning=0.0,
        ).to(self.device, self.dtype)(waveform)[0]

        self.assertEqual(result, expected, atol=5e-4, rtol=1e-4)
