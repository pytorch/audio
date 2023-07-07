import unittest

import torch
import torchaudio.prototype.functional as F
from torchaudio._internal.module_utils import is_module_available

LIBROSA_AVAILABLE = is_module_available("librosa")

if LIBROSA_AVAILABLE:
    import librosa
    import numpy as np


from torchaudio_unittest.common_utils import TestBaseMixin


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class Functional(TestBaseMixin):
    """Test suite for functions in `functional` module."""

    dtype = torch.float64

    def test_chroma_filterbank(self):
        sample_rate = 16_000
        n_stft = 400
        n_chroma = 12
        tuning = 0.0
        ctroct = 5.0
        octwidth = 2.0
        norm = 2
        base_c = True

        # NOTE: difference in convention with librosa.
        # Whereas librosa expects users to supply the full count of FFT frequency bins,
        # TorchAudio expects users to supply the count with redundant bins, i.e. those in the upper half of the
        # frequency range, removed. This is consistent with other TorchAudio filter bank functions.
        n_freqs = n_stft // 2 + 1

        torchaudio_fbank = F.chroma_filterbank(
            sample_rate=sample_rate,
            n_freqs=n_freqs,
            n_chroma=n_chroma,
            tuning=tuning,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
            base_c=base_c,
        )

        librosa_fbank = librosa.filters.chroma(
            sr=sample_rate,
            n_fft=n_stft,
            n_chroma=n_chroma,
            tuning=tuning,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
            base_c=True,
            dtype=np.float32,
        )

        self.assertEqual(torchaudio_fbank, librosa_fbank.T)
