"""Test suites for numerical compatibility with librosa"""

import unittest

import torch
import torchaudio.functional as F
from torchaudio.common_utils import IMPORT_LIBROSA

if IMPORT_LIBROSA:
    import numpy as np
    import librosa

import pytest


class TestFunctional(unittest.TestCase):
    def setUp(self):
        if not IMPORT_LIBROSA:
            raise unittest.SkipTest('Librosa not available')

    def test_griffinlim(self):

        # NOTE: This test is flaky without a fixed random seed
        # See https://github.com/pytorch/audio/issues/382
        torch.random.manual_seed(42)
        tensor = torch.rand((1, 1000))

        n_fft = 400
        ws = 400
        hop = 100
        window = torch.hann_window(ws)
        normalize = False
        momentum = 0.99
        n_iter = 8
        length = 1000
        rand_init = False
        init = 'random' if rand_init else None

        specgram = F.spectrogram(tensor, 0, window, n_fft, hop, ws, 2, normalize).sqrt()
        ta_out = F.griffinlim(specgram, window, n_fft, hop, ws, 1, normalize,
                              n_iter, momentum, length, rand_init)
        lr_out = librosa.griffinlim(specgram.squeeze(0).numpy(), n_iter=n_iter, hop_length=hop,
                                    momentum=momentum, init=init, length=length)
        lr_out = torch.from_numpy(lr_out).unsqueeze(0)

        assert torch.allclose(ta_out, lr_out, atol=5e-5)

    def _test_create_fb(self, n_mels=40, sample_rate=22050, n_fft=2048, fmin=0.0, fmax=8000.0):
        # Using a decorator here causes parametrize to fail on Python 2
        if not IMPORT_LIBROSA:
            raise unittest.SkipTest('Librosa is not available')

        librosa_fb = librosa.filters.mel(sr=sample_rate,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         fmax=fmax,
                                         fmin=fmin,
                                         htk=True,
                                         norm=None)
        fb = F.create_fb_matrix(sample_rate=sample_rate,
                                n_mels=n_mels,
                                f_max=fmax,
                                f_min=fmin,
                                n_freqs=(n_fft // 2 + 1))

        for i_mel_bank in range(n_mels):
            assert torch.allclose(fb[:, i_mel_bank], torch.tensor(librosa_fb[i_mel_bank]), atol=1e-4)

    def test_create_fb(self):
        self._test_create_fb()
        self._test_create_fb(n_mels=128, sample_rate=44100)
        self._test_create_fb(n_mels=128, fmin=2000.0, fmax=5000.0)
        self._test_create_fb(n_mels=56, fmin=100.0, fmax=9000.0)
        self._test_create_fb(n_mels=56, fmin=800.0, fmax=900.0)
        self._test_create_fb(n_mels=56, fmin=1900.0, fmax=900.0)
        self._test_create_fb(n_mels=10, fmin=1900.0, fmax=900.0)

    def test_amplitude_to_DB(self):
        spec = torch.rand((6, 201))

        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0

        # Power to DB
        multiplier = 10.0

        ta_out = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        lr_out = librosa.core.power_to_db(spec.numpy())
        lr_out = torch.from_numpy(lr_out).unsqueeze(0)

        assert torch.allclose(ta_out, lr_out, atol=5e-5)

        # Amplitude to DB
        multiplier = 20.0

        ta_out = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        lr_out = librosa.core.amplitude_to_db(spec.numpy())
        lr_out = torch.from_numpy(lr_out).unsqueeze(0)

        assert torch.allclose(ta_out, lr_out, atol=5e-5)


@pytest.mark.parametrize('complex_specgrams', [
    torch.randn(2, 1025, 400, 2)
])
@pytest.mark.parametrize('rate', [0.5, 1.01, 1.3])
@pytest.mark.parametrize('hop_length', [256])
def test_phase_vocoder(complex_specgrams, rate, hop_length):

    # Using a decorator here causes parametrize to fail on Python 2
    if not IMPORT_LIBROSA:
        raise unittest.SkipTest('Librosa is not available')

    # Due to cummulative sum, numerical error in using torch.float32 will
    # result in bottom right values of the stretched sectrogram to not
    # match with librosa.

    complex_specgrams = complex_specgrams.type(torch.float64)
    phase_advance = torch.linspace(0, np.pi * hop_length, complex_specgrams.shape[-3], dtype=torch.float64)[..., None]

    complex_specgrams_stretch = F.phase_vocoder(complex_specgrams, rate=rate, phase_advance=phase_advance)

    # == Test shape
    expected_size = list(complex_specgrams.size())
    expected_size[-2] = int(np.ceil(expected_size[-2] / rate))

    assert complex_specgrams.dim() == complex_specgrams_stretch.dim()
    assert complex_specgrams_stretch.size() == torch.Size(expected_size)

    # == Test values
    index = [0] * (complex_specgrams.dim() - 3) + [slice(None)] * 3
    mono_complex_specgram = complex_specgrams[index].numpy()
    mono_complex_specgram = mono_complex_specgram[..., 0] + \
        mono_complex_specgram[..., 1] * 1j
    expected_complex_stretch = librosa.phase_vocoder(mono_complex_specgram,
                                                     rate=rate,
                                                     hop_length=hop_length)

    complex_stretch = complex_specgrams_stretch[index].numpy()
    complex_stretch = complex_stretch[..., 0] + 1j * complex_stretch[..., 1]

    assert np.allclose(complex_stretch, expected_complex_stretch, atol=1e-5)
