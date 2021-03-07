import itertools
import unittest
from distutils.version import StrictVersion

import torch
from parameterized import parameterized, param

import torchaudio.functional as F
from torchaudio._internal.module_utils import is_module_available

LIBROSA_AVAILABLE = is_module_available('librosa')

if LIBROSA_AVAILABLE:
    import numpy as np
    import librosa

from torchaudio_unittest import common_utils


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TestFunctional(common_utils.TorchaudioTestCase):
    """Test suite for functions in `functional` module."""
    def test_griffinlim(self):
        # NOTE: This test is flaky without a fixed random seed
        # See https://github.com/pytorch/audio/issues/382
        torch.random.manual_seed(42)
        tensor = torch.rand((1, 1000))

        n_fft = 400
        ws = 400
        hop = 100
        window = torch.hann_window(ws)
        momentum = 0.99
        n_iter = 8
        length = 1000
        rand_init = False
        init = 'random' if rand_init else None

        specgram = F.spectrogram(tensor, 0, window, n_fft, hop, ws, 2, normalize).sqrt()
        ta_out = F.griffinlim(specgram, window, n_fft, hop, ws, 1,
                              n_iter, momentum, length, rand_init)
        lr_out = librosa.griffinlim(specgram.squeeze(0).numpy(), n_iter=n_iter, hop_length=hop,
                                    momentum=momentum, init=init, length=length)
        lr_out = torch.from_numpy(lr_out).unsqueeze(0)

        self.assertEqual(ta_out, lr_out, atol=5e-5, rtol=1e-5)

    @parameterized.expand([
        param(norm=norm, mel_scale=mel_scale, **p.kwargs)
        for p in [
            param(),
            param(n_mels=128, sample_rate=44100),
            param(n_mels=128, fmin=2000.0, fmax=5000.0),
            param(n_mels=56, fmin=100.0, fmax=9000.0),
            param(n_mels=56, fmin=800.0, fmax=900.0),
            param(n_mels=56, fmin=1900.0, fmax=900.0),
            param(n_mels=10, fmin=1900.0, fmax=900.0),
        ]
        for norm in [None, 'slaney']
        for mel_scale in ['htk', 'slaney']
    ])
    def test_create_fb(self, n_mels=40, sample_rate=22050, n_fft=2048,
                       fmin=0.0, fmax=8000.0, norm=None, mel_scale="htk"):
        if (norm == "slaney" and StrictVersion(librosa.__version__) < StrictVersion("0.7.2")):
            self.skipTest('Test is known to fail with older versions of librosa.')

        librosa_fb = librosa.filters.mel(sr=sample_rate,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         fmax=fmax,
                                         fmin=fmin,
                                         htk=mel_scale == "htk",
                                         norm=norm)
        fb = F.create_fb_matrix(sample_rate=sample_rate,
                                n_mels=n_mels,
                                f_max=fmax,
                                f_min=fmin,
                                n_freqs=(n_fft // 2 + 1),
                                norm=norm,
                                mel_scale=mel_scale)

        for i_mel_bank in range(n_mels):
            self.assertEqual(
                fb[:, i_mel_bank], torch.tensor(librosa_fb[i_mel_bank]), atol=1e-4, rtol=1e-5)

    def test_amplitude_to_DB(self):
        spec = torch.rand((6, 201))

        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0

        # Power to DB
        multiplier = 10.0

        ta_out = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        lr_out = librosa.core.power_to_db(spec.numpy())
        lr_out = torch.from_numpy(lr_out)

        self.assertEqual(ta_out, lr_out, atol=5e-5, rtol=1e-5)

        # Amplitude to DB
        multiplier = 20.0

        ta_out = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        lr_out = librosa.core.amplitude_to_db(spec.numpy())
        lr_out = torch.from_numpy(lr_out)

        self.assertEqual(ta_out, lr_out, atol=5e-5, rtol=1e-5)


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TestPhaseVocoder(common_utils.TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        [(2, 1025, 400, 2)],
        [0.5, 1.01, 1.3],
        [256]
    )))
    def test_phase_vocoder(self, shape, rate, hop_length):
        # Due to cummulative sum, numerical error in using torch.float32 will
        # result in bottom right values of the stretched sectrogram to not
        # match with librosa.
        torch.random.manual_seed(42)
        complex_specgrams = torch.randn(*shape)
        complex_specgrams = complex_specgrams.type(torch.float64)
        phase_advance = torch.linspace(
            0,
            np.pi * hop_length,
            complex_specgrams.shape[-3],
            dtype=torch.float64)[..., None]

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
        expected_complex_stretch = librosa.phase_vocoder(
            mono_complex_specgram,
            rate=rate,
            hop_length=hop_length)

        complex_stretch = complex_specgrams_stretch[index].numpy()
        complex_stretch = complex_stretch[..., 0] + 1j * complex_stretch[..., 1]

        self.assertEqual(complex_stretch, torch.from_numpy(expected_complex_stretch), atol=1e-5, rtol=1e-5)
