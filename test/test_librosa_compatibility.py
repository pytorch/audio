"""Test suites for numerical compatibility with librosa"""
import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.common_utils import IMPORT_LIBROSA

if IMPORT_LIBROSA:
    import numpy as np
    import librosa
    import scipy

import pytest

import common_utils


class _LibrosaMixin:
    """Automatically skip tests if librosa is not available"""
    def setUp(self):
        super().setUp()
        if not IMPORT_LIBROSA:
            raise unittest.SkipTest('Librosa not available')


class TestFunctional(_LibrosaMixin, unittest.TestCase):
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

        torch.testing.assert_allclose(ta_out, lr_out, atol=5e-5, rtol=1e-5)

    def _test_create_fb(self, n_mels=40, sample_rate=22050, n_fft=2048, fmin=0.0, fmax=8000.0):
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
            torch.testing.assert_allclose(fb[:, i_mel_bank], torch.tensor(librosa_fb[i_mel_bank]),
                                          atol=1e-4, rtol=1e-5)

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
        lr_out = torch.from_numpy(lr_out)

        torch.testing.assert_allclose(ta_out, lr_out, atol=5e-5, rtol=1e-5)

        # Amplitude to DB
        multiplier = 20.0

        ta_out = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier, top_db)
        lr_out = librosa.core.amplitude_to_db(spec.numpy())
        lr_out = torch.from_numpy(lr_out)

        torch.testing.assert_allclose(ta_out, lr_out, atol=5e-5, rtol=1e-5)


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


def _load_audio_asset(*asset_paths, **kwargs):
    file_path = common_utils.get_asset_path(*asset_paths)
    sound, sample_rate = torchaudio.load(file_path, **kwargs)
    return sound, sample_rate


def _test_compatibilities(n_fft, hop_length, power, n_mels, n_mfcc, sample_rate):
    sound, sample_rate = _load_audio_asset('sinewave.wav')
    sound = sound.double()
    sound_librosa = sound.cpu().numpy().squeeze()  # (64000)

    # test core spectrogram
    spect_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=power)
    out_librosa, _ = librosa.core.spectrum._spectrogram(
        y=sound_librosa, n_fft=n_fft, hop_length=hop_length, power=power)
    '''
    out_torch = spect_transform(sound).squeeze().cpu()
    torch.testing.assert_allclose(out_torch, torch.from_numpy(out_librosa), atol=1e-5, rtol=1e-5)
    '''

    '''
    # test mel spectrogram
    melspect_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, window_fn=torch.hann_window,
        hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)
    librosa_mel = librosa.feature.melspectrogram(
        y=sound_librosa, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels, htk=True, norm=None)
    librosa_mel_tensor = torch.from_numpy(librosa_mel)
    torch_mel = melspect_transform(sound).squeeze().cpu()
    torch.testing.assert_allclose(
        torch_mel.type(librosa_mel_tensor.dtype), librosa_mel_tensor, atol=5e-3, rtol=1e-5)
    '''

    # test s2db
    power_to_db_transform = torchaudio.transforms.AmplitudeToDB('power', 80.)
    power_to_db_torch = power_to_db_transform(spect_transform(sound)).squeeze().cpu()
    power_to_db_librosa = librosa.core.spectrum.power_to_db(out_librosa)
    torch.testing.assert_allclose(
        power_to_db_torch, torch.from_numpy(power_to_db_librosa).double(), atol=5e-3, rtol=1e-5)

    '''
    mag_to_db_transform = torchaudio.transforms.AmplitudeToDB('magnitude', 80.)
    mag_to_db_torch = mag_to_db_transform(torch.abs(sound)).squeeze().cpu()
    mag_to_db_librosa = librosa.core.spectrum.amplitude_to_db(sound_librosa)
    torch.testing.assert_allclose(mag_to_db_torch, torch.from_numpy(mag_to_db_librosa), atol=5e-3, rtol=1e-5)

    power_to_db_torch = power_to_db_transform(melspect_transform(sound)).squeeze().cpu()
    db_librosa = librosa.core.spectrum.power_to_db(librosa_mel)
    db_librosa_tensor = torch.from_numpy(db_librosa)
    torch.testing.assert_allclose(
        power_to_db_torch.type(db_librosa_tensor.dtype), db_librosa_tensor, atol=5e-3, rtol=1e-5)
    '''

    '''
    # test MFCC
    melkwargs = {'hop_length': hop_length, 'n_fft': n_fft}
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=n_mfcc, norm='ortho', melkwargs=melkwargs)

    # librosa.feature.mfcc doesn't pass kwargs properly since some of the
    # kwargs for melspectrogram and mfcc are the same. We just follow the
    # function body in
    # https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#melspectrogram
    # to mirror this function call with correct args:
    #
    # librosa_mfcc = librosa.feature.mfcc(
    #     y=sound_librosa, sr=sample_rate, n_mfcc = n_mfcc,
    #     hop_length=hop_length, n_fft=n_fft, htk=True, norm=None, n_mels=n_mels)

    librosa_mfcc = scipy.fftpack.dct(db_librosa, axis=0, type=2, norm='ortho')[:n_mfcc]
    librosa_mfcc_tensor = torch.from_numpy(librosa_mfcc)
    torch_mfcc = mfcc_transform(sound).squeeze().cpu()

    torch.testing.assert_allclose(
        torch_mfcc.type(librosa_mfcc_tensor.dtype), librosa_mfcc_tensor, atol=5e-3, rtol=1e-5)
    '''


class TestTransforms(_LibrosaMixin, unittest.TestCase):
    """Test suite for functions in `transforms` module."""
    def test_basics1(self):
        kwargs = {
            'n_fft': 400,
            'hop_length': 200,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 40,
            'sample_rate': 16000
        }
        _test_compatibilities(**kwargs)

    def test_basics2(self):
        kwargs = {
            'n_fft': 600,
            'hop_length': 100,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 20,
            'sample_rate': 16000
        }
        _test_compatibilities(**kwargs)

    # NOTE: Test passes offline, but fails on TravisCI (and CircleCI), see #372.
    @pytest.mark.xfail
    def test_basics3(self):
        kwargs = {
            'n_fft': 200,
            'hop_length': 50,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 50,
            'sample_rate': 24000
        }
        _test_compatibilities(**kwargs)

    def test_basics4(self):
        kwargs = {
            'n_fft': 400,
            'hop_length': 200,
            'power': 3.0,
            'n_mels': 128,
            'n_mfcc': 40,
            'sample_rate': 16000
        }
        _test_compatibilities(**kwargs)

    @unittest.skipIf("sox" not in common_utils.BACKENDS, "sox not available")
    @common_utils.AudioBackendScope("sox")
    def test_MelScale(self):
        """MelScale transform is comparable to that of librosa"""
        n_fft = 2048
        n_mels = 256
        hop_length = n_fft // 4

        # Prepare spectrogram input. We use torchaudio to compute one.
        sound, sample_rate = _load_audio_asset('whitenoise_1min.mp3')
        sound = sound.mean(dim=0, keepdim=True)
        spec_ta = F.spectrogram(
            sound, pad=0, window=torch.hann_window(n_fft), n_fft=n_fft,
            hop_length=hop_length, win_length=n_fft, power=2, normalized=False)
        spec_lr = spec_ta.cpu().numpy().squeeze()
        # Perform MelScale with torchaudio and librosa
        melspec_ta = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate)(spec_ta)
        melspec_lr = librosa.feature.melspectrogram(
            S=spec_lr, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, center=True, window='hann', n_mels=n_mels, htk=True, norm=None)
        # Note: Using relaxed rtol instead of atol
        torch.testing.assert_allclose(melspec_ta, torch.from_numpy(melspec_lr[None, ...]), atol=1e-8, rtol=1e-3)

    def test_InverseMelScale(self):
        """InverseMelScale transform is comparable to that of librosa"""
        n_fft = 2048
        n_mels = 256
        n_stft = n_fft // 2 + 1
        hop_length = n_fft // 4

        # Prepare mel spectrogram input. We use torchaudio to compute one.
        sound, sample_rate = _load_audio_asset(
            'steam-train-whistle-daniel_simon.wav', offset=2**10, num_frames=2**14)
        sound = sound.mean(dim=0, keepdim=True)
        spec_orig = F.spectrogram(
            sound, pad=0, window=torch.hann_window(n_fft), n_fft=n_fft,
            hop_length=hop_length, win_length=n_fft, power=2, normalized=False)
        melspec_ta = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate)(spec_orig)
        melspec_lr = melspec_ta.cpu().numpy().squeeze()
        # Perform InverseMelScale with torch audio and librosa
        spec_ta = torchaudio.transforms.InverseMelScale(
            n_stft, n_mels=n_mels, sample_rate=sample_rate)(melspec_ta)
        spec_lr = librosa.feature.inverse.mel_to_stft(
            melspec_lr, sr=sample_rate, n_fft=n_fft, power=2.0, htk=True, norm=None)
        spec_lr = torch.from_numpy(spec_lr[None, ...])

        # Align dimensions
        # librosa does not return power spectrogram while torchaudio returns power spectrogram
        spec_orig = spec_orig.sqrt()
        spec_ta = spec_ta.sqrt()

        threshold = 2.0
        # This threshold was choosen empirically, based on the following observation
        #
        # torch.dist(spec_lr, spec_ta, p=float('inf'))
        # >>> tensor(1.9666)
        #
        # The spectrograms reconstructed by librosa and torchaudio are not comparable elementwise.
        # This is because they use different approximation algorithms and resulting values can live
        # in different magnitude. (although most of them are very close)
        # See
        # https://github.com/pytorch/audio/pull/366 for the discussion of the choice of algorithm
        # https://github.com/pytorch/audio/pull/448/files#r385747021 for the distribution of P-inf
        # distance over frequencies.
        torch.testing.assert_allclose(spec_ta, spec_lr, atol=threshold, rtol=1e-5)

        threshold = 1700.0
        # This threshold was choosen empirically, based on the following observations
        #
        # torch.dist(spec_orig, spec_ta, p=1)
        # >>> tensor(1644.3516)
        # torch.dist(spec_orig, spec_lr, p=1)
        # >>> tensor(1420.7103)
        # torch.dist(spec_lr, spec_ta, p=1)
        # >>> tensor(943.2759)
        assert torch.dist(spec_orig, spec_ta, p=1) < threshold


if __name__ == '__main__':
    unittest.main()
