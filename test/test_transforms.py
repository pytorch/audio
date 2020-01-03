from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os

import torch
import torchaudio
import torchaudio.transforms as transforms
import torchaudio.functional as F
from torchaudio.common_utils import IMPORT_LIBROSA, IMPORT_SCIPY
import unittest
import common_utils

if IMPORT_LIBROSA:
    import librosa

if IMPORT_SCIPY:
    import scipy

SKIP_LIBROSA_CONSISTENCY_TEST = False
RUN_CUDA = torch.cuda.is_available()
print("Run test with cuda:", RUN_CUDA)


def _test_script_module(f, tensor, *args, **kwargs):

    py_method = f(*args, **kwargs)
    jit_method = torch.jit.script(py_method)

    py_out = py_method(tensor)
    jit_out = jit_method(tensor)

    assert torch.allclose(jit_out, py_out)

    if RUN_CUDA:

        tensor = tensor.to("cuda")

        py_method = py_method.cuda()
        jit_method = torch.jit.script(py_method)

        py_out = py_method(tensor)
        jit_out = jit_method(tensor)

        assert torch.allclose(jit_out, py_out)


class Tester(unittest.TestCase):

    # create a sinewave signal for testing
    sample_rate = 16000
    freq = 440
    volume = .3
    waveform = (torch.cos(2 * math.pi * torch.arange(0, 4 * sample_rate).float() * freq / sample_rate))
    waveform.unsqueeze_(0)  # (1, 64000)
    waveform = (waveform * volume * 2**31).long()
    # file for stereo stft test
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()
    test_filepath = os.path.join(test_dirpath, 'assets',
                                 'steam-train-whistle-daniel_simon.mp3')

    def scale(self, waveform, factor=float(2**31)):
        # scales a waveform by a factor
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.get_default_dtype())
        return waveform / factor

    def test_scriptmodule_Spectrogram(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(transforms.Spectrogram, tensor)

    def test_scriptmodule_GriffinLim(self):
        tensor = torch.rand((1, 201, 6))
        _test_script_module(transforms.GriffinLim, tensor, length=1000, rand_init=False)

    def test_mu_law_companding(self):

        quantization_channels = 256

        waveform = self.waveform.clone()
        waveform /= torch.abs(waveform).max()
        self.assertTrue(waveform.min() >= -1. and waveform.max() <= 1.)

        waveform_mu = transforms.MuLawEncoding(quantization_channels)(waveform)
        self.assertTrue(waveform_mu.min() >= 0. and waveform_mu.max() <= quantization_channels)

        waveform_exp = transforms.MuLawDecoding(quantization_channels)(waveform_mu)
        self.assertTrue(waveform_exp.min() >= -1. and waveform_exp.max() <= 1.)

    def test_scriptmodule_AmplitudeToDB(self):
        spec = torch.rand((6, 201))
        _test_script_module(transforms.AmplitudeToDB, spec)

    def test_scriptmodule_MelScale(self):
        spec_f = torch.rand((1, 6, 201))
        _test_script_module(transforms.MelScale, spec_f)

    def test_melscale_load_save(self):
        specgram = torch.ones(1, 1000, 100)
        melscale_transform = transforms.MelScale()
        melscale_transform(specgram)

        melscale_transform_copy = transforms.MelScale(n_stft=1000)
        melscale_transform_copy.load_state_dict(melscale_transform.state_dict())

        fb = melscale_transform.fb
        fb_copy = melscale_transform_copy.fb

        self.assertEqual(fb_copy.size(), (1000, 128))
        self.assertTrue(torch.allclose(fb, fb_copy))

    def test_scriptmodule_MelSpectrogram(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(transforms.MelSpectrogram, tensor)

    def test_melspectrogram_load_save(self):
        waveform = self.waveform.float()
        mel_spectrogram_transform = transforms.MelSpectrogram()
        mel_spectrogram_transform(waveform)

        mel_spectrogram_transform_copy = transforms.MelSpectrogram()
        mel_spectrogram_transform_copy.load_state_dict(mel_spectrogram_transform.state_dict())

        window = mel_spectrogram_transform.spectrogram.window
        window_copy = mel_spectrogram_transform_copy.spectrogram.window

        fb = mel_spectrogram_transform.mel_scale.fb
        fb_copy = mel_spectrogram_transform_copy.mel_scale.fb

        self.assertTrue(torch.allclose(window, window_copy))
        # the default for n_fft = 400 and n_mels = 128
        self.assertEqual(fb_copy.size(), (201, 128))
        self.assertTrue(torch.allclose(fb, fb_copy))

    def test_mel2(self):
        top_db = 80.
        s2db = transforms.AmplitudeToDB('power', top_db)

        waveform = self.waveform.clone()  # (1, 16000)
        waveform_scaled = self.scale(waveform)  # (1, 16000)
        mel_transform = transforms.MelSpectrogram()
        # check defaults
        spectrogram_torch = s2db(mel_transform(waveform_scaled))  # (1, 128, 321)
        self.assertTrue(spectrogram_torch.dim() == 3)
        self.assertTrue(spectrogram_torch.ge(spectrogram_torch.max() - top_db).all())
        self.assertEqual(spectrogram_torch.size(1), mel_transform.n_mels)
        # check correctness of filterbank conversion matrix
        self.assertTrue(mel_transform.mel_scale.fb.sum(1).le(1.).all())
        self.assertTrue(mel_transform.mel_scale.fb.sum(1).ge(0.).all())
        # check options
        kwargs = {'window_fn': torch.hamming_window, 'pad': 10, 'win_length': 500,
                  'hop_length': 125, 'n_fft': 800, 'n_mels': 50}
        mel_transform2 = transforms.MelSpectrogram(**kwargs)
        spectrogram2_torch = s2db(mel_transform2(waveform_scaled))  # (1, 50, 513)
        self.assertTrue(spectrogram2_torch.dim() == 3)
        self.assertTrue(spectrogram_torch.ge(spectrogram_torch.max() - top_db).all())
        self.assertEqual(spectrogram2_torch.size(1), mel_transform2.n_mels)
        self.assertTrue(mel_transform2.mel_scale.fb.sum(1).le(1.).all())
        self.assertTrue(mel_transform2.mel_scale.fb.sum(1).ge(0.).all())
        # check on multi-channel audio
        x_stereo, sr_stereo = torchaudio.load(self.test_filepath)  # (2, 278756), 44100
        spectrogram_stereo = s2db(mel_transform(x_stereo))  # (2, 128, 1394)
        self.assertTrue(spectrogram_stereo.dim() == 3)
        self.assertTrue(spectrogram_stereo.size(0) == 2)
        self.assertTrue(spectrogram_torch.ge(spectrogram_torch.max() - top_db).all())
        self.assertEqual(spectrogram_stereo.size(1), mel_transform.n_mels)
        # check filterbank matrix creation
        fb_matrix_transform = transforms.MelScale(
            n_mels=100, sample_rate=16000, f_min=0., f_max=None, n_stft=400)
        self.assertTrue(fb_matrix_transform.fb.sum(1).le(1.).all())
        self.assertTrue(fb_matrix_transform.fb.sum(1).ge(0.).all())
        self.assertEqual(fb_matrix_transform.fb.size(), (400, 100))

    def test_scriptmodule_MFCC(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(transforms.MFCC, tensor)

    def test_mfcc(self):
        audio_orig = self.waveform.clone()
        audio_scaled = self.scale(audio_orig)  # (1, 16000)

        sample_rate = 16000
        n_mfcc = 40
        n_mels = 128
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                    n_mfcc=n_mfcc,
                                                    norm='ortho')
        # check defaults
        torch_mfcc = mfcc_transform(audio_scaled)  # (1, 40, 321)
        self.assertTrue(torch_mfcc.dim() == 3)
        self.assertTrue(torch_mfcc.shape[1] == n_mfcc)
        self.assertTrue(torch_mfcc.shape[2] == 321)
        # check melkwargs are passed through
        melkwargs = {'win_length': 200}
        mfcc_transform2 = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                     n_mfcc=n_mfcc,
                                                     norm='ortho',
                                                     melkwargs=melkwargs)
        torch_mfcc2 = mfcc_transform2(audio_scaled)  # (1, 40, 641)
        self.assertTrue(torch_mfcc2.shape[2] == 641)

        # check norms work correctly
        mfcc_transform_norm_none = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                              n_mfcc=n_mfcc,
                                                              norm=None)
        torch_mfcc_norm_none = mfcc_transform_norm_none(audio_scaled)  # (1, 40, 321)

        norm_check = torch_mfcc.clone()
        norm_check[:, 0, :] *= math.sqrt(n_mels) * 2
        norm_check[:, 1:, :] *= math.sqrt(n_mels / 2) * 2

        self.assertTrue(torch_mfcc_norm_none.allclose(norm_check))

    @unittest.skipIf(
        SKIP_LIBROSA_CONSISTENCY_TEST or not IMPORT_LIBROSA or not IMPORT_SCIPY,
        'Librosa and scipy are not available, or consisency test disabled'
    )
    def test_librosa_consistency(self):
        def _test_librosa_consistency_helper(n_fft, hop_length, power, n_mels, n_mfcc, sample_rate):
            input_path = os.path.join(self.test_dirpath, 'assets', 'sinewave.wav')
            sound, sample_rate = torchaudio.load(input_path)
            sound_librosa = sound.cpu().numpy().squeeze()  # (64000)

            # test core spectrogram
            spect_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power)
            out_librosa, _ = librosa.core.spectrum._spectrogram(y=sound_librosa,
                                                                n_fft=n_fft,
                                                                hop_length=hop_length,
                                                                power=power)

            out_torch = spect_transform(sound).squeeze().cpu()
            self.assertTrue(torch.allclose(out_torch, torch.from_numpy(out_librosa), atol=1e-5))

            # test mel spectrogram
            melspect_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, window_fn=torch.hann_window,
                hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)
            librosa_mel = librosa.feature.melspectrogram(y=sound_librosa, sr=sample_rate,
                                                         n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                                         htk=True, norm=None)
            librosa_mel_tensor = torch.from_numpy(librosa_mel)
            torch_mel = melspect_transform(sound).squeeze().cpu()

            self.assertTrue(torch.allclose(torch_mel.type(librosa_mel_tensor.dtype), librosa_mel_tensor, atol=5e-3))

            # test s2db
            db_transform = torchaudio.transforms.AmplitudeToDB('power', 80.)
            db_torch = db_transform(spect_transform(sound)).squeeze().cpu()
            db_librosa = librosa.core.spectrum.power_to_db(out_librosa)
            self.assertTrue(torch.allclose(db_torch, torch.from_numpy(db_librosa), atol=5e-3))

            db_torch = db_transform(melspect_transform(sound)).squeeze().cpu()
            db_librosa = librosa.core.spectrum.power_to_db(librosa_mel)
            db_librosa_tensor = torch.from_numpy(db_librosa)

            # XXX Test started failing
            # self.assertTrue(torch.allclose(db_torch.type(db_librosa_tensor.dtype), db_librosa_tensor, atol=5e-3))

            # test MFCC
            melkwargs = {'hop_length': hop_length, 'n_fft': n_fft}
            mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                        n_mfcc=n_mfcc,
                                                        norm='ortho',
                                                        melkwargs=melkwargs)

            # librosa.feature.mfcc doesn't pass kwargs properly since some of the
            # kwargs for melspectrogram and mfcc are the same. We just follow the
            # function body in https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#melspectrogram
            # to mirror this function call with correct args:

    #         librosa_mfcc = librosa.feature.mfcc(y=sound_librosa,
    #                                             sr=sample_rate,
    #                                             n_mfcc = n_mfcc,
    #                                             hop_length=hop_length,
    #                                             n_fft=n_fft,
    #                                             htk=True,
    #                                             norm=None,
    #                                             n_mels=n_mels)

            librosa_mfcc = scipy.fftpack.dct(db_librosa, axis=0, type=2, norm='ortho')[:n_mfcc]
            librosa_mfcc_tensor = torch.from_numpy(librosa_mfcc)
            torch_mfcc = mfcc_transform(sound).squeeze().cpu()

            self.assertTrue(torch.allclose(torch_mfcc.type(librosa_mfcc_tensor.dtype), librosa_mfcc_tensor, atol=5e-3))

        kwargs1 = {
            'n_fft': 400,
            'hop_length': 200,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 40,
            'sample_rate': 16000
        }

        kwargs2 = {
            'n_fft': 600,
            'hop_length': 100,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 20,
            'sample_rate': 16000
        }

        kwargs3 = {
            'n_fft': 200,
            'hop_length': 50,
            'power': 2.0,
            'n_mels': 128,
            'n_mfcc': 50,
            'sample_rate': 24000
        }

        kwargs4 = {
            'n_fft': 400,
            'hop_length': 200,
            'power': 3.0,
            'n_mels': 128,
            'n_mfcc': 40,
            'sample_rate': 16000
        }

        _test_librosa_consistency_helper(**kwargs1)
        _test_librosa_consistency_helper(**kwargs2)
        _test_librosa_consistency_helper(**kwargs3)
        _test_librosa_consistency_helper(**kwargs4)

    def test_scriptmodule_Resample(self):
        tensor = torch.rand((2, 1000))
        sample_rate = 100
        sample_rate_2 = 50

        _test_script_module(transforms.Spectrogram, tensor, sample_rate, sample_rate_2)

    def test_resample_size(self):
        input_path = os.path.join(self.test_dirpath, 'assets', 'sinewave.wav')
        waveform, sample_rate = torchaudio.load(input_path)

        upsample_rate = sample_rate * 2
        downsample_rate = sample_rate // 2
        invalid_resample = torchaudio.transforms.Resample(sample_rate, upsample_rate, resampling_method='foo')

        self.assertRaises(ValueError, invalid_resample, waveform)

        upsample_resample = torchaudio.transforms.Resample(
            sample_rate, upsample_rate, resampling_method='sinc_interpolation')
        up_sampled = upsample_resample(waveform)

        # we expect the upsampled signal to have twice as many samples
        self.assertTrue(up_sampled.size(-1) == waveform.size(-1) * 2)

        downsample_resample = torchaudio.transforms.Resample(
            sample_rate, downsample_rate, resampling_method='sinc_interpolation')
        down_sampled = downsample_resample(waveform)

        # we expect the downsampled signal to have half as many samples
        self.assertTrue(down_sampled.size(-1) == waveform.size(-1) // 2)

    def test_compute_deltas(self):
        channel = 13
        n_mfcc = channel * 3
        time = 1021
        win_length = 2 * 7 + 1
        specgram = torch.randn(channel, n_mfcc, time)
        transform = transforms.ComputeDeltas(win_length=win_length)
        computed = transform(specgram)
        self.assertTrue(computed.shape == specgram.shape, (computed.shape, specgram.shape))

    def test_compute_deltas_transform_same_as_functional(self, atol=1e-6, rtol=1e-8):
        channel = 13
        n_mfcc = channel * 3
        time = 1021
        win_length = 2 * 7 + 1
        specgram = torch.randn(channel, n_mfcc, time)

        transform = transforms.ComputeDeltas(win_length=win_length)
        computed_transform = transform(specgram)

        computed_functional = F.compute_deltas(specgram, win_length=win_length)
        torch.testing.assert_allclose(computed_functional, computed_transform, atol=atol, rtol=rtol)

    def test_compute_deltas_twochannel(self):
        specgram = torch.tensor([1., 2., 3., 4.]).repeat(1, 2, 1)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]])
        transform = transforms.ComputeDeltas()
        computed = transform(specgram)
        self.assertTrue(computed.shape == specgram.shape, (computed.shape, specgram.shape))

    def test_batch_compute_deltas(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = transforms.ComputeDeltas()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = transforms.ComputeDeltas()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        self.assertTrue(computed.shape == expected.shape, (computed.shape, expected.shape))
        self.assertTrue(torch.allclose(computed, expected))

    def test_scriptmodule_MuLawEncoding(self):
        tensor = torch.rand((1, 10))
        _test_script_module(transforms.MuLawEncoding, tensor)

    def test_scriptmodule_MuLawDecoding(self):
        tensor = torch.rand((1, 10))
        _test_script_module(transforms.MuLawDecoding, tensor)

    def test_batch_mulaw(self):
        waveform, sample_rate = torchaudio.load(self.test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        waveform_encoded = transforms.MuLawEncoding()(waveform)
        expected = waveform_encoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        waveform_batched = waveform.unsqueeze(0).repeat(3, 1, 1)
        computed = transforms.MuLawEncoding()(waveform_batched)

        # shape = (3, 2, 201, 1394)
        self.assertTrue(computed.shape == expected.shape, (computed.shape, expected.shape))
        self.assertTrue(torch.allclose(computed, expected))

        # Single then transform then batch
        waveform_decoded = transforms.MuLawDecoding()(waveform_encoded)
        expected = waveform_decoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        computed = transforms.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        self.assertTrue(computed.shape == expected.shape, (computed.shape, expected.shape))
        self.assertTrue(torch.allclose(computed, expected))

    def test_batch_spectrogram(self):
        waveform, sample_rate = torchaudio.load(self.test_filepath)

        # Single then transform then batch
        expected = transforms.Spectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = transforms.Spectrogram()(waveform.repeat(3, 1, 1))

        self.assertTrue(computed.shape == expected.shape, (computed.shape, expected.shape))
        self.assertTrue(torch.allclose(computed, expected))

    def test_scriptmodule_TimeStretch(self):
        n_freq = 400
        hop_length = 512
        fixed_rate = 1.3
        tensor = torch.rand((10, 2, n_freq, 10, 2))
        _test_script_module(transforms.TimeStretch, tensor, n_freq=n_freq, hop_length=hop_length, fixed_rate=fixed_rate)

    def test_scriptmodule_FrequencyMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        _test_script_module(transforms.FrequencyMasking, tensor, freq_mask_param=60, iid_masks=False)

    def test_scriptmodule_TimeMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        _test_script_module(transforms.TimeMasking, tensor, time_mask_param=30, iid_masks=False)


if __name__ == '__main__':
    unittest.main()
