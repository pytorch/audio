import math
import unittest

import torch
import torchaudio
import torchaudio.transforms as transforms
import torchaudio.functional as F

from torchaudio_unittest import common_utils


class Tester(common_utils.TorchaudioTestCase):
    backend = 'default'

    # create a sinewave signal for testing
    sample_rate = 16000
    freq = 440
    volume = .3
    waveform = (torch.cos(2 * math.pi * torch.arange(0, 4 * sample_rate).float() * freq / sample_rate))
    waveform.unsqueeze_(0)  # (1, 64000)
    waveform = (waveform * volume * 2**31).long()

    def scale(self, waveform, factor=2.0**31):
        # scales a waveform by a factor
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.get_default_dtype())
        return waveform / factor

    def test_mu_law_companding(self):

        quantization_channels = 256

        waveform = self.waveform.clone()
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.get_default_dtype())
        waveform /= torch.abs(waveform).max()

        self.assertTrue(waveform.min() >= -1. and waveform.max() <= 1.)

        waveform_mu = transforms.MuLawEncoding(quantization_channels)(waveform)
        self.assertTrue(waveform_mu.min() >= 0. and waveform_mu.max() <= quantization_channels)

        waveform_exp = transforms.MuLawDecoding(quantization_channels)(waveform_mu)
        self.assertTrue(waveform_exp.min() >= -1. and waveform_exp.max() <= 1.)

    def test_AmplitudeToDB(self):
        filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform = common_utils.load_wav(filepath)[0]

        mag_to_db_transform = transforms.AmplitudeToDB('magnitude', 80.)
        power_to_db_transform = transforms.AmplitudeToDB('power', 80.)

        mag_to_db_torch = mag_to_db_transform(torch.abs(waveform))
        power_to_db_torch = power_to_db_transform(torch.pow(waveform, 2))

        self.assertEqual(mag_to_db_torch, power_to_db_torch)

    def test_melscale_load_save(self):
        specgram = torch.ones(1, 1000, 100)
        melscale_transform = transforms.MelScale()
        melscale_transform(specgram)

        melscale_transform_copy = transforms.MelScale(n_stft=1000)
        melscale_transform_copy.load_state_dict(melscale_transform.state_dict())

        fb = melscale_transform.fb
        fb_copy = melscale_transform_copy.fb

        self.assertEqual(fb_copy.size(), (1000, 128))
        self.assertEqual(fb, fb_copy)

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

        self.assertEqual(window, window_copy)
        # the default for n_fft = 400 and n_mels = 128
        self.assertEqual(fb_copy.size(), (201, 128))
        self.assertEqual(fb, fb_copy)

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
        filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        x_stereo = common_utils.load_wav(filepath)[0]  # (2, 278756), 44100
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

    def test_resample_size(self):
        input_path = common_utils.get_asset_path('sinewave.wav')
        waveform, sample_rate = common_utils.load_wav(input_path)

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
        self.assertEqual(computed_functional, computed_transform, atol=atol, rtol=rtol)

    def test_compute_deltas_twochannel(self):
        specgram = torch.tensor([1., 2., 3., 4.]).repeat(1, 2, 1)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]])
        transform = transforms.ComputeDeltas(win_length=3)
        computed = transform(specgram)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        self.assertEqual(computed, expected, atol=1e-6, rtol=1e-8)


class SmokeTest(common_utils.TorchaudioTestCase):

    def test_spectrogram(self):
        specgram = transforms.Spectrogram(center=False, pad_mode="reflect", onesided=False)
        self.assertEqual(specgram.center, False)
        self.assertEqual(specgram.pad_mode, "reflect")
        self.assertEqual(specgram.onesided, False)

    def test_melspectrogram(self):
        melspecgram = transforms.MelSpectrogram(center=True, pad_mode="reflect", onesided=False)
        specgram = melspecgram.spectrogram
        self.assertEqual(specgram.center, True)
        self.assertEqual(specgram.pad_mode, "reflect")
        self.assertEqual(specgram.onesided, False)
