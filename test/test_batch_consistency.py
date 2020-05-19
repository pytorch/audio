"""Test numerical consistency among single input and batched input."""
import unittest

import torch
from torch.testing._internal.common_utils import TestCase
import torchaudio
import torchaudio.functional as F

import common_utils


class TestFunctional(TestCase):
    """Test functions defined in `functional` module"""
    def assert_batch_consistency(
            self, functional, tensor, *args, batch_size=1, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        # run then batch the result
        torch.random.manual_seed(seed)
        expected = functional(tensor.clone(), *args, **kwargs)
        expected = expected.repeat([batch_size] + [1] * expected.dim())

        # batch the input and run
        torch.random.manual_seed(seed)
        pattern = [batch_size] + [1] * tensor.dim()
        computed = functional(tensor.repeat(pattern), *args, **kwargs)

        self.assertEqual(computed, expected, rtol=rtol, atol=atol)

    def assert_batch_consistencies(
            self, functional, tensor, *args, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        self.assert_batch_consistency(
            functional, tensor, *args, batch_size=1, atol=atol, rtol=rtol, seed=seed, **kwargs)
        self.assert_batch_consistency(
            functional, tensor, *args, batch_size=3, atol=atol, rtol=rtol, seed=seed, **kwargs)

    def test_griffinlim(self):
        n_fft = 400
        ws = 400
        hop = 200
        window = torch.hann_window(ws)
        power = 2
        normalize = False
        momentum = 0.99
        n_iter = 32
        length = 1000
        tensor = torch.rand((1, 201, 6))
        self.assert_batch_consistencies(
            F.griffinlim, tensor, window, n_fft, hop, ws, power, normalize, n_iter, momentum, length, 0, atol=5e-5
        )

    def test_detect_pitch_frequency(self):
        filenames = [
            'steam-train-whistle-daniel_simon.wav',  # 2ch 44100Hz
            # Files from https://www.mediacollege.com/audio/tone/download/
            '100Hz_44100Hz_16bit_05sec.wav',  # 1ch
            '440Hz_44100Hz_16bit_05sec.wav',  # 1ch
        ]
        for filename in filenames:
            filepath = common_utils.get_asset_path(filename)
            waveform, sample_rate = torchaudio.load(filepath)
            self.assert_batch_consistencies(F.detect_pitch_frequency, waveform, sample_rate)

    def test_istft(self):
        stft = torch.tensor([
            [[4., 0.], [4., 0.], [4., 0.], [4., 0.], [4., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ])
        self.assert_batch_consistencies(F.istft, stft, n_fft=4, length=4)

    def test_contrast(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(F.contrast, waveform, enhancement_amount=80.)

    def test_dcshift(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(F.dcshift, waveform, shift=0.5, limiter_gain=0.05)

    def test_overdrive(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(F.overdrive, waveform, gain=45, colour=30)

    def test_phaser(self):
        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath)
        self.assert_batch_consistencies(F.phaser, waveform, sample_rate)

    def test_flanger(self):
        waveform = torch.rand(2, 100) - 0.5
        sample_rate = 44100
        self.assert_batch_consistencies(F.flanger, waveform, sample_rate)

    def test_sliding_window_cmn(self):
        waveform = torch.randn(2, 1024) - 0.5
        self.assert_batch_consistencies(F.sliding_window_cmn, waveform, center=True, norm_vars=True)
        self.assert_batch_consistencies(F.sliding_window_cmn, waveform, center=True, norm_vars=False)
        self.assert_batch_consistencies(F.sliding_window_cmn, waveform, center=False, norm_vars=True)
        self.assert_batch_consistencies(F.sliding_window_cmn, waveform, center=False, norm_vars=False)

    def test_vad(self):
        filepath = common_utils.get_asset_path("vad-go-mono-32000.wav")
        waveform, sample_rate = torchaudio.load(filepath)
        self.assert_batch_consistencies(F.vad, waveform, sample_rate=sample_rate)


class TestTransforms(TestCase):
    """Test suite for classes defined in `transforms` module"""
    def test_batch_AmplitudeToDB(self):
        spec = torch.rand((6, 201))

        # Single then transform then batch
        expected = torchaudio.transforms.AmplitudeToDB()(spec).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.AmplitudeToDB()(spec.repeat(3, 1, 1))

        self.assertEqual(computed, expected)

    def test_batch_Resample(self):
        waveform = torch.randn(2, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.Resample()(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Resample()(waveform.repeat(3, 1, 1))

        self.assertEqual(computed, expected)

    def test_batch_MelScale(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.MelScale()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelScale()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_InverseMelScale(self):
        n_mels = 32
        n_stft = 5
        mel_spec = torch.randn(2, n_mels, 32) ** 2

        # Single then transform then batch
        expected = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec.repeat(3, 1, 1, 1))

        # shape = (3, 2, n_mels, 32)

        # Because InverseMelScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assertEqual(computed, expected, atol=1.0, rtol=1e-5)

    def test_batch_compute_deltas(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.ComputeDeltas()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.ComputeDeltas()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_mulaw(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        waveform_encoded = torchaudio.transforms.MuLawEncoding()(waveform)
        expected = waveform_encoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        waveform_batched = waveform.unsqueeze(0).repeat(3, 1, 1)
        computed = torchaudio.transforms.MuLawEncoding()(waveform_batched)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

        # Single then transform then batch
        waveform_decoded = torchaudio.transforms.MuLawDecoding()(waveform_encoded)
        expected = waveform_decoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_spectrogram(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Spectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Spectrogram()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_melspectrogram(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.MelSpectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelSpectrogram()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_mfcc(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)

        # Single then transform then batch
        expected = torchaudio.transforms.MFCC()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MFCC()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected, atol=1e-4, rtol=1e-5)

    def test_batch_TimeStretch(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        kwargs = {
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'window': torch.hann_window(2048),
            'center': True,
            'pad_mode': 'reflect',
            'normalized': True,
            'onesided': True,
        }
        rate = 2

        complex_specgrams = torch.stft(waveform, **kwargs)

        # Single then transform then batch
        expected = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams).repeat(3, 1, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams.repeat(3, 1, 1, 1, 1))

        self.assertEqual(computed, expected, atol=1e-5, rtol=1e-5)

    def test_batch_Fade(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100
        fade_in_len = 3000
        fade_out_len = 3000

        # Single then transform then batch
        expected = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_Vol(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Vol(gain=1.1)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Vol(gain=1.1)(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)


if __name__ == '__main__':
    unittest.main()
