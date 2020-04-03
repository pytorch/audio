"""Test numerical consistency among single input and batched input."""
import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F

import common_utils
from common_utils import AudioBackendScope, BACKENDS


def _test_batch_shape(functional, tensor, *args, **kwargs):

    kwargs_compare = {}
    if 'atol' in kwargs:
        atol = kwargs['atol']
        del kwargs['atol']
        kwargs_compare['atol'] = atol

    if 'rtol' in kwargs:
        rtol = kwargs['rtol']
        del kwargs['rtol']
        kwargs_compare['rtol'] = rtol

    # Single then transform then batch

    torch.random.manual_seed(42)
    expected = functional(tensor.clone(), *args, **kwargs)
    expected = expected.unsqueeze(0).unsqueeze(0)

    # 1-Batch then transform
    tensors = tensor.unsqueeze(0).unsqueeze(0)

    torch.random.manual_seed(42)
    computed = functional(tensors.clone(), *args, **kwargs)

    assert expected.shape == computed.shape, (expected.shape, computed.shape)
    assert torch.allclose(expected, computed, **kwargs_compare)

    return tensors, expected


def _test_batch(functional, tensor, *args, **kwargs):
    tensors, expected = _test_batch_shape(functional, tensor, *args, **kwargs)

    kwargs_compare = {}
    if 'atol' in kwargs:
        atol = kwargs['atol']
        del kwargs['atol']
        kwargs_compare['atol'] = atol

    if 'rtol' in kwargs:
        rtol = kwargs['rtol']
        del kwargs['rtol']
        kwargs_compare['rtol'] = rtol

    # 3-Batch then transform

    ind = [3] + [1] * (int(tensors.dim()) - 1)
    tensors = tensor.repeat(*ind)

    ind = [3] + [1] * (int(expected.dim()) - 1)
    expected = expected.repeat(*ind)

    torch.random.manual_seed(42)
    computed = functional(tensors.clone(), *args, **kwargs)

    assert expected.shape == computed.shape, (expected.shape, computed.shape)
    assert torch.allclose(expected, computed, **kwargs_compare)


class TestFunctional(unittest.TestCase):
    """Test functions defined in `functional` module"""
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
        _test_batch(
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
            filepath = os.path.join(common_utils.TEST_DIR_PATH, 'assets', filename)
            waveform, sample_rate = torchaudio.load(filepath)
            _test_batch(F.detect_pitch_frequency, waveform, sample_rate)

    def test_istft(self):
        stft = torch.tensor([
            [[4., 0.], [4., 0.], [4., 0.], [4., 0.], [4., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ])
        _test_batch(F.istft, stft, n_fft=4, length=4)


class TestTransforms(unittest.TestCase):
    """Test suite for classes defined in `transforms` module"""
    def test_batch_AmplitudeToDB(self):
        spec = torch.rand((6, 201))

        # Single then transform then batch
        expected = torchaudio.transforms.AmplitudeToDB()(spec).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.AmplitudeToDB()(spec.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_Resample(self):
        waveform = torch.randn(2, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.Resample()(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Resample()(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_MelScale(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.MelScale()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelScale()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_InverseMelScale(self):
        n_mels = 32
        n_stft = 5
        mel_spec = torch.randn(2, n_mels, 32) ** 2

        # Single then transform then batch
        expected = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec.repeat(3, 1, 1, 1))

        # shape = (3, 2, n_mels, 32)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)

        # Because InverseMelScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        assert torch.allclose(computed, expected, atol=1.0)

    def test_batch_compute_deltas(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.ComputeDeltas()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.ComputeDeltas()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_mulaw(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        waveform_encoded = torchaudio.transforms.MuLawEncoding()(waveform)
        expected = waveform_encoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        waveform_batched = waveform.unsqueeze(0).repeat(3, 1, 1)
        computed = torchaudio.transforms.MuLawEncoding()(waveform_batched)

        # shape = (3, 2, 201, 1394)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

        # Single then transform then batch
        waveform_decoded = torchaudio.transforms.MuLawDecoding()(waveform_encoded)
        expected = waveform_decoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_spectrogram(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Spectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Spectrogram()(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_melspectrogram(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.MelSpectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelSpectrogram()(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_batch_mfcc(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.mp3')
        waveform, _ = torchaudio.load(test_filepath)

        # Single then transform then batch
        expected = torchaudio.transforms.MFCC()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MFCC()(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected, atol=1e-5)

    def test_batch_TimeStretch(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
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

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected, atol=1e-5)

    def test_batch_Fade(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100
        fade_in_len = 3000
        fade_out_len = 3000

        # Single then transform then batch
        expected = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)

    def test_batch_Vol(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Vol(gain=1.1)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Vol(gain=1.1)(waveform.repeat(3, 1, 1))

        assert computed.shape == expected.shape, (computed.shape, expected.shape)
        assert torch.allclose(computed, expected)


if __name__ == '__main__':
    unittest.main()
