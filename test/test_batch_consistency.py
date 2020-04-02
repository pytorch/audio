"""Test numerical consistency among single input and batched input."""
import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F

import common_utils


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
