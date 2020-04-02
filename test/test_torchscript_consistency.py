"""Test suites for jit-ability and its numerical compatibility"""
import os
import unittest

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms

import common_utils


def _test_torchscript_functional_shape(py_method, *args, **kwargs):
    jit_method = torch.jit.script(py_method)

    jit_out = jit_method(*args, **kwargs)
    py_out = py_method(*args, **kwargs)

    assert jit_out.shape == py_out.shape
    return jit_out, py_out


def _test_torchscript_functional(py_method, *args, **kwargs):
    jit_out, py_out = _test_torchscript_functional_shape(py_method, *args, **kwargs)
    assert torch.allclose(jit_out, py_out)


class TestFunctional(unittest.TestCase):
    """Test functions in `functional` module."""
    def test_spectrogram(self):
        tensor = torch.rand((1, 1000))
        n_fft = 400
        ws = 400
        hop = 200
        pad = 0
        window = torch.hann_window(ws)
        power = 2
        normalize = False

        _test_torchscript_functional(
            F.spectrogram, tensor, pad, window, n_fft, hop, ws, power, normalize
        )

    def test_griffinlim(self):
        tensor = torch.rand((1, 201, 6))
        n_fft = 400
        ws = 400
        hop = 200
        window = torch.hann_window(ws)
        power = 2
        normalize = False
        momentum = 0.99
        n_iter = 32
        length = 1000

        _test_torchscript_functional(
            F.griffinlim, tensor, window, n_fft, hop, ws, power, normalize, n_iter, momentum, length, 0
        )

    def test_compute_deltas(self):
        channel = 13
        n_mfcc = channel * 3
        time = 1021
        win_length = 2 * 7 + 1
        specgram = torch.randn(channel, n_mfcc, time)

        _test_torchscript_functional(F.compute_deltas, specgram, win_length=win_length)

    def test_detect_pitch_frequency(self):
        filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.mp3')
        waveform, sample_rate = torchaudio.load(filepath)
        _test_torchscript_functional(F.detect_pitch_frequency, waveform, sample_rate)

    def test_create_fb_matrix(self):
        n_stft = 100
        f_min = 0.0
        f_max = 20.0
        n_mels = 10
        sample_rate = 16000

        _test_torchscript_functional(F.create_fb_matrix, n_stft, f_min, f_max, n_mels, sample_rate)

    def test_amplitude_to_DB(self):
        spec = torch.rand((6, 201))
        multiplier = 10.0
        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0

        _test_torchscript_functional(F.amplitude_to_DB, spec, multiplier, amin, db_multiplier, top_db)

    def test_DB_to_amplitude(self):
        x = torch.rand((1, 100))
        ref = 1.
        power = 1.

        _test_torchscript_functional(F.DB_to_amplitude, x, ref, power)

    def test_create_dct(self):
        n_mfcc = 40
        n_mels = 128
        norm = "ortho"

        _test_torchscript_functional(F.create_dct, n_mfcc, n_mels, norm)

    def test_mu_law_encoding(self):
        tensor = torch.rand((1, 10))
        qc = 256

        _test_torchscript_functional(F.mu_law_encoding, tensor, qc)

    def test_mu_law_decoding(self):
        tensor = torch.rand((1, 10))
        qc = 256

        _test_torchscript_functional(F.mu_law_decoding, tensor, qc)

    def test_complex_norm(self):
        complex_tensor = torch.randn(1, 2, 1025, 400, 2)
        power = 2

        _test_torchscript_functional(F.complex_norm, complex_tensor, power)

    def test_mask_along_axis(self):
        specgram = torch.randn(2, 1025, 400)
        mask_param = 100
        mask_value = 30.
        axis = 2

        _test_torchscript_functional(F.mask_along_axis, specgram, mask_param, mask_value, axis)

    def test_mask_along_axis_iid(self):
        specgrams = torch.randn(4, 2, 1025, 400)
        mask_param = 100
        mask_value = 30.
        axis = 2

        _test_torchscript_functional(F.mask_along_axis_iid, specgrams, mask_param, mask_value, axis)

    def test_gain(self):
        tensor = torch.rand((1, 1000))
        gainDB = 2.0

        _test_torchscript_functional(F.gain, tensor, gainDB)

    def test_dither(self):
        tensor = torch.rand((2, 1000))

        _test_torchscript_functional_shape(F.dither, tensor)
        _test_torchscript_functional_shape(F.dither, tensor, "RPDF")
        _test_torchscript_functional_shape(F.dither, tensor, "GPDF")


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


class TestTransforms(unittest.TestCase):
    def test_Spectrogram(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(torchaudio.transforms.Spectrogram, tensor)

    def test_GriffinLim(self):
        tensor = torch.rand((1, 201, 6))
        _test_script_module(torchaudio.transforms.GriffinLim, tensor, length=1000, rand_init=False)

    def test_AmplitudeToDB(self):
        spec = torch.rand((6, 201))
        _test_script_module(torchaudio.transforms.AmplitudeToDB, spec)

    def test_MelScale(self):
        spec_f = torch.rand((1, 6, 201))
        _test_script_module(torchaudio.transforms.MelScale, spec_f)

    def test_MelSpectrogram(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(torchaudio.transforms.MelSpectrogram, tensor)

    def test_MFCC(self):
        tensor = torch.rand((1, 1000))
        _test_script_module(torchaudio.transforms.MFCC, tensor)

    def test_Resample(self):
        tensor = torch.rand((2, 1000))
        sample_rate = 100.
        sample_rate_2 = 50.

        _test_script_module(torchaudio.transforms.Resample, tensor, sample_rate, sample_rate_2)

    def test_ComplexNorm(self):
        tensor = torch.rand((1, 2, 201, 2))
        _test_script_module(torchaudio.transforms.ComplexNorm, tensor)

    def test_MuLawEncoding(self):
        tensor = torch.rand((1, 10))
        _test_script_module(torchaudio.transforms.MuLawEncoding, tensor)

    def test_MuLawDecoding(self):
        tensor = torch.rand((1, 10))
        _test_script_module(torchaudio.transforms.MuLawDecoding, tensor)

    def test_TimeStretch(self):
        n_freq = 400
        hop_length = 512
        fixed_rate = 1.3
        tensor = torch.rand((10, 2, n_freq, 10, 2))
        _test_script_module(
            torchaudio.transforms.TimeStretch,
            tensor, n_freq=n_freq, hop_length=hop_length, fixed_rate=fixed_rate)

    def test_Fade(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)
        fade_in_len = 3000
        fade_out_len = 3000

        _test_script_module(torchaudio.transforms.Fade, waveform, fade_in_len, fade_out_len)

    def test_FrequencyMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        _test_script_module(
            torchaudio.transforms.FrequencyMasking, tensor, freq_mask_param=60, iid_masks=False)

    def test_TimeMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        _test_script_module(
            torchaudio.transforms.TimeMasking, tensor, time_mask_param=30, iid_masks=False)

    def test_Vol(self):
        test_filepath = os.path.join(
            common_utils.TEST_DIR_PATH, 'assets', 'steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)
        _test_script_module(torchaudio.transforms.Vol, waveform, 1.1)
