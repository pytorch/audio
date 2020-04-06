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
    torch.testing.assert_allclose(jit_out, py_out)


def _test_lfilter(waveform):
    """
    Design an IIR lowpass filter using scipy.signal filter design
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign

    Example
        >>> from scipy.signal import iirdesign
        >>> b, a = iirdesign(0.2, 0.3, 1, 60)
    """
    b_coeffs = torch.tensor(
        [
            0.00299893,
            -0.0051152,
            0.00841964,
            -0.00747802,
            0.00841964,
            -0.0051152,
            0.00299893,
        ],
        device=waveform.device,
    )
    a_coeffs = torch.tensor(
        [
            1.0,
            -4.8155751,
            10.2217618,
            -12.14481273,
            8.49018171,
            -3.3066882,
            0.56088705,
        ],
        device=waveform.device,
    )
    _test_torchscript_functional(F.lfilter, waveform, a_coeffs, b_coeffs)


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

    def test_lfilter(self):
        filepath = os.path.join(common_utils.TEST_DIR_PATH, 'assets', 'whitenoise.wav')
        waveform, _ = torchaudio.load(filepath, normalization=True)
        _test_lfilter(waveform)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_lfilter_cuda(self):
        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)
        _test_lfilter(waveform.cuda(device=torch.device("cuda:0")))

    def test_lowpass(self):
        cutoff_freq = 3000

        filepath = os.path.join(common_utils.TEST_DIR_PATH, 'assets', 'whitenoise.wav')
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.lowpass_biquad, waveform, sample_rate, cutoff_freq)

    def test_highpass(self):
        cutoff_freq = 2000

        filepath = os.path.join(common_utils.TEST_DIR_PATH, 'assets', 'whitenoise.wav')
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.highpass_biquad, waveform, sample_rate, cutoff_freq)

    def test_allpass(self):
        central_freq = 1000
        q = 0.707

        filepath = os.path.join(common_utils.TEST_DIR_PATH, 'assets', 'whitenoise.wav')
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.allpass_biquad, waveform, sample_rate, central_freq, q)

    def test_bandpass_with_csg(self):
        central_freq = 1000
        q = 0.707
        const_skirt_gain = True

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(
            F.bandpass_biquad, waveform, sample_rate, central_freq, q, const_skirt_gain)

    def test_bandpass_withou_csg(self):
        central_freq = 1000
        q = 0.707
        const_skirt_gain = False

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(
            F.bandpass_biquad, waveform, sample_rate, central_freq, q, const_skirt_gain)

    def test_bandreject(self):
        central_freq = 1000
        q = 0.707

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(
            F.bandreject_biquad, waveform, sample_rate, central_freq, q)

    def test_band_with_noise(self):
        central_freq = 1000
        q = 0.707
        noise = True

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.band_biquad, waveform, sample_rate, central_freq, q, noise)

    def test_band_without_noise(self):
        central_freq = 1000
        q = 0.707
        noise = False

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.band_biquad, waveform, sample_rate, central_freq, q, noise)

    def test_treble(self):
        gain = 40
        central_freq = 1000
        q = 0.707

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.treble_biquad, waveform, sample_rate, gain, central_freq, q)

    def test_deemph(self):
        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.deemph_biquad, waveform, sample_rate)

    def test_riaa(self):
        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.riaa_biquad, waveform, sample_rate)

    def test_equalizer(self):
        center_freq = 300
        gain = 1
        q = 0.707

        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(
            F.equalizer_biquad, waveform, sample_rate, center_freq, gain, q)

    def test_perf_biquad_filtering(self):
        a = torch.tensor([0.7, 0.2, 0.6])
        b = torch.tensor([0.4, 0.2, 0.9])
        filepath = os.path.join(common_utils.TEST_DIR_PATH, "assets", "whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)
        _test_torchscript_functional(F.lfilter, waveform, a, b)


RUN_CUDA = torch.cuda.is_available()
print("Run test with cuda:", RUN_CUDA)


def _test_script_module(f, tensor, *args, **kwargs):

    py_method = f(*args, **kwargs)
    jit_method = torch.jit.script(py_method)

    py_out = py_method(tensor)
    jit_out = jit_method(tensor)

    torch.testing.assert_allclose(jit_out, py_out)

    if RUN_CUDA:

        tensor = tensor.to("cuda")

        py_method = py_method.cuda()
        jit_method = torch.jit.script(py_method)

        py_out = py_method(tensor)
        jit_out = jit_method(tensor)

        torch.testing.assert_allclose(jit_out, py_out)


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


if __name__ == '__main__':
    unittest.main()
