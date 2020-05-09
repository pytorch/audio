"""Test suites for jit-ability and its numerical compatibility"""
import unittest
import pytest

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import common_utils


def _assert_functional_consistency(func, tensor, shape_only=False):
    ts_func = torch.jit.script(func)
    output = func(tensor)
    ts_output = ts_func(tensor)

    if shape_only:
        assert ts_output.shape == output.shape, (ts_output.shape, output.shape)
    else:
        torch.testing.assert_allclose(ts_output, output)


def _assert_transforms_consistency(transform, tensor):
    ts_transform = torch.jit.script(transform)
    output = transform(tensor)
    ts_output = ts_transform(tensor)
    torch.testing.assert_allclose(ts_output, output)


class Functional(common_utils.TestBaseMixin):
    """Implements test for `functinoal` modul that are performed for different devices"""
    def _assert_consistency(self, func, tensor, shape_only=False):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return _assert_functional_consistency(func, tensor, shape_only=shape_only)

    def test_spectrogram(self):
        def func(tensor):
            n_fft = 400
            ws = 400
            hop = 200
            pad = 0
            window = torch.hann_window(ws, device=tensor.device, dtype=tensor.dtype)
            power = 2.
            normalize = False
            return F.spectrogram(tensor, pad, window, n_fft, hop, ws, power, normalize)

        tensor = torch.rand((1, 1000))
        self._assert_consistency(func, tensor)

    def test_griffinlim(self):
        def func(tensor):
            n_fft = 400
            ws = 400
            hop = 200
            window = torch.hann_window(ws, device=tensor.device, dtype=tensor.dtype)
            power = 2.
            normalize = False
            momentum = 0.99
            n_iter = 32
            length = 1000
            rand_int = False
            return F.griffinlim(tensor, window, n_fft, hop, ws, power, normalize, n_iter, momentum, length, rand_int)

        tensor = torch.rand((1, 201, 6))
        self._assert_consistency(func, tensor)

    def test_compute_deltas(self):
        def func(tensor):
            win_length = 2 * 7 + 1
            return F.compute_deltas(tensor, win_length=win_length)

        channel = 13
        n_mfcc = channel * 3
        time = 1021
        tensor = torch.randn(channel, n_mfcc, time)
        self._assert_consistency(func, tensor)

    def test_detect_pitch_frequency(self):
        filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(filepath)

        def func(tensor):
            sample_rate = 44100
            return F.detect_pitch_frequency(tensor, sample_rate)

        self._assert_consistency(func, waveform)

    def test_create_fb_matrix(self):
        if self.device != torch.device('cpu'):
            raise unittest.SkipTest('No need to perform test on device other than CPU')

        def func(_):
            n_stft = 100
            f_min = 0.0
            f_max = 20.0
            n_mels = 10
            sample_rate = 16000
            return F.create_fb_matrix(n_stft, f_min, f_max, n_mels, sample_rate)

        dummy = torch.zeros(1, 1)
        self._assert_consistency(func, dummy)

    def test_amplitude_to_DB(self):
        def func(tensor):
            multiplier = 10.0
            amin = 1e-10
            db_multiplier = 0.0
            top_db = 80.0
            return F.amplitude_to_DB(tensor, multiplier, amin, db_multiplier, top_db)

        tensor = torch.rand((6, 201))
        self._assert_consistency(func, tensor)

    def test_DB_to_amplitude(self):
        def func(tensor):
            ref = 1.
            power = 1.
            return F.DB_to_amplitude(tensor, ref, power)

        tensor = torch.rand((1, 100))
        self._assert_consistency(func, tensor)

    def test_create_dct(self):
        if self.device != torch.device('cpu'):
            raise unittest.SkipTest('No need to perform test on device other than CPU')

        def func(_):
            n_mfcc = 40
            n_mels = 128
            norm = "ortho"
            return F.create_dct(n_mfcc, n_mels, norm)

        dummy = torch.zeros(1, 1)
        self._assert_consistency(func, dummy)

    def test_mu_law_encoding(self):
        def func(tensor):
            qc = 256
            return F.mu_law_encoding(tensor, qc)

        tensor = torch.rand((1, 10))
        self._assert_consistency(func, tensor)

    def test_mu_law_decoding(self):
        def func(tensor):
            qc = 256
            return F.mu_law_decoding(tensor, qc)

        tensor = torch.rand((1, 10))
        self._assert_consistency(func, tensor)

    def test_complex_norm(self):
        def func(tensor):
            power = 2.
            return F.complex_norm(tensor, power)

        tensor = torch.randn(1, 2, 1025, 400, 2)
        self._assert_consistency(func, tensor)

    def test_mask_along_axis(self):
        def func(tensor):
            mask_param = 100
            mask_value = 30.
            axis = 2
            return F.mask_along_axis(tensor, mask_param, mask_value, axis)

        tensor = torch.randn(2, 1025, 400)
        self._assert_consistency(func, tensor)

    def test_mask_along_axis_iid(self):
        def func(tensor):
            mask_param = 100
            mask_value = 30.
            axis = 2
            return F.mask_along_axis_iid(tensor, mask_param, mask_value, axis)

        tensor = torch.randn(4, 2, 1025, 400)
        self._assert_consistency(func, tensor)

    def test_gain(self):
        def func(tensor):
            gainDB = 2.0
            return F.gain(tensor, gainDB)

        tensor = torch.rand((1, 1000))
        self._assert_consistency(func, tensor)

    def test_dither_TPDF(self):
        def func(tensor):
            return F.dither(tensor, 'TPDF')

        tensor = torch.rand((2, 1000))
        self._assert_consistency(func, tensor, shape_only=True)

    def test_dither_RPDF(self):
        def func(tensor):
            return F.dither(tensor, 'RPDF')

        tensor = torch.rand((2, 1000))
        self._assert_consistency(func, tensor, shape_only=True)

    def test_dither_GPDF(self):
        def func(tensor):
            return F.dither(tensor, 'GPDF')

        tensor = torch.rand((2, 1000))
        self._assert_consistency(func, tensor, shape_only=True)

    def test_lfilter(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path('whitenoise.wav')
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            # Design an IIR lowpass filter using scipy.signal filter design
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign
            #
            # Example
            #     >>> from scipy.signal import iirdesign
            #     >>> b, a = iirdesign(0.2, 0.3, 1, 60)
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
                device=tensor.device,
                dtype=tensor.dtype,
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
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return F.lfilter(tensor, a_coeffs, b_coeffs)

        self._assert_consistency(func, waveform)

    def test_lowpass(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path('whitenoise.wav')
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 3000.
            return F.lowpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, waveform)

    def test_highpass(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path('whitenoise.wav')
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 2000.
            return F.highpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, waveform)

    def test_allpass(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path('whitenoise.wav')
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            return F.allpass_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_bandpass_with_csg(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, waveform)

    def test_bandpass_without_csg(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, waveform)

    def test_bandreject(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            return F.bandreject_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_band_with_noise(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            noise = True
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, waveform)

    def test_band_without_noise(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            noise = False
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, waveform)

    def test_treble(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            gain = 40.
            central_freq = 1000.
            q = 0.707
            return F.treble_biquad(tensor, sample_rate, gain, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_deemph(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            return F.deemph_biquad(tensor, sample_rate)

        self._assert_consistency(func, waveform)

    def test_riaa(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            return F.riaa_biquad(tensor, sample_rate)

        self._assert_consistency(func, waveform)

    def test_equalizer(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            sample_rate = 44100
            center_freq = 300.
            gain = 1.
            q = 0.707
            return F.equalizer_biquad(tensor, sample_rate, center_freq, gain, q)

        self._assert_consistency(func, waveform)

    def test_perf_biquad_filtering(self):
        if self.dtype == torch.float64:
            pytest.xfail("This test is known to fail for float64")

        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            a = torch.tensor([0.7, 0.2, 0.6], device=tensor.device, dtype=tensor.dtype)
            b = torch.tensor([0.4, 0.2, 0.9], device=tensor.device, dtype=tensor.dtype)
            return F.lfilter(tensor, a, b)

        self._assert_consistency(func, waveform)

    def test_sliding_window_cmn(self):
        def func(tensor):
            cmn_window = 600
            min_cmn_window = 100
            center = False
            norm_vars = False
            a = torch.tensor(
                [
                    [
                        -1.915875792503357,
                        1.147700309753418
                    ],
                    [
                        1.8242558240890503,
                        1.3869990110397339
                    ]
                ],
                device=tensor.device,
                dtype=tensor.dtype
            )
            return F.sliding_window_cmn(a, cmn_window, min_cmn_window, center, norm_vars)
        b = torch.tensor(
            [
                [
                    -1.8701,
                    -0.1196
                ],
                [
                    1.8701,
                    0.1196
                ]
            ]
        )
        self._assert_consistency(func, b)

    def test_contrast(self):
        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            enhancement_amount = 80.
            return F.contrast(tensor, enhancement_amount)

        self._assert_consistency(func, waveform)

    def test_dcshift(self):
        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            shift = 0.5
            limiter_gain = 0.05
            return F.dcshift(tensor, shift, limiter_gain)

        self._assert_consistency(func, waveform)

    def test_overdrive(self):
        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, _ = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            gain = 30.
            colour = 50.
            return F.overdrive(tensor, gain, colour)

        self._assert_consistency(func, waveform)

    def test_phaser(self):
        filepath = common_utils.get_asset_path("whitenoise.wav")
        waveform, sample_rate = torchaudio.load(filepath, normalization=True)

        def func(tensor):
            gain_in = 0.5
            gain_out = 0.8
            delay_ms = 2.0
            decay = 0.4
            speed = 0.5
            sample_rate = 44100
            return F.phaser(tensor, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=True)

        self._assert_consistency(func, waveform)


class Transforms(common_utils.TestBaseMixin):
    """Implements test for Transforms that are performed for different devices"""
    def _assert_consistency(self, transform, tensor):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        transform = transform.to(device=self.device, dtype=self.dtype)
        _assert_transforms_consistency(transform, tensor)

    def test_Spectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.Spectrogram(), tensor)

    def test_GriffinLim(self):
        tensor = torch.rand((1, 201, 6))
        self._assert_consistency(T.GriffinLim(length=1000, rand_init=False), tensor)

    def test_AmplitudeToDB(self):
        spec = torch.rand((6, 201))
        self._assert_consistency(T.AmplitudeToDB(), spec)

    def test_MelScale(self):
        spec_f = torch.rand((1, 6, 201))
        self._assert_consistency(T.MelScale(), spec_f)

    def test_MelSpectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MelSpectrogram(), tensor)

    def test_MFCC(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MFCC(), tensor)

    def test_Resample(self):
        tensor = torch.rand((2, 1000))
        sample_rate = 100.
        sample_rate_2 = 50.
        self._assert_consistency(T.Resample(sample_rate, sample_rate_2), tensor)

    def test_ComplexNorm(self):
        tensor = torch.rand((1, 2, 201, 2))
        self._assert_consistency(T.ComplexNorm(), tensor)

    def test_MuLawEncoding(self):
        tensor = torch.rand((1, 10))
        self._assert_consistency(T.MuLawEncoding(), tensor)

    def test_MuLawDecoding(self):
        tensor = torch.rand((1, 10))
        self._assert_consistency(T.MuLawDecoding(), tensor)

    def test_TimeStretch(self):
        n_freq = 400
        hop_length = 512
        fixed_rate = 1.3
        tensor = torch.rand((10, 2, n_freq, 10, 2))
        self._assert_consistency(
            T.TimeStretch(n_freq=n_freq, hop_length=hop_length, fixed_rate=fixed_rate),
            tensor,
        )

    def test_Fade(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)
        fade_in_len = 3000
        fade_out_len = 3000
        self._assert_consistency(T.Fade(fade_in_len, fade_out_len), waveform)

    def test_FrequencyMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.FrequencyMasking(freq_mask_param=60, iid_masks=False), tensor)

    def test_TimeMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.TimeMasking(time_mask_param=30, iid_masks=False), tensor)

    def test_Vol(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)
        self._assert_consistency(T.Vol(1.1), waveform)

    def test_SlidingWindowCmn(self):
        tensor = torch.rand((1000, 10))
        self._assert_consistency(T.SlidingWindowCmn(), tensor)

    def test_Vad(self):
        filepath = common_utils.get_asset_path("vad-go-mono-32000.wav")
        waveform, sample_rate = torchaudio.load(filepath)
        self._assert_consistency(T.Vad(sample_rate=sample_rate), waveform)


common_utils.define_test_suites(globals(), [Functional, Transforms])
