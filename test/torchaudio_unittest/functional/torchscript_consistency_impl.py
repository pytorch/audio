"""Test suites for jit-ability and its numerical compatibility"""
import unittest
import numpy as np

import torch
import torchaudio.functional as F

from torchaudio_unittest import common_utils


class Functional(common_utils.TestBaseMixin):
    """Implements test for `functinoal` modul that are performed for different devices"""
    def _assert_consistency(self, func, tensor, shape_only=False):
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        ts_func = torch.jit.script(func)
        output = func(tensor)
        ts_output = ts_func(tensor)
        if shape_only:
            ts_output = ts_output.shape
            output = output.shape
        self.assertEqual(ts_output, output)

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

        tensor = common_utils.get_whitenoise()
        self._assert_consistency(func, tensor)

    def test_griffinlim(self):
        def func(tensor):
            n_fft = 400
            ws = 400
            hop = 200
            window = torch.hann_window(ws, device=tensor.device, dtype=tensor.dtype)
            power = 2.
            momentum = 0.99
            n_iter = 32
            length = 1000
            rand_int = False
            return F.griffinlim(tensor, window, n_fft, hop, ws, power, n_iter, momentum, length, rand_int)

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
        waveform = common_utils.get_sinusoid(sample_rate=44100)

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
            norm = "slaney"
            return F.create_fb_matrix(n_stft, f_min, f_max, n_mels, sample_rate, norm)

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

        waveform = common_utils.get_whitenoise()
        self._assert_consistency(func, waveform)

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

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, tensor, shape_only=True)

    def test_dither_RPDF(self):
        def func(tensor):
            return F.dither(tensor, 'RPDF')

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, tensor, shape_only=True)

    def test_dither_GPDF(self):
        def func(tensor):
            return F.dither(tensor, 'GPDF')

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, tensor, shape_only=True)

    def test_dither_noise_shaping(self):
        def func(tensor):
            return F.dither(tensor, noise_shaping=True)

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, tensor)

    def test_lfilter(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise()

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
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 3000.
            return F.lowpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, waveform)

    def test_highpass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 2000.
            return F.highpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, waveform)

    def test_allpass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            return F.allpass_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_bandpass_with_csg(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, waveform)

    def test_bandpass_without_csg(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, waveform)

    def test_bandreject(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            return F.bandreject_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_band_with_noise(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            noise = True
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, waveform)

    def test_band_without_noise(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.
            q = 0.707
            noise = False
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, waveform)

    def test_treble(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            gain = 40.
            central_freq = 1000.
            q = 0.707
            return F.treble_biquad(tensor, sample_rate, gain, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_bass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            gain = 40.
            central_freq = 1000.
            q = 0.707
            return F.bass_biquad(tensor, sample_rate, gain, central_freq, q)

        self._assert_consistency(func, waveform)

    def test_deemph(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            return F.deemph_biquad(tensor, sample_rate)

        self._assert_consistency(func, waveform)

    def test_riaa(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            return F.riaa_biquad(tensor, sample_rate)

        self._assert_consistency(func, waveform)

    def test_equalizer(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            center_freq = 300.
            gain = 1.
            q = 0.707
            return F.equalizer_biquad(tensor, sample_rate, center_freq, gain, q)

        self._assert_consistency(func, waveform)

    def test_perf_biquad_filtering(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise()

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
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            enhancement_amount = 80.
            return F.contrast(tensor, enhancement_amount)

        self._assert_consistency(func, waveform)

    def test_dcshift(self):
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            shift = 0.5
            limiter_gain = 0.05
            return F.dcshift(tensor, shift, limiter_gain)

        self._assert_consistency(func, waveform)

    def test_overdrive(self):
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            gain = 30.
            colour = 50.
            return F.overdrive(tensor, gain, colour)

        self._assert_consistency(func, waveform)

    def test_phaser(self):
        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            gain_in = 0.5
            gain_out = 0.8
            delay_ms = 2.0
            decay = 0.4
            speed = 0.5
            sample_rate = 44100
            return F.phaser(tensor, sample_rate, gain_in, gain_out, delay_ms, decay, speed, sinusoidal=True)

        self._assert_consistency(func, waveform)

    def test_flanger(self):
        torch.random.manual_seed(40)
        waveform = torch.rand(2, 100) - 0.5

        def func(tensor):
            delay = 0.8
            depth = 0.88
            regen = 3.0
            width = 0.23
            speed = 1.3
            phase = 60.
            sample_rate = 44100
            return F.flanger(tensor, sample_rate, delay, depth, regen, width, speed,
                             phase, modulation='sinusoidal', interpolation='linear')

        self._assert_consistency(func, waveform)

    def test_spectral_centroid(self):

        def func(tensor):
            sample_rate = 44100
            n_fft = 400
            ws = 400
            hop = 200
            pad = 0
            window = torch.hann_window(ws, device=tensor.device, dtype=tensor.dtype)
            return F.spectral_centroid(tensor, sample_rate, pad, window, n_fft, hop, ws)

        tensor = common_utils.get_whitenoise(sample_rate=44100)
        self._assert_consistency(func, tensor)

    def test_phase_vocoder(self):
        def func(tensor):
            rate = 0.5
            hop_length = 256
            phase_advance = torch.linspace(
                0,
                np.pi * hop_length,
                tensor.shape[-3],
                dtype=torch.float64)[..., None]
            return F.phase_vocoder(tensor, rate, phase_advance)

        tensor = torch.randn(2, 1025, 400, 2)
        self._assert_consistency(func, tensor)

    @common_utils.skipIfNoKaldi
    def test_compute_kaldi_pitch(self):
        if self.dtype != torch.float32 or self.device != torch.device('cpu'):
            raise unittest.SkipTest("Only float32, cpu is supported.")

        def func(tensor):
            sample_rate: float = 44100.
            return F.compute_kaldi_pitch(tensor, sample_rate)

        tensor = common_utils.get_whitenoise(sample_rate=44100)
        self._assert_consistency(func, tensor)
