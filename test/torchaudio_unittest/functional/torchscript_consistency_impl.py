"""Test suites for jit-ability and its numerical compatibility"""
import unittest

import torch
import torchaudio.functional as F
from parameterized import parameterized
from torchaudio_unittest import common_utils
from torchaudio_unittest.common_utils import skipIfRocm, TempDirMixin, TestBaseMixin, torch_script


class Functional(TempDirMixin, TestBaseMixin):
    """Implements test for `functional` module that are performed for different devices"""

    def _assert_consistency(self, func, inputs, shape_only=False):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(device=self.device, dtype=self.dtype)
            inputs_.append(i)
        ts_func = torch_script(func)

        torch.random.manual_seed(40)
        output = func(*inputs_)

        torch.random.manual_seed(40)
        ts_output = ts_func(*inputs_)

        if shape_only:
            ts_output = ts_output.shape
            output = output.shape
        self.assertEqual(ts_output, output)

    def _assert_consistency_complex(self, func, inputs):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=self.complex_dtype if i.is_complex() else self.dtype, device=self.device)
            inputs_.append(i)
        ts_func = torch_script(func)

        torch.random.manual_seed(40)
        output = func(*inputs_)

        torch.random.manual_seed(40)
        ts_output = ts_func(*inputs_)

        self.assertEqual(ts_output, output)

    @parameterized.expand(
        [
            (True,),
            (False,),
            ("window",),
            ("frame_length",),
        ]
    )
    def test_spectrogram(self, normalize):
        waveform = common_utils.get_whitenoise()
        n_fft = 400
        ws = 400
        hop = 200
        pad = 0
        window = torch.hann_window(ws, device=waveform.device, dtype=waveform.dtype)
        power = None
        self._assert_consistency(
            F.spectrogram, (waveform, pad, window, n_fft, hop, ws, power, normalize, True, "reflect", True, True)
        )

    @parameterized.expand(
        [
            (True,),
            (False,),
            ("window",),
            ("frame_length",),
        ]
    )
    def test_inverse_spectrogram(self, normalize):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=0.05)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400, hop_length=200)
        length = 400
        n_fft = 400
        hop = 200
        ws = 400
        pad = 0
        window = torch.hann_window(ws, device=specgram.device, dtype=torch.float64)
        self._assert_consistency_complex(
            F.inverse_spectrogram, (specgram, length, pad, window, n_fft, hop, ws, normalize, True, "reflect", True)
        )

    @skipIfRocm
    def test_griffinlim(self):
        tensor = torch.rand((1, 201, 6))
        n_fft = 400
        ws = 400
        hop = 200
        window = torch.hann_window(ws, device=tensor.device, dtype=tensor.dtype)
        power = 2.0
        momentum = 0.99
        n_iter = 32
        length = 1000
        rand_int = False
        self._assert_consistency(
            F.griffinlim, (tensor, window, n_fft, hop, ws, power, n_iter, momentum, length, rand_int)
        )

    def test_compute_deltas(self):
        channel = 13
        n_mfcc = channel * 3
        time = 1021
        tensor = torch.randn(channel, n_mfcc, time)
        win_length = 2 * 7 + 1
        self._assert_consistency(F.compute_deltas, (tensor, win_length, "replicate"))

    def test_detect_pitch_frequency(self):
        waveform = common_utils.get_sinusoid(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            return F.detect_pitch_frequency(tensor, sample_rate)

        self._assert_consistency(func, (waveform,))

    def test_measure_loudness(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        sample_rate = 44100
        waveform = common_utils.get_sinusoid(sample_rate=sample_rate, device=self.device)
        self._assert_consistency(F.loudness, (waveform, sample_rate))

    def test_melscale_fbanks(self):
        if self.device != torch.device("cpu"):
            raise unittest.SkipTest("No need to perform test on device other than CPU")

        n_stft = 100
        f_min = 0.0
        f_max = 20.0
        n_mels = 10
        sample_rate = 16000
        norm = "slaney"
        self._assert_consistency(F.melscale_fbanks, (n_stft, f_min, f_max, n_mels, sample_rate, norm, "htk"))

    def test_linear_fbanks(self):
        if self.device != torch.device("cpu"):
            raise unittest.SkipTest("No need to perform test on device other than CPU")

        n_stft = 100
        f_min = 0.0
        f_max = 20.0
        n_filter = 10
        sample_rate = 16000
        self._assert_consistency(F.linear_fbanks, (n_stft, f_min, f_max, n_filter, sample_rate))

    def test_amplitude_to_DB(self):
        tensor = torch.rand((6, 201))
        multiplier = 10.0
        amin = 1e-10
        db_multiplier = 0.0
        top_db = 80.0
        self._assert_consistency(F.amplitude_to_DB, (tensor, multiplier, amin, db_multiplier, top_db))

    def test_DB_to_amplitude(self):
        tensor = torch.rand((1, 100))
        ref = 1.0
        power = 1.0
        self._assert_consistency(F.DB_to_amplitude, (tensor, ref, power))

    def test_create_dct(self):
        if self.device != torch.device("cpu"):
            raise unittest.SkipTest("No need to perform test on device other than CPU")

        n_mfcc = 40
        n_mels = 128
        norm = "ortho"
        self._assert_consistency(F.create_dct, (n_mfcc, n_mels, norm))

    def test_mu_law_encoding(self):
        def func(tensor):
            qc = 256
            return F.mu_law_encoding(tensor, qc)

        waveform = common_utils.get_whitenoise()
        self._assert_consistency(func, (waveform,))

    def test_mu_law_decoding(self):
        def func(tensor):
            qc = 256
            return F.mu_law_decoding(tensor, qc)

        tensor = torch.rand((1, 10))
        self._assert_consistency(func, (tensor,))

    def test_mask_along_axis(self):
        def func(tensor):
            mask_param = 100
            mask_value = 30.0
            axis = 2
            return F.mask_along_axis(tensor, mask_param, mask_value, axis)

        tensor = torch.randn(2, 1025, 400)
        self._assert_consistency(func, (tensor,))

    def test_mask_along_axis_iid(self):
        def func(tensor):
            mask_param = 100
            mask_value = 30.0
            axis = 2
            return F.mask_along_axis_iid(tensor, mask_param, mask_value, axis)

        tensor = torch.randn(4, 2, 1025, 400)
        self._assert_consistency(func, (tensor,))

    def test_gain(self):
        def func(tensor):
            gainDB = 2.0
            return F.gain(tensor, gainDB)

        tensor = torch.rand((1, 1000))
        self._assert_consistency(func, (tensor,))

    def test_dither_TPDF(self):
        def func(tensor):
            return F.dither(tensor, "TPDF")

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, (tensor,), shape_only=True)

    def test_dither_RPDF(self):
        def func(tensor):
            return F.dither(tensor, "RPDF")

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, (tensor,), shape_only=True)

    def test_dither_GPDF(self):
        def func(tensor):
            return F.dither(tensor, "GPDF")

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, (tensor,), shape_only=True)

    def test_dither_noise_shaping(self):
        def func(tensor):
            return F.dither(tensor, noise_shaping=True)

        tensor = common_utils.get_whitenoise(n_channels=2)
        self._assert_consistency(func, (tensor,))

    def test_lfilter(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise()
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
            device=waveform.device,
            dtype=waveform.dtype,
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
            dtype=waveform.dtype,
        )
        self._assert_consistency(F.lfilter, (waveform, a_coeffs, b_coeffs, True, True))

    def test_filtfilt(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000)
        b_coeffs = torch.rand(4, device=waveform.device, dtype=waveform.dtype)
        a_coeffs = torch.rand(4, device=waveform.device, dtype=waveform.dtype)
        self._assert_consistency(F.filtfilt, (waveform, a_coeffs, b_coeffs, True))

    def test_lowpass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 3000.0
            return F.lowpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, (waveform,))

    def test_highpass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            cutoff_freq = 2000.0
            return F.highpass_biquad(tensor, sample_rate, cutoff_freq)

        self._assert_consistency(func, (waveform,))

    def test_allpass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            return F.allpass_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, (waveform,))

    def test_bandpass_with_csg(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, (waveform,))

    def test_bandpass_without_csg(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            const_skirt_gain = True
            return F.bandpass_biquad(tensor, sample_rate, central_freq, q, const_skirt_gain)

        self._assert_consistency(func, (waveform,))

    def test_bandreject(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            return F.bandreject_biquad(tensor, sample_rate, central_freq, q)

        self._assert_consistency(func, (waveform,))

    def test_band_with_noise(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            noise = True
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, (waveform,))

    def test_band_without_noise(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            central_freq = 1000.0
            q = 0.707
            noise = False
            return F.band_biquad(tensor, sample_rate, central_freq, q, noise)

        self._assert_consistency(func, (waveform,))

    def test_treble(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            gain = 40.0
            central_freq = 1000.0
            q = 0.707
            return F.treble_biquad(tensor, sample_rate, gain, central_freq, q)

        self._assert_consistency(func, (waveform,))

    def test_bass(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            gain = 40.0
            central_freq = 1000.0
            q = 0.707
            return F.bass_biquad(tensor, sample_rate, gain, central_freq, q)

        self._assert_consistency(func, (waveform,))

    def test_deemph(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            return F.deemph_biquad(tensor, sample_rate)

        self._assert_consistency(func, (waveform,))

    def test_riaa(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            return F.riaa_biquad(tensor, sample_rate)

        self._assert_consistency(func, (waveform,))

    def test_equalizer(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise(sample_rate=44100)

        def func(tensor):
            sample_rate = 44100
            center_freq = 300.0
            gain = 1.0
            q = 0.707
            return F.equalizer_biquad(tensor, sample_rate, center_freq, gain, q)

        self._assert_consistency(func, (waveform,))

    def test_perf_biquad_filtering(self):
        if self.dtype == torch.float64:
            raise unittest.SkipTest("This test is known to fail for float64")

        waveform = common_utils.get_whitenoise()

        def func(tensor):
            a = torch.tensor([0.7, 0.2, 0.6], device=tensor.device, dtype=tensor.dtype)
            b = torch.tensor([0.4, 0.2, 0.9], device=tensor.device, dtype=tensor.dtype)
            return F.lfilter(tensor, a, b)

        self._assert_consistency(func, (waveform,))

    def test_sliding_window_cmn(self):
        def func(tensor):
            cmn_window = 600
            min_cmn_window = 100
            center = False
            norm_vars = False
            a = torch.tensor(
                [[-1.915875792503357, 1.147700309753418], [1.8242558240890503, 1.3869990110397339]],
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return F.sliding_window_cmn(a, cmn_window, min_cmn_window, center, norm_vars)

        b = torch.tensor([[-1.8701, -0.1196], [1.8701, 0.1196]])
        self._assert_consistency(func, (b,))

    def test_contrast(self):
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            enhancement_amount = 80.0
            return F.contrast(tensor, enhancement_amount)

        self._assert_consistency(func, (waveform,))

    def test_dcshift(self):
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            shift = 0.5
            limiter_gain = 0.05
            return F.dcshift(tensor, shift, limiter_gain)

        self._assert_consistency(func, (waveform,))

    def test_overdrive(self):
        waveform = common_utils.get_whitenoise()

        def func(tensor):
            gain = 30.0
            colour = 50.0
            return F.overdrive(tensor, gain, colour)

        self._assert_consistency(func, (waveform,))

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

        self._assert_consistency(func, (waveform,))

    def test_flanger(self):
        waveform = torch.rand(2, 100) - 0.5

        def func(tensor):
            delay = 0.8
            depth = 0.88
            regen = 3.0
            width = 0.23
            speed = 1.3
            phase = 60.0
            sample_rate = 44100
            return F.flanger(
                tensor,
                sample_rate,
                delay,
                depth,
                regen,
                width,
                speed,
                phase,
                modulation="sinusoidal",
                interpolation="linear",
            )

        self._assert_consistency(func, (waveform,))

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
        self._assert_consistency(func, (tensor,))

    def test_resample_sinc(self):
        def func(tensor):
            sr1, sr2 = 16000, 8000
            return F.resample(tensor, sr1, sr2, resampling_method="sinc_interp_hann")

        tensor = common_utils.get_whitenoise(sample_rate=16000)
        self._assert_consistency(func, (tensor,))

    @parameterized.expand(
        [
            (None,),
            (6.0,),
        ]
    )
    def test_resample_kaiser(self, beta):
        tensor = common_utils.get_whitenoise(sample_rate=16000)
        sr1, sr2 = 16000, 8000
        lowpass_filter_width = 6
        rolloff = 0.99
        self._assert_consistency(
            F.resample, (tensor, sr1, sr2, lowpass_filter_width, rolloff, "sinc_interp_kaiser", beta)
        )

    def test_phase_vocoder(self):
        tensor = torch.view_as_complex(torch.randn(2, 1025, 400, 2))
        n_freq = tensor.size(-2)
        rate = 0.5
        hop_length = 256
        phase_advance = torch.linspace(
            0,
            3.14 * hop_length,
            n_freq,
            dtype=torch.real(tensor).dtype,
            device=tensor.device,
        )[..., None]
        self._assert_consistency_complex(F.phase_vocoder, (tensor, rate, phase_advance))

    def test_psd(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        frame = 10
        normalize = True
        eps = 1e-10
        tensor = torch.rand(batch_size, channel, n_fft_bin, frame, dtype=self.complex_dtype)
        self._assert_consistency_complex(F.psd, (tensor, None, normalize, eps))

    def test_psd_with_mask(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        frame = 10
        normalize = True
        eps = 1e-10
        specgram = torch.rand(batch_size, channel, n_fft_bin, frame, dtype=self.complex_dtype)
        mask = torch.rand(batch_size, n_fft_bin, frame, device=self.device)
        self._assert_consistency_complex(F.psd, (specgram, mask, normalize, eps))

    def test_mvdr_weights_souden(self):
        channel = 4
        n_fft_bin = 10
        diagonal_loading = True
        diag_eps = 1e-7
        eps = 1e-8
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        self._assert_consistency_complex(
            F.mvdr_weights_souden, (psd_speech, psd_noise, 0, diagonal_loading, diag_eps, eps)
        )

    def test_mvdr_weights_souden_with_tensor(self):
        channel = 4
        n_fft_bin = 10
        diagonal_loading = True
        diag_eps = 1e-7
        eps = 1e-8
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(channel)
        reference_channel[..., 0].fill_(1)
        self._assert_consistency_complex(
            F.mvdr_weights_souden, (psd_speech, psd_noise, reference_channel, diagonal_loading, diag_eps, eps)
        )

    def test_mvdr_weights_rtf(self):
        channel = 4
        n_fft_bin = 10
        diagonal_loading = True
        diag_eps = 1e-7
        eps = 1e-8
        rtf = torch.rand(n_fft_bin, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        reference_channel = 0
        self._assert_consistency_complex(
            F.mvdr_weights_rtf, (rtf, psd_noise, reference_channel, diagonal_loading, diag_eps, eps)
        )

    def test_mvdr_weights_rtf_with_tensor(self):
        channel = 4
        n_fft_bin = 10
        diagonal_loading = True
        diag_eps = 1e-7
        eps = 1e-8
        rtf = torch.rand(n_fft_bin, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        reference_channel = torch.zeros(channel)
        reference_channel[..., 0].fill_(1)
        self._assert_consistency_complex(
            F.mvdr_weights_rtf, (rtf, psd_noise, reference_channel, diagonal_loading, diag_eps, eps)
        )

    def test_rtf_evd(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 129
        tensor = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=self.complex_dtype)
        self._assert_consistency_complex(F.rtf_evd, (tensor,))

    @parameterized.expand(
        [
            (1, True),
            (3, False),
        ]
    )
    def test_rtf_power(self, n_iter, diagonal_loading):
        channel = 4
        n_fft_bin = 10
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        reference_channel = 0
        diag_eps = 1e-7
        self._assert_consistency_complex(
            F.rtf_power, (psd_speech, psd_noise, reference_channel, n_iter, diagonal_loading, diag_eps)
        )

    @parameterized.expand(
        [
            (1, True),
            (3, False),
        ]
    )
    def test_rtf_power_with_tensor(self, n_iter, diagonal_loading):
        channel = 4
        n_fft_bin = 10
        psd_speech = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        psd_noise = torch.rand(n_fft_bin, channel, channel, dtype=self.complex_dtype)
        reference_channel = torch.zeros(channel)
        reference_channel[..., 0].fill_(1)
        diag_eps = 1e-7
        self._assert_consistency_complex(
            F.rtf_power, (psd_speech, psd_noise, reference_channel, n_iter, diagonal_loading, diag_eps)
        )

    def test_apply_beamforming(self):
        num_channels = 4
        n_fft_bin = 201
        num_frames = 10
        beamform_weights = torch.rand(n_fft_bin, num_channels, dtype=self.complex_dtype, device=self.device)
        specgram = torch.rand(num_channels, n_fft_bin, num_frames, dtype=self.complex_dtype, device=self.device)
        self._assert_consistency_complex(F.apply_beamforming, (beamform_weights, specgram))

    @common_utils.nested_params(
        ["convolve", "fftconvolve"],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        self._assert_consistency(getattr(F, fn), (x, y, mode))

    @common_utils.nested_params([True, False])
    def test_add_noise(self, use_lengths):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        if use_lengths:
            lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            lengths = None
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self._assert_consistency(F.add_noise, (waveform, noise, snr, lengths))

    @common_utils.nested_params([True, False])
    def test_speed(self, use_lengths):
        leading_dims = (3, 2)
        T = 200
        waveform = torch.rand(*leading_dims, T, dtype=self.dtype, device=self.device, requires_grad=True)
        if use_lengths:
            lengths = torch.randint(1, T, leading_dims, dtype=self.dtype, device=self.device)
        else:
            lengths = None
        self._assert_consistency(F.speed, (waveform, 1000, 1.1, lengths))

    def test_preemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype)
        coeff = 0.9
        self._assert_consistency(F.preemphasis, (waveform, coeff))

    def test_deemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype)
        coeff = 0.9
        self._assert_consistency(F.deemphasis, (waveform, coeff))


class FunctionalFloat32Only(TestBaseMixin):
    def test_rnnt_loss(self):
        def func(tensor):
            targets = torch.tensor([[1, 2]], device=tensor.device, dtype=torch.int32)
            logit_lengths = torch.tensor([2], device=tensor.device, dtype=torch.int32)
            target_lengths = torch.tensor([2], device=tensor.device, dtype=torch.int32)
            return F.rnnt_loss(tensor, targets, logit_lengths, target_lengths)

        logits = torch.tensor(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        tensor = logits.to(device=self.device, dtype=torch.float32)
        self._assert_consistency(func, (tensor,))
