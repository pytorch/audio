"""Test definition common to CPU and CUDA"""
import math
import itertools
import warnings

import numpy as np
import torch
import torchaudio.functional as F
from parameterized import parameterized
from scipy import signal

from torchaudio_unittest.common_utils import TestBaseMixin, get_sinusoid, nested_params, get_whitenoise


class Functional(TestBaseMixin):
    def _test_resample_waveform_accuracy(self, up_scale_factor=None, down_scale_factor=None,
                                         resampling_method="sinc_interpolation", atol=1e-1, rtol=1e-4):
        # resample the signal and compare it to the ground truth
        n_to_trim = 20
        sample_rate = 1000
        new_sample_rate = sample_rate

        if up_scale_factor is not None:
            new_sample_rate *= up_scale_factor

        if down_scale_factor is not None:
            new_sample_rate //= down_scale_factor

        duration = 5  # seconds
        original_timestamps = torch.arange(0, duration, 1.0 / sample_rate)

        sound = 123 * torch.cos(2 * math.pi * 3 * original_timestamps).unsqueeze(0)
        estimate = F.resample(sound, sample_rate, new_sample_rate,
                              resampling_method=resampling_method).squeeze()

        new_timestamps = torch.arange(0, duration, 1.0 / new_sample_rate)[:estimate.size(0)]
        ground_truth = 123 * torch.cos(2 * math.pi * 3 * new_timestamps)

        # trim the first/last n samples as these points have boundary effects
        ground_truth = ground_truth[..., n_to_trim:-n_to_trim]
        estimate = estimate[..., n_to_trim:-n_to_trim]

        self.assertEqual(estimate, ground_truth, atol=atol, rtol=rtol)

    def test_lfilter_simple(self):
        """
        Create a very basic signal,
        Then make a simple 4th order delay
        The output should be same as the input but shifted
        """

        torch.random.manual_seed(42)
        waveform = torch.rand(2, 44100 * 1, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)

        self.assertEqual(output_waveform[:, 3:], waveform[:, 0:-3], atol=1e-5, rtol=1e-5)

    def test_lfilter_clamp(self):
        input_signal = torch.ones(1, 44100 * 1, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([1, 0], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, -0.95], dtype=self.dtype, device=self.device)
        output_signal = F.lfilter(input_signal, a_coeffs, b_coeffs, clamp=True)
        assert output_signal.max() <= 1
        output_signal = F.lfilter(input_signal, a_coeffs, b_coeffs, clamp=False)
        assert output_signal.max() > 1

    @parameterized.expand([
        ((44100,),),
        ((3, 44100),),
        ((2, 3, 44100),),
        ((1, 2, 3, 44100),)
    ])
    def test_lfilter_shape(self, shape):
        torch.random.manual_seed(42)
        waveform = torch.rand(*shape, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)
        assert shape == waveform.size() == output_waveform.size()

    def test_lfilter_9th_order_filter_stability(self):
        """
        Validate the precision of lfilter against reference scipy implementation when using high order filter.
        The reference implementation use cascaded second-order filters so is more numerically accurate.
        """
        # create an impulse signal
        x = torch.zeros(1024, dtype=self.dtype, device=self.device)
        x[0] = 1

        # get target impulse response
        sos = signal.butter(9, 850, 'hp', fs=22050, output='sos')
        y = torch.from_numpy(signal.sosfilt(sos, x.cpu().numpy())).to(self.dtype).to(self.device)

        # get lfilter coefficients
        b, a = signal.butter(9, 850, 'hp', fs=22050, output='ba')
        b, a = torch.from_numpy(b).to(self.dtype).to(self.device), torch.from_numpy(
            a).to(self.dtype).to(self.device)

        # predict impulse response
        yhat = F.lfilter(x, a, b, False)
        self.assertEqual(yhat, y, atol=1e-4, rtol=1e-5)

    @parameterized.expand([(0., ), (1., ), (2., ), (3., )])
    def test_spectogram_grad_at_zero(self, power):
        """The gradient of power spectrogram should not be nan but zero near x=0

        https://github.com/pytorch/audio/issues/993
        """
        x = torch.zeros(1, 22050, requires_grad=True)
        spec = F.spectrogram(
            x,
            pad=0,
            window=None,
            n_fft=2048,
            hop_length=None,
            win_length=None,
            power=power,
            normalized=False,
        )
        spec.sum().backward()
        assert not x.grad.isnan().sum()

    def test_compute_deltas_one_channel(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=self.dtype, device=self.device)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5]]], dtype=self.dtype, device=self.device)
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)

    def test_compute_deltas_two_channels(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                  [1.0, 2.0, 3.0, 4.0]]], dtype=self.dtype, device=self.device)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]], dtype=self.dtype, device=self.device)
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)

    @parameterized.expand([(100,), (440,)])
    def test_detect_pitch_frequency_pitch(self, frequency):
        sample_rate = 44100
        test_sine_waveform = get_sinusoid(
            frequency=frequency, sample_rate=sample_rate, duration=5
        )

        freq = F.detect_pitch_frequency(test_sine_waveform, sample_rate)

        threshold = 1
        s = ((freq - frequency).abs() > threshold).sum()
        self.assertFalse(s)

    @parameterized.expand([([100, 100],), ([2, 100, 100],), ([2, 2, 100, 100],)])
    def test_amplitude_to_DB_reversible(self, shape):
        """Round trip between amplitude and db should return the original for various shape

        This implicitly also tests `DB_to_amplitude`.

        """
        amplitude_mult = 20.
        power_mult = 10.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))

        torch.manual_seed(0)
        spec = torch.rand(*shape, dtype=self.dtype, device=self.device) * 200

        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 0.5)

        self.assertEqual(x2, spec, atol=5e-5, rtol=1e-5)

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, power_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 1.)

        self.assertEqual(x2, spec)

    @parameterized.expand([([100, 100],), ([2, 100, 100],), ([2, 2, 100, 100],)])
    def test_amplitude_to_DB_top_db_clamp(self, shape):
        """Ensure values are properly clamped when `top_db` is supplied."""
        amplitude_mult = 20.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.

        torch.manual_seed(0)
        # A random tensor is used for increased entropy, but the max and min for
        # each spectrogram still need to be predictable. The max determines the
        # decibel cutoff, and the distance from the min must be large enough
        # that it triggers a clamp.
        spec = torch.rand(*shape, dtype=self.dtype, device=self.device)
        # Ensure each spectrogram has a min of 0 and a max of 1.
        spec -= spec.amin([-2, -1])[..., None, None]
        spec /= spec.amax([-2, -1])[..., None, None]
        # Expand the range to (0, 200) - wide enough to properly test clamping.
        spec *= 200

        decibels = F.amplitude_to_DB(spec, amplitude_mult, amin,
                                     db_mult, top_db=top_db)
        # Ensure the clamp was applied
        below_limit = decibels < 6.0205
        assert not below_limit.any(), (
            "{} decibel values were below the expected cutoff:\n{}".format(
                below_limit.sum().item(), decibels
            )
        )
        # Ensure it didn't over-clamp
        close_to_limit = decibels < 6.0207
        assert close_to_limit.any(), (
            f"No values were close to the limit. Did it over-clamp?\n{decibels}"
        )

    @parameterized.expand(
        list(itertools.product([(1, 2, 1025, 400, 2), (1025, 400, 2)], [1, 2, 0.7]))
    )
    def test_complex_norm(self, shape, power):
        torch.random.manual_seed(42)
        complex_tensor = torch.randn(*shape, dtype=self.dtype, device=self.device)
        expected_norm_tensor = complex_tensor.pow(2).sum(-1).pow(power / 2)
        norm_tensor = F.complex_norm(complex_tensor, power)
        self.assertEqual(norm_tensor, expected_norm_tensor, atol=1e-5, rtol=1e-5)

    @parameterized.expand(
        list(itertools.product([(2, 1025, 400), (1, 201, 100)], [100], [0., 30.], [1, 2]))
    )
    def test_mask_along_axis(self, shape, mask_param, mask_value, axis):
        torch.random.manual_seed(42)
        specgram = torch.randn(*shape, dtype=self.dtype, device=self.device)
        mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis)

        other_axis = 1 if axis == 2 else 2

        masked_columns = (mask_specgram == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgram.size(other_axis)).sum()
        num_masked_columns = torch.div(
            num_masked_columns, mask_specgram.size(0), rounding_mode='floor')

        assert mask_specgram.size() == specgram.size()
        assert num_masked_columns < mask_param

    @parameterized.expand(list(itertools.product([100], [0., 30.], [2, 3])))
    def test_mask_along_axis_iid(self, mask_param, mask_value, axis):
        torch.random.manual_seed(42)
        specgrams = torch.randn(4, 2, 1025, 400, dtype=self.dtype, device=self.device)

        mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

        other_axis = 2 if axis == 3 else 3

        masked_columns = (mask_specgrams == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgrams.size(other_axis)).sum(-1)

        assert mask_specgrams.size() == specgrams.size()
        assert (num_masked_columns < mask_param).sum() == num_masked_columns.numel()

    @parameterized.expand(
        list(itertools.product([(2, 1025, 400), (1, 201, 100)], [100], [0., 30.], [1, 2]))
    )
    def test_mask_along_axis_preserve(self, shape, mask_param, mask_value, axis):
        """mask_along_axis should not alter original input Tensor

        Test is run 5 times to bound the probability of no masking occurring to 1e-10
        See https://github.com/pytorch/audio/issues/1478
        """
        torch.random.manual_seed(42)
        for _ in range(5):
            specgram = torch.randn(*shape, dtype=self.dtype, device=self.device)
            specgram_copy = specgram.clone()
            F.mask_along_axis(specgram, mask_param, mask_value, axis)

            self.assertEqual(specgram, specgram_copy)

    @parameterized.expand(list(itertools.product([100], [0., 30.], [2, 3])))
    def test_mask_along_axis_iid_preserve(self, mask_param, mask_value, axis):
        """mask_along_axis_iid should not alter original input Tensor

        Test is run 5 times to bound the probability of no masking occurring to 1e-10
        See https://github.com/pytorch/audio/issues/1478
        """
        torch.random.manual_seed(42)
        for _ in range(5):
            specgrams = torch.randn(4, 2, 1025, 400, dtype=self.dtype, device=self.device)
            specgrams_copy = specgrams.clone()
            F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

            self.assertEqual(specgrams, specgrams_copy)

    @parameterized.expand(list(itertools.product(
        ["sinc_interpolation", "kaiser_window"],
        [16000, 44100],
    )))
    def test_resample_identity(self, resampling_method, sample_rate):
        waveform = get_whitenoise(sample_rate=sample_rate, duration=1)

        resampled = F.resample(waveform, sample_rate, sample_rate)
        self.assertEqual(waveform, resampled)

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_upsample_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(sample_rate=sr, duration=0.5,)
        upsampled = F.resample(waveform, sr, sr * 2, resampling_method=resampling_method)
        assert upsampled.size(-1) == waveform.size(-1) * 2

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_downsample_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(sample_rate=sr, duration=0.5,)
        downsampled = F.resample(waveform, sr, sr // 2, resampling_method=resampling_method)
        assert downsampled.size(-1) == waveform.size(-1) // 2

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_identity_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(sample_rate=sr, duration=0.5,)
        resampled = F.resample(waveform, sr, sr, resampling_method=resampling_method)
        assert resampled.size(-1) == waveform.size(-1)

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_downsample_accuracy(self, resampling_method):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(down_scale_factor=i * 2, resampling_method=resampling_method)

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_upsample_accuracy(self, resampling_method):
        for i in range(1, 20):
            self._test_resample_waveform_accuracy(up_scale_factor=1.0 + i / 20.0, resampling_method=resampling_method)

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform_multi_channel(self, resampling_method):
        num_channels = 3
        sr = 16000
        waveform = get_whitenoise(sample_rate=sr, duration=0.5,)

        multi_sound = waveform.repeat(num_channels, 1)  # (num_channels, 8000 smp)

        for i in range(num_channels):
            multi_sound[i, :] *= (i + 1) * 1.5

        multi_sound_sampled = F.resample(multi_sound, sr, sr // 2,
                                         resampling_method=resampling_method)

        # check that sampling is same whether using separately or in a tensor of size (c, n)
        for i in range(num_channels):
            single_channel = waveform * (i + 1) * 1.5
            single_channel_sampled = F.resample(single_channel, sr, sr // 2,
                                                resampling_method=resampling_method)
            self.assertEqual(multi_sound_sampled[i, :], single_channel_sampled[0], rtol=1e-4, atol=1e-7)

    def test_resample_no_warning(self):
        sample_rate = 44100
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.resample(waveform, float(sample_rate), sample_rate / 2.)
        assert len(w) == 0

    def test_resample_warning(self):
        """resample should throw a warning if an input frequency is not of an integer value"""
        sample_rate = 44100
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.resample(waveform, sample_rate, 5512.5)
        assert len(w) == 1

    @nested_params(
        [0.5, 1.01, 1.3],
        [True, False],
    )
    def test_phase_vocoder_shape(self, rate, test_pseudo_complex):
        """Verify the output shape of phase vocoder"""
        hop_length = 256
        num_freq = 1025
        num_frames = 400
        batch_size = 2

        torch.random.manual_seed(42)
        spec = torch.randn(
            batch_size, num_freq, num_frames, dtype=self.complex_dtype, device=self.device)
        if test_pseudo_complex:
            spec = torch.view_as_real(spec)

        phase_advance = torch.linspace(
            0,
            np.pi * hop_length,
            num_freq,
            dtype=self.dtype, device=self.device)[..., None]

        spec_stretch = F.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

        assert spec.dim() == spec_stretch.dim()
        expected_shape = torch.Size([batch_size, num_freq, int(np.ceil(num_frames / rate))])
        output_shape = (torch.view_as_complex(spec_stretch) if test_pseudo_complex else spec_stretch).shape
        assert output_shape == expected_shape


class FunctionalCPUOnly(TestBaseMixin):
    def test_create_fb_matrix_no_warning_high_n_freq(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(288, 0, 8000, 128, 16000)
        assert len(w) == 0

    def test_create_fb_matrix_no_warning_low_n_mels(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(201, 0, 8000, 89, 16000)
        assert len(w) == 0

    def test_create_fb_matrix_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(201, 0, 8000, 128, 16000)
        assert len(w) == 1
