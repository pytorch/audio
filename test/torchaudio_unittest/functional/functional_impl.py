"""Test definition common to CPU and CUDA"""
import math
import itertools
import warnings

import numpy as np
import torch
import torchaudio.functional as F
from parameterized import parameterized
from scipy import signal

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_sinusoid,
    nested_params,
    get_whitenoise,
    rnnt_utils,
)


class Functional(TestBaseMixin):
    def _test_resample_waveform_accuracy(self, up_scale_factor=None, down_scale_factor=None,
                                         resampling_method="sinc_interpolation", atol=1e-1, rtol=1e-4):
        # resample the signal and compare it to the ground truth
        n_to_trim = 20
        sample_rate = 1000
        new_sample_rate = sample_rate

        if up_scale_factor is not None:
            new_sample_rate = int(new_sample_rate * up_scale_factor)

        if down_scale_factor is not None:
            new_sample_rate = int(new_sample_rate / down_scale_factor)

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

    def _test_costs_and_gradients(
        self, data, ref_costs, ref_gradients, atol=1e-6, rtol=1e-2
    ):
        logits_shape = data["logits"].shape
        costs, gradients = rnnt_utils.compute_with_pytorch_transducer(data=data)
        self.assertEqual(costs, ref_costs, atol=atol, rtol=rtol)
        self.assertEqual(logits_shape, gradients.shape)
        self.assertEqual(gradients, ref_gradients, atol=atol, rtol=rtol)

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
        ((44100,), (4,), (44100,)),
        ((3, 44100), (4,), (3, 44100,)),
        ((2, 3, 44100), (4,), (2, 3, 44100,)),
        ((1, 2, 3, 44100), (4,), (1, 2, 3, 44100,)),
        ((44100,), (2, 4), (2, 44100)),
        ((3, 44100), (1, 4), (3, 1, 44100)),
        ((1, 2, 44100), (3, 4), (1, 2, 3, 44100))
    ])
    def test_lfilter_shape(self, input_shape, coeff_shape, target_shape):
        torch.random.manual_seed(42)
        waveform = torch.rand(*input_shape, dtype=self.dtype, device=self.device)
        b_coeffs = torch.rand(*coeff_shape, dtype=self.dtype, device=self.device)
        a_coeffs = torch.rand(*coeff_shape, dtype=self.dtype, device=self.device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs, batching=False)
        assert input_shape == waveform.size()
        assert target_shape == output_waveform.size()

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

    def test_filtfilt_simple(self):
        """
        Check that, for an arbitrary signal, applying filtfilt with filter coefficients
        corresponding to a pure delay filter imparts no time delay.
        """
        waveform = get_whitenoise(sample_rate=8000, n_channels=2, dtype=self.dtype).to(
            device=self.device
        )
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        padded_waveform = torch.cat(
            (waveform, torch.zeros(2, 3, dtype=self.dtype, device=self.device)), axis=1
        )
        output_waveform = F.filtfilt(padded_waveform, a_coeffs, b_coeffs)

        self.assertEqual(output_waveform, padded_waveform, atol=1e-5, rtol=1e-5)

    def test_filtfilt_filter_sinusoid(self):
        """
        Check that, for a signal comprising two sinusoids, applying filtfilt
        with appropriate filter coefficients correctly removes the higher-frequency
        sinusoid while imparting no time delay.
        """
        T = 1.0
        samples = 1000

        waveform_k0 = get_sinusoid(
            frequency=5, sample_rate=samples // T, dtype=self.dtype, device=self.device
        ).squeeze(0)
        waveform_k1 = get_sinusoid(
            frequency=200,
            sample_rate=samples // T,
            dtype=self.dtype,
            device=self.device,
        ).squeeze(0)
        waveform = waveform_k0 + waveform_k1

        # Transfer function numerator and denominator polynomial coefficients
        # corresponding to 8th-order Butterworth filter with 100-cycle/T cutoff.
        # Generated with
        # >>> from scipy import signal
        # >>> b_coeffs, a_coeffs = signal.butter(8, 0.2)
        b_coeffs = torch.tensor(
            [
                2.39596441e-05,
                1.91677153e-04,
                6.70870035e-04,
                1.34174007e-03,
                1.67717509e-03,
                1.34174007e-03,
                6.70870035e-04,
                1.91677153e-04,
                2.39596441e-05,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        a_coeffs = torch.tensor(
            [
                1.0,
                -4.78451489,
                10.44504107,
                -13.45771989,
                11.12933104,
                -6.0252604,
                2.0792738,
                -0.41721716,
                0.0372001,
            ],
            dtype=self.dtype,
            device=self.device,
        )

        # Extend waveform in each direction, preserving periodicity.
        padded_waveform = torch.cat((waveform[:-1], waveform, waveform[1:]))

        output_waveform = F.filtfilt(padded_waveform, a_coeffs, b_coeffs)

        # Remove padding from output waveform; confirm that result
        # closely matches waveform_k0.
        self.assertEqual(
            output_waveform[samples - 1: 2 * samples - 1],
            waveform_k0,
            atol=1e-3,
            rtol=1e-3,
        )

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

    @parameterized.expand(list(itertools.product(
        ["sinc_interpolation", "kaiser_window"],
        list(range(1, 20)),
    )))
    def test_resample_waveform_downsample_accuracy(self, resampling_method, i):
        self._test_resample_waveform_accuracy(down_scale_factor=i * 2, resampling_method=resampling_method)

    @parameterized.expand(list(itertools.product(
        ["sinc_interpolation", "kaiser_window"],
        list(range(1, 20)),
    )))
    def test_resample_waveform_upsample_accuracy(self, resampling_method, i):
        self._test_resample_waveform_accuracy(up_scale_factor=1.0 + i / 20.0, resampling_method=resampling_method)

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

    @parameterized.expand(
        [
            # words
            ["", "", 0],  # equal
            ["abc", "abc", 0],
            ["ᑌᑎIᑕO", "ᑌᑎIᑕO", 0],

            ["abc", "", 3],  # deletion
            ["aa", "aaa", 1],
            ["aaa", "aa", 1],
            ["ᑌᑎI", "ᑌᑎIᑕO", 2],

            ["aaa", "aba", 1],  # substitution
            ["aba", "aaa", 1],
            ["aba", "   ", 3],

            ["abc", "bcd", 2],  # mix deletion and substitution
            ["0ᑌᑎI", "ᑌᑎIᑕO", 3],

            # sentences
            [["hello", "", "Tᕮ᙭T"], ["hello", "", "Tᕮ᙭T"], 0],  # equal
            [[], [], 0],

            [["hello", "world"], ["hello", "world", "!"], 1],  # deletion
            [["hello", "world"], ["world"], 1],
            [["hello", "world"], [], 2],

            [["Tᕮ᙭T", ], ["world"], 1],  # substitution
            [["Tᕮ᙭T", "XD"], ["world", "hello"], 2],
            [["", "XD"], ["world", ""], 2],
            ["aba", "   ", 3],

            [["hello", "world"], ["world", "hello", "!"], 2],  # mix deletion and substitution
            [["Tᕮ᙭T", "world", "LOL", "XD"], ["world", "hello", "ʕ•́ᴥ•̀ʔっ"], 3],
        ]
    )
    def test_simple_case_edit_distance(self, seq1, seq2, distance):
        assert F.edit_distance(seq1, seq2) == distance
        assert F.edit_distance(seq2, seq1) == distance

    @nested_params(
        [-4, -2, 0, 2, 4],
    )
    def test_pitch_shift_shape(self, n_steps):
        sample_rate = 16000
        torch.random.manual_seed(42)
        waveform = torch.rand(2, 44100 * 1, dtype=self.dtype, device=self.device)
        waveform_shift = F.pitch_shift(waveform, sample_rate, n_steps)
        assert waveform.size() == waveform_shift.size()

    def test_rnnt_loss_basic_backward(self):
        logits, targets, logit_lengths, target_lengths = rnnt_utils.get_basic_data(self.device)
        loss = F.rnnt_loss(logits, targets, logit_lengths, target_lengths)
        loss.backward()

    def test_rnnt_loss_basic_forward_no_grad(self):
        """In early stage, calls to `rnnt_loss` resulted in segmentation fault when
        `logits` have `requires_grad = False`. This test makes sure that this no longer
        occurs and the functional call runs without error.

        See https://github.com/pytorch/audio/pull/1707
        """
        logits, targets, logit_lengths, target_lengths = rnnt_utils.get_basic_data(self.device)
        logits.requires_grad_(False)
        F.rnnt_loss(logits, targets, logit_lengths, target_lengths)

    @parameterized.expand([
        (rnnt_utils.get_B1_T2_U3_D5_data, torch.float32, 1e-6, 1e-2),
        (rnnt_utils.get_B2_T4_U3_D3_data, torch.float32, 1e-6, 1e-2),
        (rnnt_utils.get_B1_T2_U3_D5_data, torch.float16, 1e-3, 1e-2),
        (rnnt_utils.get_B2_T4_U3_D3_data, torch.float16, 1e-3, 1e-2),
    ])
    def test_rnnt_loss_costs_and_gradients(self, data_func, dtype, atol, rtol):
        data, ref_costs, ref_gradients = data_func(
            dtype=dtype,
            device=self.device,
        )
        self._test_costs_and_gradients(
            data=data,
            ref_costs=ref_costs,
            ref_gradients=ref_gradients,
            atol=atol,
            rtol=rtol,
        )

    def test_rnnt_loss_costs_and_gradients_random_data_with_numpy_fp32(self):
        seed = 777
        for i in range(5):
            data = rnnt_utils.get_random_data(dtype=torch.float32, device=self.device, seed=(seed + i))
            ref_costs, ref_gradients = rnnt_utils.compute_with_numpy_transducer(data=data)
            self._test_costs_and_gradients(
                data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
            )


class FunctionalCPUOnly(TestBaseMixin):
    def test_melscale_fbanks_no_warning_high_n_freq(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.melscale_fbanks(288, 0, 8000, 128, 16000)
        assert len(w) == 0

    def test_melscale_fbanks_no_warning_low_n_mels(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.melscale_fbanks(201, 0, 8000, 89, 16000)
        assert len(w) == 0

    def test_melscale_fbanks_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.melscale_fbanks(201, 0, 8000, 128, 16000)
        assert len(w) == 1
