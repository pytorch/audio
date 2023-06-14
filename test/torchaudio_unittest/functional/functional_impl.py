"""Test definition common to CPU and CUDA"""
import itertools
import math
import warnings

import numpy as np
import torch
import torchaudio.functional as F
from parameterized import parameterized
from scipy import signal
from torchaudio_unittest.common_utils import (
    beamform_utils,
    get_sinusoid,
    get_whitenoise,
    nested_params,
    rnnt_utils,
    TestBaseMixin,
)


class Functional(TestBaseMixin):
    def _test_resample_waveform_accuracy(
        self, up_scale_factor=None, down_scale_factor=None, resampling_method="sinc_interp_hann", atol=1e-1, rtol=1e-4
    ):
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
        estimate = F.resample(sound, sample_rate, new_sample_rate, resampling_method=resampling_method).squeeze()

        new_timestamps = torch.arange(0, duration, 1.0 / new_sample_rate)[: estimate.size(0)]
        ground_truth = 123 * torch.cos(2 * math.pi * 3 * new_timestamps)

        # trim the first/last n samples as these points have boundary effects
        ground_truth = ground_truth[..., n_to_trim:-n_to_trim]
        estimate = estimate[..., n_to_trim:-n_to_trim]

        self.assertEqual(estimate, ground_truth, atol=atol, rtol=rtol)

    def _test_costs_and_gradients(self, data, ref_costs, ref_gradients, atol=1e-6, rtol=1e-2):
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

    @parameterized.expand(
        [
            ((44100,), (4,), (44100,)),
            (
                (3, 44100),
                (4,),
                (
                    3,
                    44100,
                ),
            ),
            (
                (2, 3, 44100),
                (4,),
                (
                    2,
                    3,
                    44100,
                ),
            ),
            (
                (1, 2, 3, 44100),
                (4,),
                (
                    1,
                    2,
                    3,
                    44100,
                ),
            ),
            ((44100,), (2, 4), (2, 44100)),
            ((3, 44100), (1, 4), (3, 1, 44100)),
            ((1, 2, 44100), (3, 4), (1, 2, 3, 44100)),
        ]
    )
    def test_lfilter_shape(self, input_shape, coeff_shape, target_shape):
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
        sos = signal.butter(9, 850, "hp", fs=22050, output="sos")
        y = torch.from_numpy(signal.sosfilt(sos, x.cpu().numpy())).to(self.dtype).to(self.device)

        # get lfilter coefficients
        b, a = signal.butter(9, 850, "hp", fs=22050, output="ba")
        b, a = torch.from_numpy(b).to(self.dtype).to(self.device), torch.from_numpy(a).to(self.dtype).to(self.device)

        # predict impulse response
        yhat = F.lfilter(x, a, b, False)
        self.assertEqual(yhat, y, atol=1e-4, rtol=1e-5)

    def test_filtfilt_simple(self):
        """
        Check that, for an arbitrary signal, applying filtfilt with filter coefficients
        corresponding to a pure delay filter imparts no time delay.
        """
        waveform = get_whitenoise(sample_rate=8000, n_channels=2, dtype=self.dtype).to(device=self.device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        padded_waveform = torch.cat((waveform, torch.zeros(2, 3, dtype=self.dtype, device=self.device)), axis=1)
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

        waveform_k0 = get_sinusoid(frequency=5, sample_rate=samples // T, dtype=self.dtype, device=self.device).squeeze(
            0
        )
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
            output_waveform[samples - 1 : 2 * samples - 1],
            waveform_k0,
            atol=1e-3,
            rtol=1e-3,
        )

    @parameterized.expand([(0.0,), (1.0,), (2.0,), (3.0,)])
    def test_spectrogram_grad_at_zero(self, power):
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

    @parameterized.expand(
        [
            (1024,),
            (2048,),
            (4096,),
        ]
    )
    def test_spectrogram_normalization_hann_window(self, nfft):
        """This test assumes that currently, torch.stft and the existing math behind spectrogram are correct.
        The test is checking that in relation to one another, the normalization factors correctly align based on
        mathematical prediction. Using spec_false as a base, which has no normalization factors, we check to see that
        turning normalized as ``True`` or ``"window"`` will have a normalization factor of the sum of squares of hann
        window, which is calculated to be sqrt(3 * nfft / 8).
        Next, when ``normalized`` is ``"frame_length"``, we are using the normalization in torch.stft, therefore we
        assume that it is correctly normalized by a factor of sqrt(nfft). This test does not test the accuracy of
        spectrogram, but is testing the relative factors of normalization and that they align upon the frame_length
        and chosen normalize parameter.
        https://github.com/pytorch/pytorch/issues/81428
        """
        x = torch.rand(1, 22050)
        spec_false = F.spectrogram(
            x,
            pad=0,
            window=torch.hann_window(nfft, device=x.device, dtype=x.dtype),
            n_fft=nfft,
            hop_length=4,
            win_length=nfft,
            power=None,
            normalized=False,
        )

        spec_true = F.spectrogram(
            x,
            pad=0,
            window=torch.hann_window(nfft, device=x.device, dtype=x.dtype),
            n_fft=nfft,
            hop_length=4,
            win_length=nfft,
            power=None,
            normalized=True,
        )

        spec_window = F.spectrogram(
            x,
            pad=0,
            window=torch.hann_window(nfft, device=x.device, dtype=x.dtype),
            n_fft=nfft,
            hop_length=4,
            win_length=nfft,
            power=None,
            normalized="window",
        )

        spec_frame = F.spectrogram(
            x,
            pad=0,
            window=torch.hann_window(nfft, device=x.device, dtype=x.dtype),
            n_fft=nfft,
            hop_length=4,
            win_length=nfft,
            power=None,
            normalized="frame_length",
        )

        norm_factor = math.sqrt(3 * nfft / 8)
        frame_norm_factor = math.sqrt(nfft)

        self.assertEqual(spec_true, spec_window)
        self.assertEqual(spec_true, spec_false / norm_factor)
        self.assertEqual(spec_frame, spec_false / frame_norm_factor)

    def test_compute_deltas_one_channel(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=self.dtype, device=self.device)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5]]], dtype=self.dtype, device=self.device)
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)

    def test_compute_deltas_two_channels(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]], dtype=self.dtype, device=self.device)
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 0.5]]], dtype=self.dtype, device=self.device)
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)

    @parameterized.expand([(100,), (440,)])
    def test_detect_pitch_frequency_pitch(self, frequency):
        sample_rate = 44100
        test_sine_waveform = get_sinusoid(frequency=frequency, sample_rate=sample_rate, duration=5)

        freq = F.detect_pitch_frequency(test_sine_waveform, sample_rate)

        threshold = 1
        s = ((freq - frequency).abs() > threshold).sum()
        self.assertFalse(s)

    @parameterized.expand([([100, 100],), ([2, 100, 100],), ([2, 2, 100, 100],)])
    def test_amplitude_to_DB_reversible(self, shape):
        """Round trip between amplitude and db should return the original for various shape

        This implicitly also tests `DB_to_amplitude`.

        """
        amplitude_mult = 20.0
        power_mult = 10.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))

        spec = torch.rand(*shape, dtype=self.dtype, device=self.device) * 200

        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 0.5)

        self.assertEqual(x2, spec, atol=5e-5, rtol=1e-5)

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, power_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 1.0)

        self.assertEqual(x2, spec)

    @parameterized.expand([([100, 100],), ([2, 100, 100],), ([2, 2, 100, 100],)])
    def test_amplitude_to_DB_top_db_clamp(self, shape):
        """Ensure values are properly clamped when `top_db` is supplied."""
        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.0

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

        decibels = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=top_db)
        # Ensure the clamp was applied
        below_limit = decibels < 6.0205
        assert not below_limit.any(), "{} decibel values were below the expected cutoff:\n{}".format(
            below_limit.sum().item(), decibels
        )
        # Ensure it didn't over-clamp
        close_to_limit = decibels < 6.0207
        assert close_to_limit.any(), f"No values were close to the limit. Did it over-clamp?\n{decibels}"

    @parameterized.expand(list(itertools.product([(1, 201, 100), (10, 2, 201, 300)])))
    def test_mask_along_axis_input_axis_check(self, shape):
        specgram = torch.randn(*shape, dtype=self.dtype, device=self.device)
        message = "Only Frequency and Time masking are supported"
        with self.assertRaisesRegex(ValueError, message):
            F.mask_along_axis(specgram, 100, 0.0, 0, 1.0)

    @parameterized.expand(
        list(
            itertools.product([(1025, 400), (1, 201, 100), (10, 2, 201, 300)], [100], [0.0, 30.0], [1, 2], [0.33, 1.0])
        )
    )
    def test_mask_along_axis(self, shape, mask_param, mask_value, last_axis, p):
        specgram = torch.randn(*shape, dtype=self.dtype, device=self.device)

        # last_axis = 1 means the last axis; 2 means the second-to-last axis.
        axis = len(shape) - last_axis
        if p != 1.0:
            mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis, p=p)
        else:
            mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis)

        other_axis = axis - 1 if last_axis == 1 else axis + 1

        masked_columns = (mask_specgram == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgram.size(other_axis)).sum()

        den = 1
        for i in range(len(shape) - 2):
            den *= mask_specgram.size(i)

        num_masked_columns = torch.div(num_masked_columns, den, rounding_mode="floor")

        if p != 1.0:
            mask_param = min(mask_param, int(specgram.shape[axis] * p))

        assert mask_specgram.size() == specgram.size()
        assert num_masked_columns < mask_param

    @parameterized.expand(list(itertools.product([100], [0.0, 30.0], [2, 3], [0.2, 1.0])))
    def test_mask_along_axis_iid(self, mask_param, mask_value, axis, p):
        specgrams = torch.randn(4, 2, 1025, 400, dtype=self.dtype, device=self.device)

        if p != 1.0:
            mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis, p=p)
        else:
            mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

        other_axis = 2 if axis == 3 else 3

        masked_columns = (mask_specgrams == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgrams.size(other_axis)).sum(-1)

        if p != 1.0:
            mask_param = min(mask_param, int(specgrams.shape[axis] * p))

        assert mask_specgrams.size() == specgrams.size()
        assert (num_masked_columns < mask_param).sum() == num_masked_columns.numel()

    @parameterized.expand(list(itertools.product([(2, 1025, 400), (1, 201, 100)], [100], [0.0, 30.0], [1, 2])))
    def test_mask_along_axis_preserve(self, shape, mask_param, mask_value, axis):
        """mask_along_axis should not alter original input Tensor

        Test is run 5 times to bound the probability of no masking occurring to 1e-10
        See https://github.com/pytorch/audio/issues/1478
        """
        for _ in range(5):
            specgram = torch.randn(*shape, dtype=self.dtype, device=self.device)
            specgram_copy = specgram.clone()
            F.mask_along_axis(specgram, mask_param, mask_value, axis)

            self.assertEqual(specgram, specgram_copy)

    @parameterized.expand(list(itertools.product([100], [0.0, 30.0], [2, 3])))
    def test_mask_along_axis_iid_preserve(self, mask_param, mask_value, axis):
        """mask_along_axis_iid should not alter original input Tensor

        Test is run 5 times to bound the probability of no masking occurring to 1e-10
        See https://github.com/pytorch/audio/issues/1478
        """
        for _ in range(5):
            specgrams = torch.randn(4, 2, 1025, 400, dtype=self.dtype, device=self.device)
            specgrams_copy = specgrams.clone()
            F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

            self.assertEqual(specgrams, specgrams_copy)

    @parameterized.expand(
        list(
            itertools.product(
                ["sinc_interp_hann", "sinc_interp_kaiser"],
                [16000, 44100],
            )
        )
    )
    def test_resample_identity(self, resampling_method, sample_rate):
        waveform = get_whitenoise(sample_rate=sample_rate, duration=1)

        resampled = F.resample(waveform, sample_rate, sample_rate)
        self.assertEqual(waveform, resampled)

    @parameterized.expand([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_upsample_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        upsampled = F.resample(waveform, sr, sr * 2, resampling_method=resampling_method)
        assert upsampled.size(-1) == waveform.size(-1) * 2

    @parameterized.expand([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_downsample_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        downsampled = F.resample(waveform, sr, sr // 2, resampling_method=resampling_method)
        assert downsampled.size(-1) == waveform.size(-1) // 2

    @parameterized.expand([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_identity_size(self, resampling_method):
        sr = 16000
        waveform = get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        resampled = F.resample(waveform, sr, sr, resampling_method=resampling_method)
        assert resampled.size(-1) == waveform.size(-1)

    @parameterized.expand(
        list(
            itertools.product(
                ["sinc_interp_hann", "sinc_interp_kaiser"],
                list(range(1, 20)),
            )
        )
    )
    def test_resample_waveform_downsample_accuracy(self, resampling_method, i):
        self._test_resample_waveform_accuracy(down_scale_factor=i * 2, resampling_method=resampling_method)

    @parameterized.expand(
        list(
            itertools.product(
                ["sinc_interp_hann", "sinc_interp_kaiser"],
                list(range(1, 20)),
            )
        )
    )
    def test_resample_waveform_upsample_accuracy(self, resampling_method, i):
        self._test_resample_waveform_accuracy(up_scale_factor=1.0 + i / 20.0, resampling_method=resampling_method)

    @nested_params([0.5, 1.01, 1.3])
    def test_phase_vocoder_shape(self, rate):
        """Verify the output shape of phase vocoder"""
        hop_length = 256
        num_freq = 1025
        num_frames = 400
        batch_size = 2

        spec = torch.randn(batch_size, num_freq, num_frames, dtype=self.complex_dtype, device=self.device)

        phase_advance = torch.linspace(0, np.pi * hop_length, num_freq, dtype=self.dtype, device=self.device)[..., None]

        spec_stretch = F.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

        assert spec.dim() == spec_stretch.dim()
        expected_shape = torch.Size([batch_size, num_freq, int(np.ceil(num_frames / rate))])
        output_shape = spec_stretch.shape
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
            [
                [
                    "Tᕮ᙭T",
                ],
                ["world"],
                1,
            ],  # substitution
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

    @parameterized.expand(
        [
            (rnnt_utils.get_B1_T2_U3_D5_data, torch.float32, 1e-6, 1e-2),
            (rnnt_utils.get_B2_T4_U3_D3_data, torch.float32, 1e-6, 1e-2),
            (rnnt_utils.get_B1_T2_U3_D5_data, torch.float16, 1e-3, 1e-2),
            (rnnt_utils.get_B2_T4_U3_D3_data, torch.float16, 1e-3, 1e-2),
        ]
    )
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

    @parameterized.expand([(True,), (False,)])
    def test_rnnt_loss_costs_and_gradients_random_data_with_numpy_fp32(self, fused_log_softmax):
        seed = 777
        for i in range(5):
            data = rnnt_utils.get_random_data(
                fused_log_softmax=fused_log_softmax, dtype=torch.float32, device=self.device, seed=(seed + i)
            )
            ref_costs, ref_gradients = rnnt_utils.compute_with_numpy_transducer(data=data)
            self._test_costs_and_gradients(data=data, ref_costs=ref_costs, ref_gradients=ref_gradients)

    def test_rnnt_loss_nonfused_softmax(self):
        data = rnnt_utils.get_B1_T10_U3_D4_data()
        ref_costs, ref_gradients = rnnt_utils.compute_with_numpy_transducer(data=data)
        self._test_costs_and_gradients(
            data=data,
            ref_costs=ref_costs,
            ref_gradients=ref_gradients,
        )

    def test_psd(self):
        """Verify the ``F.psd`` method by the numpy implementation.
        Given the multi-channel complex-valued spectrum as the input,
        the output of ``F.psd`` should be identical to that of ``psd_numpy``.
        """
        channel = 4
        n_fft_bin = 10
        frame = 5
        specgram = np.random.random((channel, n_fft_bin, frame)) + np.random.random((channel, n_fft_bin, frame)) * 1j
        psd = beamform_utils.psd_numpy(specgram)
        psd_audio = F.psd(torch.tensor(specgram, dtype=self.complex_dtype, device=self.device))
        self.assertEqual(torch.tensor(psd, dtype=self.complex_dtype, device=self.device), psd_audio)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_psd_with_mask(self, normalize: bool):
        """Verify the ``F.psd`` method by the numpy implementation.
        Given the multi-channel complex-valued spectrum and the single-channel real-valued mask
        as the inputs, the output of ``F.psd`` should be identical to that of ``psd_numpy``.
        """
        channel = 4
        n_fft_bin = 10
        frame = 5
        specgram = np.random.random((channel, n_fft_bin, frame)) + np.random.random((channel, n_fft_bin, frame)) * 1j
        mask = np.random.random((n_fft_bin, frame))
        psd = beamform_utils.psd_numpy(specgram, mask, normalize)
        psd_audio = F.psd(
            torch.tensor(specgram, dtype=self.complex_dtype, device=self.device),
            torch.tensor(mask, dtype=self.dtype, device=self.device),
            normalize=normalize,
        )
        self.assertEqual(torch.tensor(psd, dtype=self.complex_dtype, device=self.device), psd_audio)

    def test_mvdr_weights_souden(self):
        """Verify ``F.mvdr_weights_souden`` method by numpy implementation.
        Given the PSD matrices of target speech and noise (Tensor of dimension `(..., freq, channel, channel`)
        and an integer indicating the reference channel, ``F.mvdr_weights_souden`` outputs the mvdr weights
        (Tensor of dimension `(..., freq, channel)`), which should be close to the output of
        ``mvdr_weights_souden_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = 0
        psd_s = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        beamform_weights = beamform_utils.mvdr_weights_souden_numpy(psd_s, psd_n, reference_channel)
        beamform_weights_audio = F.mvdr_weights_souden(
            torch.tensor(psd_s, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            reference_channel,
        )
        self.assertEqual(
            torch.tensor(beamform_weights, dtype=self.complex_dtype, device=self.device),
            beamform_weights_audio,
            atol=1e-3,
            rtol=1e-6,
        )

    def test_mvdr_weights_souden_with_tensor(self):
        """Verify ``F.mvdr_weights_souden`` method by numpy implementation.
        Given the PSD matrices of target speech and noise (Tensor of dimension `(..., freq, channel, channel`)
        and a one-hot Tensor indicating the reference channel, ``F.mvdr_weights_souden`` outputs the mvdr weights
        (Tensor of dimension `(..., freq, channel)`), which should be close to the output of
        ``mvdr_weights_souden_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = np.zeros(channel)
        reference_channel[0] = 1
        psd_s = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        beamform_weights = beamform_utils.mvdr_weights_souden_numpy(psd_s, psd_n, reference_channel)
        beamform_weights_audio = F.mvdr_weights_souden(
            torch.tensor(psd_s, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            torch.tensor(reference_channel, dtype=self.dtype, device=self.device),
        )
        self.assertEqual(
            torch.tensor(beamform_weights, dtype=self.complex_dtype, device=self.device),
            beamform_weights_audio,
            atol=1e-3,
            rtol=1e-6,
        )

    def test_mvdr_weights_rtf(self):
        """Verify ``F.mvdr_weights_rtf`` method by numpy implementation.
        Given the relative transfer function (RTF) of target speech (Tensor of dimension `(..., freq, channel)`),
        the PSD matrix of noise (Tensor of dimension `(..., freq, channel, channel)`), and an integer
        indicating the reference channel as inputs, ``F.mvdr_weights_rtf`` outputs the mvdr weights
        (Tensor of dimension `(..., freq, channel)`), which should be close to the output of
        ``mvdr_weights_rtf_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = 0
        rtf = np.random.random((n_fft_bin, channel)) + np.random.random((n_fft_bin, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        beamform_weights = beamform_utils.mvdr_weights_rtf_numpy(rtf, psd_n, reference_channel)
        beamform_weights_audio = F.mvdr_weights_rtf(
            torch.tensor(rtf, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            reference_channel,
        )
        self.assertEqual(
            torch.tensor(beamform_weights, dtype=self.complex_dtype, device=self.device),
            beamform_weights_audio,
            atol=1e-3,
            rtol=1e-6,
        )

    def test_mvdr_weights_rtf_with_tensor(self):
        """Verify ``F.mvdr_weights_rtf`` method by numpy implementation.
        Given the relative transfer function (RTF) of target speech (Tensor of dimension `(..., freq, channel)`),
        the PSD matrix of noise (Tensor of dimension `(..., freq, channel, channel)`), and a one-hot Tensor
        indicating the reference channel as inputs, ``F.mvdr_weights_rtf`` outputs the mvdr weights
        (Tensor of dimension `(..., freq, channel)`), which should be close to the output of
        ``mvdr_weights_rtf_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = np.zeros(channel)
        reference_channel[0] = 1
        rtf = np.random.random((n_fft_bin, channel)) + np.random.random((n_fft_bin, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        beamform_weights = beamform_utils.mvdr_weights_rtf_numpy(rtf, psd_n, reference_channel)
        beamform_weights_audio = F.mvdr_weights_rtf(
            torch.tensor(rtf, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            torch.tensor(reference_channel, dtype=self.dtype, device=self.device),
        )
        self.assertEqual(
            torch.tensor(beamform_weights, dtype=self.complex_dtype, device=self.device),
            beamform_weights_audio,
            atol=1e-3,
            rtol=1e-6,
        )

    def test_rtf_evd(self):
        """Verify ``F.rtf_evd`` method by the numpy implementation.
        Given the multi-channel complex-valued spectrum, we compute the PSD matrix as the input,
        ``F.rtf_evd`` outputs the relative transfer function (RTF) (Tensor of dimension `(..., freq, channel)`),
        which should be identical to the output of ``rtf_evd_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        specgram = np.random.random((n_fft_bin, channel)) + np.random.random((n_fft_bin, channel)) * 1j
        psd = np.einsum("fc,fd->fcd", specgram.conj(), specgram)
        rtf = beamform_utils.rtf_evd_numpy(psd)
        rtf_audio = F.rtf_evd(torch.tensor(psd, dtype=self.complex_dtype, device=self.device))
        self.assertEqual(torch.tensor(rtf, dtype=self.complex_dtype, device=self.device), rtf_audio)

    @parameterized.expand(
        [
            (1, True),
            (2, False),
            (3, True),
        ]
    )
    def test_rtf_power(self, n_iter, diagonal_loading):
        """Verify ``F.rtf_power`` method by numpy implementation.
        Given the PSD matrices of target speech and noise (Tensor of dimension `(..., freq, channel, channel`)
        an integer indicating the reference channel, and an integer for number of iterations, ``F.rtf_power``
        outputs the relative transfer function (RTF) (Tensor of dimension `(..., freq, channel)`),
        which should be identical to the output of ``rtf_power_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = 0
        psd_s = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        rtf = beamform_utils.rtf_power_numpy(psd_s, psd_n, reference_channel, n_iter, diagonal_loading)
        rtf_audio = F.rtf_power(
            torch.tensor(psd_s, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            reference_channel,
            n_iter,
            diagonal_loading=diagonal_loading,
        )
        self.assertEqual(torch.tensor(rtf, dtype=self.complex_dtype, device=self.device), rtf_audio)

    @parameterized.expand(
        [
            (1, True),
            (2, False),
            (3, True),
        ]
    )
    def test_rtf_power_with_tensor(self, n_iter, diagonal_loading):
        """Verify ``F.rtf_power`` method by numpy implementation.
        Given the PSD matrices of target speech and noise (Tensor of dimension `(..., freq, channel, channel`)
        a one-hot Tensor indicating the reference channel, and an integer for number of iterations, ``F.rtf_power``
        outputs the relative transfer function (RTF) (Tensor of dimension `(..., freq, channel)`),
        which should be identical to the output of ``rtf_power_numpy``.
        """
        n_fft_bin = 10
        channel = 4
        reference_channel = np.zeros(channel)
        reference_channel[0] = 1
        psd_s = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        psd_n = np.random.random((n_fft_bin, channel, channel)) + np.random.random((n_fft_bin, channel, channel)) * 1j
        rtf = beamform_utils.rtf_power_numpy(psd_s, psd_n, reference_channel, n_iter, diagonal_loading)
        rtf_audio = F.rtf_power(
            torch.tensor(psd_s, dtype=self.complex_dtype, device=self.device),
            torch.tensor(psd_n, dtype=self.complex_dtype, device=self.device),
            torch.tensor(reference_channel, dtype=self.dtype, device=self.device),
            n_iter,
            diagonal_loading=diagonal_loading,
        )
        self.assertEqual(torch.tensor(rtf, dtype=self.complex_dtype, device=self.device), rtf_audio)

    def test_apply_beamforming(self):
        """Verify ``F.apply_beamforming`` method by numpy implementation.
        Given the multi-channel complex-valued spectrum and complex-valued
        beamforming weights (Tensor of dimension `(..., freq, channel)`) as inputs,
        ``F.apply_beamforming`` outputs the single-channel complex-valued enhanced
        spectrum, which should be identical to the output of ``apply_beamforming_numpy``.
        """
        channel = 4
        n_fft_bin = 10
        frame = 5
        beamform_weights = np.random.random((n_fft_bin, channel)) + np.random.random((n_fft_bin, channel)) * 1j
        specgram = np.random.random((channel, n_fft_bin, frame)) + np.random.random((channel, n_fft_bin, frame)) * 1j
        specgram_enhanced = beamform_utils.apply_beamforming_numpy(beamform_weights, specgram)
        specgram_enhanced_audio = F.apply_beamforming(
            torch.tensor(beamform_weights, dtype=self.complex_dtype, device=self.device),
            torch.tensor(specgram, dtype=self.complex_dtype, device=self.device),
        )
        self.assertEqual(
            torch.tensor(specgram_enhanced, dtype=self.complex_dtype, device=self.device), specgram_enhanced_audio
        )

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_convolve_numerics(self, leading_dims, lengths, mode):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.convolve(x, y, mode=mode)

        num_signals = torch.tensor(leading_dims).prod() if leading_dims else 1
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy(), mode=mode)
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape(leading_dims + (-1,))

        self.assertEqual(expected, actual)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_fftconvolve_numerics(self, leading_dims, lengths, mode):
        """Check that fftconvolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.fftconvolve(x, y, mode=mode)

        expected = signal.fftconvolve(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=-1, mode=mode)
        expected = torch.tensor(expected)

        self.assertEqual(expected, actual)

    @nested_params(
        ["convolve", "fftconvolve"],
        [(5, 2, 3)],
        [(5, 1, 3), (1, 2, 3), (1, 1, 3)],
    )
    def test_convolve_broadcast(self, fn, x_shape, y_shape):
        """convolve works for Tensors for different shapes if they are broadcast-able"""
        # 1. Test broadcast case
        x = torch.rand(x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(y_shape, dtype=self.dtype, device=self.device)
        out1 = getattr(F, fn)(x, y)
        # 2. Test without broadcast
        y_clone = y.expand(x_shape).clone()
        assert y is not y_clone
        assert y_clone.shape == x.shape
        out2 = getattr(F, fn)(x, y_clone)
        # check that they are same
        self.assertEqual(out1, out2)

    @parameterized.expand(
        [
            # fmt: off
            # different ndim
            (0, F.convolve, (4, 3, 1, 2), (10, 4)),
            (0, F.convolve, (4, 3, 1, 2), (2, 2, 2)),
            (0, F.convolve, (1, ), (10, 4)),
            (0, F.convolve, (1, ), (2, 2, 2)),
            (0, F.fftconvolve, (4, 3, 1, 2), (10, 4)),
            (0, F.fftconvolve, (4, 3, 1, 2), (2, 2, 2)),
            (0, F.fftconvolve, (1, ), (10, 4)),
            (0, F.fftconvolve, (1, ), (2, 2, 2)),
            # non-broadcastable leading dimensions
            (1, F.convolve, (5, 2, 3), (5, 3, 3)),
            (1, F.convolve, (5, 2, 3), (5, 3, 4)),
            (1, F.convolve, (5, 2, 3), (5, 3, 5)),
            (1, F.fftconvolve, (5, 2, 3), (5, 3, 3)),
            (1, F.fftconvolve, (5, 2, 3), (5, 3, 4)),
            (1, F.fftconvolve, (5, 2, 3), (5, 3, 5)),
            # fmt: on
        ],
    )
    def test_convolve_input_dim_check(self, case, fn, x_shape, y_shape):
        """Check that convolve properly rejects inputs with incompatible dimensions."""
        x = torch.rand(*x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(*y_shape, dtype=self.dtype, device=self.device)

        message = [
            "The operands must be the same dimension",
            "Leading dimensions of x and y are not broadcastable",
        ][case]
        with self.assertRaisesRegex(ValueError, message):
            fn(x, y)

    def test_add_noise_broadcast(self):
        """Check that add_noise produces correct outputs when broadcasting input dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(5, 1, 1, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(5, 1, 3, dtype=self.dtype, device=self.device)
        snr = torch.rand(1, 1, 1, dtype=self.dtype, device=self.device) * 10
        actual = F.add_noise(waveform, noise, snr, lengths)

        noise_expanded = noise.expand(*leading_dims, L)
        snr_expanded = snr.expand(*leading_dims)
        lengths_expanded = lengths.expand(*leading_dims)
        expected = F.add_noise(waveform, noise_expanded, snr_expanded, lengths_expanded)

        self.assertEqual(expected, actual)

    @parameterized.expand(
        [((5, 2, 3), (2, 1, 1), (5, 2), (5, 2, 3)), ((2, 1), (5,), (5,), (5,)), ((3,), (5, 2, 3), (2, 1, 1), (5, 2))]
    )
    def test_add_noise_leading_dim_check(self, waveform_dims, noise_dims, lengths_dims, snr_dims):
        """Check that add_noise properly rejects inputs with different leading dimension lengths."""
        L = 51

        waveform = torch.rand(*waveform_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*noise_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*lengths_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*snr_dims, dtype=self.dtype, device=self.device) * 10

        with self.assertRaisesRegex(ValueError, "Input leading dimensions"):
            F.add_noise(waveform, noise, snr, lengths)

    def test_add_noise_length_check(self):
        """Check that add_noise properly rejects inputs that have inconsistent length dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, 50, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        with self.assertRaisesRegex(ValueError, "Length dimensions"):
            F.add_noise(waveform, noise, snr, lengths)

    def test_speed_identity(self):
        """speed of 1.0 does not alter input waveform and length"""
        leading_dims = (5, 4, 2)
        T = 1000
        waveform = torch.rand(*leading_dims, T)
        lengths = torch.randint(1, 1000, leading_dims)
        actual_waveform, actual_lengths = F.speed(waveform, orig_freq=1000, factor=1.0, lengths=lengths)
        self.assertEqual(waveform, actual_waveform)
        self.assertEqual(lengths, actual_lengths)

    @nested_params([0.8, 1.1, 1.2], [True, False])
    def test_speed_accuracy(self, factor, use_lengths):
        """sinusoidal waveform is properly compressed by factor"""
        n_to_trim = 20

        sample_rate = 1000
        freq = 2
        times = torch.arange(0, 5, 1.0 / sample_rate)
        waveform = torch.cos(2 * math.pi * freq * times).unsqueeze(0).to(self.device, self.dtype)

        if use_lengths:
            lengths = torch.tensor([waveform.size(1)])
        else:
            lengths = None

        output, output_lengths = F.speed(waveform, orig_freq=sample_rate, factor=factor, lengths=lengths)

        if use_lengths:
            self.assertEqual(output.size(1), output_lengths[0])
        else:
            self.assertEqual(None, output_lengths)

        new_times = torch.arange(0, 5 / factor, 1.0 / sample_rate)
        expected_waveform = torch.cos(2 * math.pi * freq * factor * new_times).unsqueeze(0).to(self.device, self.dtype)

        self.assertEqual(
            expected_waveform[..., n_to_trim:-n_to_trim], output[..., n_to_trim:-n_to_trim], atol=1e-1, rtol=1e-4
        )

    @nested_params(
        [(3, 2, 100), (95,)],
        [0.97, 0.9, 0.68],
    )
    def test_preemphasis(self, input_shape, coeff):
        waveform = torch.rand(*input_shape, device=self.device, dtype=self.dtype)
        actual = F.preemphasis(waveform, coeff=coeff)

        a_coeffs = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
        b_coeffs = torch.tensor([1.0, -coeff], device=self.device, dtype=self.dtype)
        expected = F.lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)
        self.assertEqual(actual, expected)

    @nested_params(
        [(3, 2, 100), (95,)],
        [0.97, 0.9, 0.68],
    )
    def test_preemphasis_deemphasis_roundtrip(self, input_shape, coeff):
        waveform = torch.rand(*input_shape, device=self.device, dtype=self.dtype)
        preemphasized = F.preemphasis(waveform, coeff=coeff)
        deemphasized = F.deemphasis(preemphasized, coeff=coeff)
        self.assertEqual(deemphasized, waveform)

    @parameterized.expand(
        [
            ([0, 1, 1, 0], [0, 1, 5, 1, 0], torch.int32),
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], torch.int32),
            ([3, 3, 3], [3, 5, 3, 5, 3], torch.int64),
            ([0, 1, 2], [0, 1, 1, 1, 2], torch.int64),
        ]
    )
    def test_forced_align(self, targets, ref_path, targets_dtype):
        emission = torch.tensor(
            [
                [0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
                [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
                [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
                [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
                [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        blank = 5
        ref_path = torch.tensor(ref_path, dtype=targets_dtype, device=self.device)
        ref_scores = torch.tensor(
            [torch.log(emission[i, ref_path[i]]).item() for i in range(emission.shape[0])],
            dtype=emission.dtype,
            device=self.device,
        )
        log_probs = torch.log(emission)
        targets = torch.tensor(targets, dtype=targets_dtype, device=self.device)
        input_lengths = torch.tensor((log_probs.shape[0]))
        target_lengths = torch.tensor((targets.shape[0]))
        hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)
        self.assertEqual(hyp_path, ref_path)
        self.assertEqual(hyp_scores, ref_scores)

    @parameterized.expand([(torch.int32,), (torch.int64,)])
    def test_forced_align_fail(self, targets_dtype):
        log_probs = torch.rand(5, 6, dtype=self.dtype, device=self.device)
        targets = torch.tensor([0, 1, 2, 3, 4, 4], dtype=targets_dtype, device=self.device)
        blank = 5
        input_lengths = torch.tensor((log_probs.shape[0]), device=self.device)
        target_lengths = torch.tensor((targets.shape[0]), device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"targets length is too long for CTC"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        targets = torch.tensor([5, 3, 3], dtype=targets_dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, r"targets Tensor shouldn't contain blank index"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        log_probs = log_probs.int()
        targets = torch.tensor([0, 1, 2, 3], dtype=targets_dtype, device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"log_probs must be float64, float32 or float16"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        log_probs = log_probs.float()
        targets = targets.float()
        with self.assertRaisesRegex(RuntimeError, r"targets must be int32 or int64 type"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        log_probs = torch.rand(3, 4, 6, dtype=self.dtype, device=self.device)
        targets = targets.int()
        with self.assertRaisesRegex(RuntimeError, r"3-D tensor is not yet supported for log_probs"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        targets = torch.randint(0, 4, (3, 4), device=self.device)
        log_probs = torch.rand(3, 6, dtype=self.dtype, device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"2-D tensor is not yet supported for targets"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        targets = torch.tensor([0, 1, 2, 3], dtype=targets_dtype, device=self.device)
        input_lengths = torch.randint(1, 5, (3,), device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"input_lengths must be 0-D"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        input_lengths = torch.tensor((log_probs.shape[0]), device=self.device)
        target_lengths = torch.randint(1, 5, (3,), device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"target_lengths must be 0-D"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        input_lengths = torch.tensor((10000), device=self.device)
        target_lengths = torch.tensor((targets.shape[0]), device=self.device)
        with self.assertRaisesRegex(RuntimeError, r"input length mismatch"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        input_lengths = torch.tensor((log_probs.shape[0]))
        target_lengths = torch.tensor((10000))
        with self.assertRaisesRegex(RuntimeError, r"target length mismatch"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        targets = torch.tensor([7, 8, 9, 10], dtype=targets_dtype, device=self.device)
        log_probs = torch.rand(10, 5, dtype=self.dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, r"targets values must be less than the CTC dimension"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)

        targets = torch.tensor([1, 3, 3], dtype=targets_dtype, device=self.device)
        blank = 10000
        with self.assertRaisesRegex(RuntimeError, r"blank must be within \[0, num classes\)"):
            hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths, blank)


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


class FunctionalCUDAOnly(TestBaseMixin):
    @nested_params(
        [torch.half, torch.float, torch.double],
        [torch.int32, torch.int64],
        [(50, 100), (100, 100)],
        [(10,), (40,), (45,)],
    )
    def test_forced_align_same_result(self, log_probs_dtype, targets_dtype, log_probs_shape, targets_shape):
        log_probs = torch.rand(log_probs_shape, dtype=log_probs_dtype, device=self.device)
        targets = torch.randint(1, 100, targets_shape, dtype=targets_dtype, device=self.device)
        input_lengths = torch.tensor((log_probs.shape[0]), device=self.device)
        target_lengths = torch.tensor((targets.shape[0]), device=self.device)
        log_probs_cuda = log_probs.cuda()
        targets_cuda = targets.cuda()
        input_lengths_cuda = input_lengths.cuda()
        target_lengths_cuda = target_lengths.cuda()
        hyp_path, hyp_scores = F.forced_align(log_probs, targets, input_lengths, target_lengths)
        hyp_path_cuda, hyp_scores_cuda = F.forced_align(
            log_probs_cuda, targets_cuda, input_lengths_cuda, target_lengths_cuda
        )
        self.assertEqual(hyp_path, hyp_path_cuda.cpu())
        self.assertEqual(hyp_scores, hyp_scores_cuda.cpu())
