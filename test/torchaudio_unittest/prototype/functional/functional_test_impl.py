import math

import numpy as np
import torch
import torchaudio.prototype.functional as F
from parameterized import param, parameterized
from scipy import signal
from torchaudio._internal import module_utils as _mod_utils
from torchaudio.functional import lfilter
from torchaudio_unittest.common_utils import nested_params, skipIfNoModule, TestBaseMixin

from .dsp_utils import oscillator_bank as oscillator_bank_np, sinc_ir as sinc_ir_np

if _mod_utils.is_module_available("pyroomacoustics"):
    import pyroomacoustics as pra


def _prod(l):
    r = 1
    for p in l:
        r *= p
    return r


class FunctionalTestImpl(TestBaseMixin):
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

    @parameterized.expand(
        [
            # fmt: off
            ((5, 2, 3), (5, 1, 3)),
            ((5, 2, 3), (1, 2, 3)),
            ((5, 2, 3), (1, 1, 3)),
            # fmt: on
        ]
    )
    def test_fftconvolve_broadcast(self, x_shape, y_shape):
        """fftconvolve works for Tensors for different shapes if they are broadcast-able"""
        # 1. Test broad cast case
        x = torch.rand(x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(y_shape, dtype=self.dtype, device=self.device)
        out1 = F.fftconvolve(x, y)
        # 2. Test without broadcast
        y_clone = y.expand(x_shape).clone()
        assert y is not y_clone
        assert y_clone.shape == x.shape
        out2 = F.fftconvolve(x, y_clone)
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
            # incompatible shape except the last dim
            (1, F.convolve, (5, 2, 3), (5, 3, 3)),
            (1, F.convolve, (5, 2, 3), (5, 3, 4)),
            (1, F.convolve, (5, 2, 3), (5, 3, 5)),
            (2, F.fftconvolve, (5, 2, 3), (5, 3, 3)),
            (2, F.fftconvolve, (5, 2, 3), (5, 3, 4)),
            (2, F.fftconvolve, (5, 2, 3), (5, 3, 5)),
            # broadcast-able (only for convolve)
            (1, F.convolve, (5, 2, 3), (5, 1, 3)),
            (1, F.convolve, (5, 2, 3), (5, 1, 4)),
            (1, F.convolve, (5, 2, 3), (5, 1, 5)),
            # fmt: on
        ],
    )
    def test_convolve_input_leading_dim_check(self, case, fn, x_shape, y_shape):
        """Check that convolve properly rejects inputs with different leading dimensions."""
        x = torch.rand(*x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(*y_shape, dtype=self.dtype, device=self.device)

        message = [
            "The operands must be the same dimension",
            "Leading dimensions of x and y don't match",
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
        actual = F.add_noise(waveform, noise, lengths, snr)

        noise_expanded = noise.expand(*leading_dims, L)
        snr_expanded = snr.expand(*leading_dims)
        lengths_expanded = lengths.expand(*leading_dims)
        expected = F.add_noise(waveform, noise_expanded, lengths_expanded, snr_expanded)

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
            F.add_noise(waveform, noise, lengths, snr)

    def test_add_noise_length_check(self):
        """Check that add_noise properly rejects inputs that have inconsistent length dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, 50, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        with self.assertRaisesRegex(ValueError, "Length dimensions"):
            F.add_noise(waveform, noise, lengths, snr)

    @nested_params(
        [(2, 3), (2, 3, 5), (2, 3, 5, 7)],
        ["sum", "mean", "none"],
    )
    def test_oscillator_bank_smoke_test(self, shape, reduction):
        """oscillator_bank supports variable dimension inputs on different device/dtypes"""
        sample_rate = 8000

        freqs = sample_rate // 2 * torch.rand(shape, dtype=self.dtype, device=self.device)
        amps = torch.rand(shape, dtype=self.dtype, device=self.device)

        waveform = F.oscillator_bank(freqs, amps, sample_rate, reduction=reduction)
        expected_shape = shape if reduction == "none" else shape[:-1]
        assert waveform.shape == expected_shape
        assert waveform.dtype == self.dtype
        assert waveform.device == self.device

    def test_oscillator_invalid(self):
        """oscillator_bank rejects/warns invalid inputs"""
        valid_shape = [2, 3, 5]
        sample_rate = 8000

        freqs = torch.ones(*valid_shape, dtype=self.dtype, device=self.device)
        amps = torch.rand(*valid_shape, dtype=self.dtype, device=self.device)

        # mismatching shapes
        with self.assertRaises(ValueError):
            F.oscillator_bank(freqs[0], amps, sample_rate)

        # frequencies out of range
        nyquist = sample_rate / 2
        with self.assertWarnsRegex(UserWarning, r"above nyquist frequency"):
            F.oscillator_bank(nyquist * freqs, amps, sample_rate)

        with self.assertWarnsRegex(UserWarning, r"above nyquist frequency"):
            F.oscillator_bank(-nyquist * freqs, amps, sample_rate)

    @parameterized.expand(
        [
            # Attack (full)
            param(
                num_frames=11,
                expected=[i / 10 for i in range(11)],
                attack=1.0,
            ),
            # Attack (partial)
            param(
                num_frames=11,
                expected=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 0, 0, 0, 0, 0],
                attack=0.5,
            ),
            # Hold (partial with attack)
            param(
                num_frames=11,
                expected=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                attack=0.5,
                hold=0.5,
            ),
            # Hold (partial without attack)
            param(
                num_frames=11,
                expected=[1.0] * 6 + [0.0] * 5,
                hold=0.5,
            ),
            # Hold (full)
            param(
                num_frames=11,
                expected=[1.0] * 11,
                hold=1.0,
            ),
            # Decay (partial - linear, preceded by attack)
            param(
                num_frames=11,
                expected=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0],
                attack=0.5,
                decay=0.5,
                n_decay=1,
            ),
            # Decay (partial - linear, preceded by hold)
            param(
                num_frames=11,
                expected=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0],
                hold=0.5,
                decay=0.5,
                n_decay=1,
            ),
            # Decay (partial - linear)
            param(
                num_frames=11,
                expected=[1.0, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0, 0],
                decay=0.5,
                n_decay=1,
            ),
            # Decay (partial - polynomial)
            param(
                num_frames=11,
                expected=[1.0, 0.64, 0.36, 0.16, 0.04, 0, 0, 0, 0, 0, 0],
                decay=0.5,
                n_decay=2,
            ),
            # Decay (full - linear)
            param(
                num_frames=11,
                expected=[1.0 - i / 10 for i in range(11)],
                decay=1.0,
                n_decay=1,
            ),
            # Decay (full - polynomial)
            param(
                num_frames=11,
                expected=[(1.0 - i / 10) ** 2 for i in range(11)],
                decay=1.0,
                n_decay=2,
            ),
            # Sustain (partial - preceded by decay)
            param(
                num_frames=11,
                expected=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                decay=0.5,
                sustain=0.5,
                n_decay=1,
            ),
            # Sustain (partial - preceded by decay)
            param(
                num_frames=11,
                expected=[1.0, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                decay=0.3,
                sustain=0.4,
                n_decay=1,
            ),
            # Sustain (full)
            param(
                num_frames=11,
                expected=[0.3] * 11,
                sustain=0.3,
            ),
            # Release (partial - preceded by decay)
            param(
                num_frames=11,
                expected=[1.0, 0.84, 0.68, 0.52, 0.36, 0.2, 0.16, 0.12, 0.08, 0.04, 0.0],
                decay=0.5,
                sustain=0.2,
                release=0.5,
                n_decay=1,
            ),
            # Release (partial - preceded by sustain)
            param(
                num_frames=11,
                expected=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                sustain=0.5,
                release=0.5,
            ),
            # Release (full)
            param(
                num_frames=11,
                expected=[1 - i / 10 for i in range(11)],
                sustain=1.0,
                release=1.0,
            ),
        ]
    )
    def test_adsr_envelope(
        self, num_frames, expected, attack=0.0, hold=0.0, decay=0.0, sustain=0.0, release=0.0, n_decay=2.0
    ):
        """the distribution of time are correct"""
        out = F.adsr_envelope(
            num_frames,
            attack=attack,
            hold=hold,
            decay=decay,
            sustain=sustain,
            release=release,
            n_decay=n_decay,
            device=self.device,
            dtype=self.dtype,
        )
        self.assertEqual(out, torch.tensor(expected, device=self.device, dtype=self.dtype))

    def test_extend_pitch(self):
        num_frames = 5
        input = torch.ones((num_frames, 1), device=self.device, dtype=self.dtype)

        num_pitches = 7
        pattern = [i + 1 for i in range(num_pitches)]
        expected = torch.tensor([pattern] * num_frames).to(dtype=self.dtype, device=self.device)

        # passing int will append harmonic tones
        output = F.extend_pitch(input, num_pitches)
        self.assertEqual(output, expected)

        # Same can be done with passing the list of multipliers
        output = F.extend_pitch(input, pattern)
        self.assertEqual(output, expected)

        # or with tensor
        pat = torch.tensor(pattern).to(dtype=self.dtype, device=self.device)
        output = F.extend_pitch(input, pat)
        self.assertEqual(output, expected)

    @nested_params(
        # fmt: off
        [(1,), (10,), (2, 5), (3, 5, 7)],
        [1, 3, 65, 129, 257, 513, 1025],
        [True, False],
        # fmt: on
    )
    def test_sinc_ir_shape(self, input_shape, window_size, high_pass):
        """The shape of sinc_impulse_response is correct"""
        numel = _prod(input_shape)
        cutoff = torch.linspace(1, numel, numel).reshape(input_shape)
        cutoff = cutoff.to(dtype=self.dtype, device=self.device)

        filt = F.sinc_impulse_response(cutoff, window_size, high_pass)
        assert filt.shape == input_shape + (window_size,)

    @nested_params([True, False])
    def test_sinc_ir_size(self, high_pass):
        """Increasing window size expand the filter at the ends. Core parts must stay same"""
        cutoff = torch.tensor([200, 300, 400, 500, 600, 700])
        cutoff = cutoff.to(dtype=self.dtype, device=self.device)

        filt_5 = F.sinc_impulse_response(cutoff, 5, high_pass)
        filt_3 = F.sinc_impulse_response(cutoff, 3, high_pass)

        self.assertEqual(filt_3, filt_5[..., 1:-1])

    @nested_params(
        # fmt: off
        [0, 0.1, 0.5, 0.9, 1.0],
        [1, 3, 5, 65, 129, 257, 513, 1025, 2049],
        [False, True],
        # fmt: on
    )
    def test_sinc_ir_reference(self, cutoff, window_size, high_pass):
        """sinc_impulse_response produces the same result as reference implementation"""
        cutoff = torch.tensor([cutoff], device=self.device, dtype=self.dtype)

        hyp = F.sinc_impulse_response(cutoff, window_size, high_pass)
        ref = sinc_ir_np(cutoff.cpu().numpy(), window_size, high_pass)

        self.assertEqual(hyp, ref)

    def test_speed_identity(self):
        """speed of 1.0 does not alter input waveform and length"""
        leading_dims = (5, 4, 2)
        T = 1000
        waveform = torch.rand(*leading_dims, T)
        lengths = torch.randint(1, 1000, leading_dims)
        actual_waveform, actual_lengths = F.speed(waveform, lengths, orig_freq=1000, factor=1.0)
        self.assertEqual(waveform, actual_waveform)
        self.assertEqual(lengths, actual_lengths)

    @nested_params(
        [0.8, 1.1, 1.2],
    )
    def test_speed_accuracy(self, factor):
        """sinusoidal waveform is properly compressed by factor"""
        n_to_trim = 20

        sample_rate = 1000
        freq = 2
        times = torch.arange(0, 5, 1.0 / sample_rate)
        waveform = torch.cos(2 * math.pi * freq * times).unsqueeze(0).to(self.device, self.dtype)
        lengths = torch.tensor([waveform.size(1)])

        output, output_lengths = F.speed(waveform, lengths, orig_freq=sample_rate, factor=factor)
        self.assertEqual(output.size(1), output_lengths[0])

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
        expected = lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)
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


class Functional64OnlyTestImpl(TestBaseMixin):
    @nested_params(
        [1, 10, 100, 1000],
        [1, 2, 4, 8],
        [8000, 16000],
    )
    def test_oscillator_ref(self, f0, num_pitches, sample_rate):
        """oscillator_bank returns the matching values as reference implementation

        Note: It looks like NumPy performs cumsum on higher precision and thus this test
        does not pass on float32.
        """
        duration = 4.0

        num_frames = int(sample_rate * duration)
        freq0 = f0 * torch.arange(1, num_pitches + 1, device=self.device, dtype=self.dtype)
        amps = 1.0 / num_pitches * torch.ones_like(freq0)

        ones = torch.ones([num_frames, num_pitches], device=self.device, dtype=self.dtype)
        freq = ones * freq0[None, :]
        amps = ones * amps[None, :]

        wavs_ref = oscillator_bank_np(freq.cpu().numpy(), amps.cpu().numpy(), sample_rate)
        wavs_hyp = F.oscillator_bank(freq, amps, sample_rate, reduction="none")

        # Debug code to see what goes wrong.
        # keeping it for future reference
        def _debug_plot():
            """
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(num_pitches, 3, sharex=True, sharey=True)
            for p in range(num_pitches):
                (ax0, ax1, ax2) = axes[p] if num_pitches > 1 else axes
                spec_ref, ys, xs, _ = ax0.specgram(wavs_ref[:, p])
                spec_hyp, _, _, _ = ax1.specgram(wavs_hyp[:, p])
                spec_diff = spec_ref - spec_hyp
                ax2.imshow(spec_diff, aspect="auto", extent=[xs[0], xs[-1], ys[0], ys[-1]])
            plt.show()
            """
            pass

        try:
            self.assertEqual(wavs_hyp, wavs_ref)
        except AssertionError:
            _debug_plot()
            raise


class FunctionalCPUOnlyTestImpl(TestBaseMixin):
    @parameterized.expand(
        [
            (0.1, 0.2, (2, 1, 2500)),  # both float
            # Per-wall
            (torch.rand(4), 0.2, (2, 1, 2500)),
            (0.1, torch.rand(4), (2, 1, 2500)),
            (torch.rand(4), torch.rand(4), (2, 1, 2500)),
            # Per-band and per-wall
            (torch.rand(6, 4), 0.2, (2, 6, 2500)),
            (0.1, torch.rand(6, 4), (2, 6, 2500)),
            (torch.rand(6, 4), torch.rand(6, 4), (2, 6, 2500)),
        ]
    )
    def test_ray_tracing_output_shape(self, absorption, scattering, expected_shape):
        room_dim = torch.tensor([20, 25], dtype=self.dtype)
        mic_array = torch.tensor([[2, 2], [8, 8]], dtype=self.dtype)
        source = torch.tensor([7, 6], dtype=self.dtype)
        num_rays = 100

        hist = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
        )

        assert hist.shape == expected_shape

    def test_ray_tracing_input_errors(self):
        with self.assertRaisesRegex(ValueError, "room must be a 1D tensor"):
            F.ray_tracing(
                room=torch.tensor([[4, 5]]), source=torch.tensor([0, 0]), mic_array=torch.tensor([[3, 4]]), num_rays=10
            )
        with self.assertRaisesRegex(ValueError, "room must be a 1D tensor"):
            F.ray_tracing(
                room=torch.tensor([4, 5, 4, 5]),
                source=torch.tensor([0, 0]),
                mic_array=torch.tensor([[3, 4]]),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, r"mic_array must be 1D tensor of shape \(D,\), or 2D tensor"):
            F.ray_tracing(
                room=torch.tensor([4, 5]), source=torch.tensor([0, 0]), mic_array=torch.tensor([[[3, 4]]]), num_rays=10
            )
        with self.assertRaisesRegex(ValueError, "room must be of float32 or float64 dtype"):
            F.ray_tracing(
                room=torch.tensor([4, 5]).to(torch.int),
                source=torch.tensor([0, 0]),
                mic_array=torch.tensor([3, 4]),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, "dtype of room, source and mic_array must be the same"):
            F.ray_tracing(
                room=torch.tensor([4, 5]).to(torch.float64),
                source=torch.tensor([0, 0]).to(torch.float32),
                mic_array=torch.tensor([3, 4]),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, "Room dimension D must match with source and mic_array"):
            F.ray_tracing(
                room=torch.tensor([4, 5, 10], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, "Room dimension D must match with source and mic_array"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, "Room dimension D must match with source and mic_array"):
            F.ray_tracing(
                room=torch.tensor([4, 5, 10], dtype=torch.float),
                source=torch.tensor([0, 0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
            )
        with self.assertRaisesRegex(ValueError, "time_thres=10 must be at least greater than hist_bin_size=11"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                time_thres=10,
                hist_bin_size=11,
            )
        with self.assertRaisesRegex(ValueError, "The shape of absorption must be"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                absorption=torch.rand(5, dtype=torch.float),
            )
        with self.assertRaisesRegex(ValueError, "The shape of scattering must be"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                scattering=torch.rand(5, 5, dtype=torch.float),
            )
        with self.assertRaisesRegex(ValueError, "The shape of absorption must be"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                absorption=torch.rand(5, 5, dtype=torch.float),
            )
        with self.assertRaisesRegex(ValueError, "The shape of scattering must be"):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                scattering=torch.rand(5, dtype=torch.float),
            )
        with self.assertRaisesRegex(
            ValueError, "absorption and scattering must have the same number of bands and walls"
        ):
            F.ray_tracing(
                room=torch.tensor([4, 5], dtype=torch.float),
                source=torch.tensor([0, 0], dtype=torch.float),
                mic_array=torch.tensor([3, 4], dtype=torch.float),
                num_rays=10,
                absorption=torch.rand(6, 4, dtype=torch.float),
                scattering=torch.rand(5, 4, dtype=torch.float),
            )

        # Make sure passing different shapes for absorption or scattering doesn't raise an error
        # float and tensor
        F.ray_tracing(
            room=torch.tensor([4, 5], dtype=torch.float),
            source=torch.tensor([0, 0], dtype=torch.float),
            mic_array=torch.tensor([3, 4], dtype=torch.float),
            num_rays=10,
            absorption=0.1,
            scattering=torch.rand(5, 4, dtype=torch.float),
        )
        F.ray_tracing(
            room=torch.tensor([4, 5], dtype=torch.float),
            source=torch.tensor([0, 0], dtype=torch.float),
            mic_array=torch.tensor([3, 4], dtype=torch.float),
            num_rays=10,
            absorption=torch.rand(5, 4, dtype=torch.float),
            scattering=0.1,
        )
        # per-wall only and per-band + per-wall
        F.ray_tracing(
            room=torch.tensor([4, 5], dtype=torch.float),
            source=torch.tensor([0, 0], dtype=torch.float),
            mic_array=torch.tensor([3, 4], dtype=torch.float),
            num_rays=10,
            absorption=torch.rand(4, dtype=torch.float),
            scattering=torch.rand(6, 4, dtype=torch.float),
        )
        F.ray_tracing(
            room=torch.tensor([4, 5], dtype=torch.float),
            source=torch.tensor([0, 0], dtype=torch.float),
            mic_array=torch.tensor([3, 4], dtype=torch.float),
            num_rays=10,
            absorption=torch.rand(6, 4, dtype=torch.float),
            scattering=torch.rand(4, dtype=torch.float),
        )

    def test_ray_tracing_per_band_per_wall_absorption(self):
        """Check that when the value of absorption and scattering are the same
        across walls and frequency bands, the output histograms are:
        - all equal across frequency bands
        - equal to simply passing a float value instead of a (num_bands, D) or
        (D,) tensor.
        """

        room_dim = torch.tensor([20, 25], dtype=self.dtype)
        mic_array = torch.tensor([[2, 2], [8, 8]], dtype=self.dtype)
        source = torch.tensor([7, 6], dtype=self.dtype)
        num_rays = 1_000
        ABS, SCAT = 0.1, 0.2

        absorption = torch.full(fill_value=ABS, size=(6, 4), dtype=self.dtype)
        scattering = torch.full(fill_value=SCAT, size=(6, 4), dtype=self.dtype)
        hist_per_band_per_wall = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
        )
        absorption = torch.full(fill_value=ABS, size=(4,), dtype=self.dtype)
        scattering = torch.full(fill_value=SCAT, size=(4,), dtype=self.dtype)
        hist_per_wall = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
        )

        absorption = ABS
        scattering = SCAT
        hist_single = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
        )
        assert hist_per_band_per_wall.shape == (2, 6, 2500)
        assert hist_per_wall.shape == (2, 1, 2500)
        assert hist_single.shape == (2, 1, 2500)
        torch.testing.assert_close(hist_single, hist_per_wall)

        hist_single = hist_single.expand(2, 6, 2500)
        torch.testing.assert_close(hist_single, hist_per_band_per_wall)

    @skipIfNoModule("pyroomacoustics")
    @parameterized.expand(
        [
            ([20, 25], [2, 2], [[8, 8], [7, 6]], 10_000),  # 2D with 2 mics
            ([20, 25, 30], [1, 10, 5], [[8, 8, 22]], 1_000),  # 3D with 1 mic
        ]
    )
    def test_ray_tracing_same_results_as_pyroomacoustics(self, room_dim, source, mic_array, num_rays):

        walls = ["west", "east", "south", "north"]
        if len(room_dim) == 3:
            walls += ["floor", "ceiling"]
        num_walls = len(walls)
        num_bands = 6  # Note: in ray tracing, we don't need to restrict the number of bands to 7

        absorption = torch.rand(num_bands, num_walls, dtype=self.dtype)
        scattering = torch.rand(num_bands, num_walls, dtype=self.dtype)
        energy_thres = 1e-7
        time_thres = 10.0
        hist_bin_size = 0.004
        mic_radius = 0.5
        sound_speed = 343.0

        room_dim = torch.tensor(room_dim, dtype=self.dtype)
        source = torch.tensor(source, dtype=self.dtype)
        mic_array = torch.tensor(mic_array, dtype=self.dtype)

        room = pra.ShoeBox(
            room_dim.tolist(),
            ray_tracing=True,
            materials={
                walls[i]: pra.Material(
                    energy_absorption={
                        "coeffs": absorption[:, i].reshape(-1).detach().numpy(),
                        "center_freqs": 125 * 2 ** np.arange(num_bands),
                    },
                    scattering={
                        "coeffs": scattering[:, i].reshape(-1).detach().numpy(),
                        "center_freqs": 125 * 2 ** np.arange(num_bands),
                    },
                )
                for i in range(num_walls)
            },
            air_absorption=False,
            max_order=0,  # Make sure PRA doesn't use the hybrid method (we just want ray tracing)
        )
        room.add_microphone_array(mic_array.T.tolist())
        room.add_source(source.tolist())
        room.set_ray_tracing(
            n_rays=num_rays,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
            receiver_radius=mic_radius,
        )
        room.set_sound_speed(sound_speed)

        room.compute_rir()
        hist_pra = torch.tensor(np.array(room.rt_histograms))[:, 0, 0]

        hist = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
            sound_speed=sound_speed,
            mic_radius=mic_radius,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
        )

        assert hist.ndim == 3
        assert hist.shape == hist_pra.shape
        self.assertEqual(hist.to(torch.float32), hist_pra)
