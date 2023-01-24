import math

import torch
import torchaudio.prototype.functional as F
from parameterized import param, parameterized
from torchaudio.functional import lfilter
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin

from .dsp_utils import freq_ir as freq_ir_np, oscillator_bank as oscillator_bank_np, sinc_ir as sinc_ir_np


def _prod(l):
    r = 1
    for p in l:
        r *= p
    return r


class FunctionalTestImpl(TestBaseMixin):
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

    def test_freq_ir_warns_negative_values(self):
        """frequency_impulse_response warns negative input value"""
        magnitudes = -torch.ones((1, 30), device=self.device, dtype=self.dtype)
        with self.assertWarnsRegex(UserWarning, "^.+should not contain negative values.$"):
            F.frequency_impulse_response(magnitudes)

    @parameterized.expand([((2, 3, 4),), ((1000,),)])
    def test_freq_ir_reference(self, shape):
        """frequency_impulse_response produces the same result as reference implementation"""
        magnitudes = torch.rand(shape, device=self.device, dtype=self.dtype)

        hyp = F.frequency_impulse_response(magnitudes)
        ref = freq_ir_np(magnitudes.cpu().numpy())

        self.assertEqual(hyp, ref)

    @parameterized.expand(
        [
            # fmt: off
            # INPUT: single-dim waveform and 2D filter
            # The number of frames is divisible with the number of filters (15 % 3 == 0),
            # thus waveform must be split into chunks without padding
            ((15, ), (3, 3)),  # filter size (3) is shorter than chunk size (15 // 3 == 5)
            ((15, ), (3, 5)),  # filter size (5) matches than chunk size
            ((15, ), (3, 7)),  # filter size (7) is longer than chunk size
            # INPUT: single-dim waveform and 2D filter
            # The number of frames is NOT divisible with the number of filters (15 % 4 != 0),
            # thus waveform must be padded before padding
            ((15, ), (4, 3)),  # filter size (3) is shorter than chunk size (16 // 4 == 4)
            ((15, ), (4, 4)),  # filter size (4) is shorter than chunk size
            ((15, ), (4, 5)),  # filter size (5) is longer than chunk size
            # INPUT: multi-dim waveform and 2D filter
            # The number of frames is divisible with the number of filters (15 % 3 == 0),
            # thus waveform must be split into chunks without padding
            ((7, 2, 15), (3, 3)),
            ((7, 2, 15), (3, 5)),
            ((7, 2, 15), (3, 7)),
            # INPUT: single-dim waveform and 2D filter
            # The number of frames is NOT divisible with the number of filters (15 % 4 != 0),
            # thus waveform must be padded before padding
            ((7, 2, 15), (4, 3)),
            ((7, 2, 15), (4, 4)),
            ((7, 2, 15), (4, 5)),
            # INPUT: multi-dim waveform and multi-dim filter
            # The number of frames is divisible with the number of filters (15 % 3 == 0),
            # thus waveform must be split into chunks without padding
            ((7, 2, 15), (7, 2, 3, 3)),
            ((7, 2, 15), (7, 2, 3, 5)),
            ((7, 2, 15), (7, 2, 3, 7)),
            # INPUT: multi-dim waveform and multi-dim filter
            # The number of frames is NOT divisible with the number of filters (15 % 4 != 0),
            # thus waveform must be padded before padding
            ((7, 2, 15), (7, 2, 4, 3)),
            ((7, 2, 15), (7, 2, 4, 4)),
            ((7, 2, 15), (7, 2, 4, 5)),
            # INPUT: multi-dim waveform and (broadcast) multi-dim filter
            # The number of frames is divisible with the number of filters (15 % 3 == 0),
            # thus waveform must be split into chunks without padding
            ((7, 2, 15), (1, 1, 3, 3)),
            ((7, 2, 15), (1, 1, 3, 5)),
            ((7, 2, 15), (1, 1, 3, 7)),
            # INPUT: multi-dim waveform and (broadcast) multi-dim filter
            # The number of frames is NOT divisible with the number of filters (15 % 4 != 0),
            # thus waveform must be padded before padding
            ((7, 2, 15), (1, 1, 4, 3)),
            ((7, 2, 15), (1, 1, 4, 4)),
            ((7, 2, 15), (1, 1, 4, 5)),
            # fmt: on
        ]
    )
    def test_filter_waveform_shape(self, waveform_shape, filter_shape):
        """filter_waveform returns the waveform with the same number of samples"""
        waveform = torch.randn(waveform_shape, dtype=self.dtype, device=self.device)
        filters = torch.randn(filter_shape, dtype=self.dtype, device=self.device)

        filtered = F.filter_waveform(waveform, filters)

        assert filtered.shape == waveform.shape

    @nested_params([1, 3, 5], [3, 5, 7, 4, 6, 8])
    def test_filter_waveform_delta(self, num_filters, kernel_size):
        """Applying delta kernel preserves the origianl waveform"""
        waveform = torch.arange(-10, 10, dtype=self.dtype, device=self.device)
        kernel = torch.zeros((num_filters, kernel_size), dtype=self.dtype, device=self.device)
        kernel[:, kernel_size // 2] = 1

        result = F.filter_waveform(waveform, kernel)
        self.assertEqual(waveform, result)

    def test_filter_waveform_same(self, kernel_size=5):
        """Applying the same filter returns the original waveform"""
        waveform = torch.arange(-10, 10, dtype=self.dtype, device=self.device)
        kernel = torch.randn((1, kernel_size), dtype=self.dtype, device=self.device)
        kernels = torch.cat([kernel] * 3)

        out1 = F.filter_waveform(waveform, kernel)
        out2 = F.filter_waveform(waveform, kernels)
        self.assertEqual(out1, out2)

    def test_filter_waveform_diff(self):
        """Filters are applied from the first to the last"""
        kernel_size = 3
        waveform = torch.arange(-10, 10, dtype=self.dtype, device=self.device)
        kernels = torch.randn((2, kernel_size), dtype=self.dtype, device=self.device)

        # use both filters.
        mix = F.filter_waveform(waveform, kernels)
        # use only one of them
        ref1 = F.filter_waveform(waveform[:10], kernels[0:1])
        ref2 = F.filter_waveform(waveform[10:], kernels[1:2])

        print("mix:", mix)
        print("ref1:", ref1)
        print("ref2:", ref2)
        # The first filter is effective in the first half
        self.assertEqual(mix[:10], ref1[:10])
        # The second filter is effective in the second half
        self.assertEqual(mix[-9:], ref2[-9:])
        # the middle portion is where the two filters affect


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
