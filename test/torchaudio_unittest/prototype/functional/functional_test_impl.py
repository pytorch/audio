import numpy as np
import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from scipy import signal
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class FunctionalTestImpl(TestBaseMixin):
    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
    )
    def test_convolve_numerics(self, leading_dims, lengths):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.convolve(x, y)

        num_signals = torch.tensor(leading_dims).prod() if leading_dims else 1
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy())
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape(leading_dims + (-1,))

        self.assertEqual(expected, actual)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
    )
    def test_fftconvolve_numerics(self, leading_dims, lengths):
        """Check that fftconvolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.fftconvolve(x, y)

        expected = signal.fftconvolve(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=-1)
        expected = torch.tensor(expected)

        self.assertEqual(expected, actual)

    @nested_params(
        [F.convolve, F.fftconvolve],
        [(4, 3, 1, 2), (1,)],
        [(10, 4), (2, 2, 2)],
    )
    def test_convolve_input_leading_dim_check(self, fn, x_shape, y_shape):
        """Check that convolve properly rejects inputs with different leading dimensions."""
        x = torch.rand(*x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(*y_shape, dtype=self.dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, "Leading dimensions"):
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
