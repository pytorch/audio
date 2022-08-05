import numpy as np
import torch
import torchaudio.prototype.functional as F
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
