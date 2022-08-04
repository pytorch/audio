import numpy as np
import torch
import torchaudio.prototype.functional as F
from scipy import signal
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class FunctionalTestImpl(TestBaseMixin):
    @nested_params(
        [F.convolve, F.fftconvolve],
        [(10, 4), (4, 3, 1, 2), (2,)],
        [(100, 43), (21, 45)],
    )
    def test_convolve_numerics(self, fn, leading_dims, lengths):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        actual = fn(x, y)

        num_signals = torch.tensor(leading_dims).prod()
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy())
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape((*leading_dims, -1))

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

    @nested_params(
        [F.convolve, F.fftconvolve],
        [(1,)],
        [(1,), (2,)],
    )
    def test_convolve_input_min_dim_check(self, fn, x_shape, y_shape):
        """Check that convolve properly rejects inputs that don't have a sufficient number of dimensions."""
        x = torch.rand(*x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(*y_shape, dtype=self.dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, "Inputs must have at least"):
            fn(x, y)
