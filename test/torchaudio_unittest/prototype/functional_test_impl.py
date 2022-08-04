import math

import numpy as np
import torch
import torchaudio.prototype.functional as F
from scipy import signal
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class FunctionalTestImpl(TestBaseMixin):
    @nested_params(
        [False, True],
        [(10, 4), (4, 3, 1, 2), (2,)],
        [(100, 43), (21, 45)],
    )
    def test_convolve(self, use_fft, leading_dims, lengths):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        actual = F.convolve(x, y, use_fft=use_fft)

        num_signals = math.prod(leading_dims)
        x_reshaped = x.view((num_signals, L_x))
        y_reshaped = y.view((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy())
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.view((*leading_dims, -1))

        self.assertEqual(expected, actual)
