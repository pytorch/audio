import torch
import torchaudio.prototype.functional as F
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    @nested_params(
        [F.convolve, F.fftconvolve],
    )
    def test_convolve(self, fn):
        leading_dims = (2, 3)
        L_x, L_y = 89, 43
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        actual = fn(x, y)
        expected = torch.stack(
            [
                torch.stack([fn(x[i, j].unsqueeze(0), y[i, j].unsqueeze(0)).squeeze(0) for j in range(leading_dims[1])])
                for i in range(leading_dims[0])
            ]
        )

        self.assertEqual(expected, actual)
