import torch
import torchaudio.prototype.transforms as T
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    @nested_params(
        [T.Convolve, T.FFTConvolve],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (2, 3)
        L_x, L_y = 89, 43
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        convolve = cls(mode=mode)
        actual = convolve(x, y)
        expected = torch.stack(
            [
                torch.stack(
                    [convolve(x[i, j].unsqueeze(0), y[i, j].unsqueeze(0)).squeeze(0) for j in range(leading_dims[1])]
                )
                for i in range(leading_dims[0])
            ]
        )

        self.assertEqual(expected, actual)
