import torch
import torchaudio.prototype.functional as F
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (2, 3)
        L_x, L_y = 89, 43
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        actual = fn(x, y, mode)
        expected = torch.stack(
            [
                torch.stack(
                    [fn(x[i, j].unsqueeze(0), y[i, j].unsqueeze(0), mode).squeeze(0) for j in range(leading_dims[1])]
                )
                for i in range(leading_dims[0])
            ]
        )

        self.assertEqual(expected, actual)

    def test_add_noise(self):
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        actual = F.add_noise(waveform, noise, lengths, snr)

        expected = []
        for i in range(leading_dims[0]):
            for j in range(leading_dims[1]):
                for k in range(leading_dims[2]):
                    expected.append(F.add_noise(waveform[i][j][k], noise[i][j][k], lengths[i][j][k], snr[i][j][k]))

        self.assertEqual(torch.stack(expected), actual.reshape(-1, L))
