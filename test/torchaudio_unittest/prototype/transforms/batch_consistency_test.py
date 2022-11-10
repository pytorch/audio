import os

import torch
import torchaudio.transforms as transforms
import torchaudio.prototype.transforms as T
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    def assert_batch_consistency(self, transform, batch, *args, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        n = batch.size(0)

        # Compute items separately, then batch the result
        torch.random.manual_seed(seed)
        items_input = batch.clone()
        items_result = torch.stack([transform(items_input[i], *args, **kwargs) for i in range(n)])

        # Batch the input and run
        torch.random.manual_seed(seed)
        batch_input = batch.clone()
        batch_result = transform(batch_input, *args, **kwargs)

        self.assertEqual(items_input, batch_input, rtol=rtol, atol=atol)
        self.assertEqual(items_result, batch_result, rtol=rtol, atol=atol)

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

    def test_batch_BarkScale(self):
        specgram = torch.randn(3, 2, 201, 256)

        atol = 1e-4 if os.name == "nt" else 1e-6
        transform = T.BarkScale()

        self.assert_batch_consistency(transform, specgram, atol=atol)

    def test_batch_InverseBarkScale(self):
        n_barks = 32
        n_stft = 5
        bark_spec = torch.randn(3, 2, n_barks, 32) ** 2
        transform = transforms.InverseMelScale(n_stft, n_barks)

        # Because InverseBarkScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assert_batch_consistency(transform, bark_spec, atol=1.0, rtol=1e-5)
