"""Test numerical consistency among single input and batched input."""
import torch
from beamforming.mvdr import PSD, MVDR
from parameterized import parameterized

from torchaudio_unittest import common_utils


class TestTransforms(common_utils.TorchaudioTestCase):
    def test_batch_PSD(self):
        spec = torch.rand((4, 6, 201, 100), dtype=torch.cdouble)

        # Single then transform then batch
        expected = []
        for i in range(4):
            expected.append(PSD()(spec[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = PSD()(spec)

        self.assertEqual(computed, expected)

    def test_batch_PSD_with_mask(self):
        spec = torch.rand((4, 6, 201, 100), dtype=torch.cdouble)
        mask = torch.rand((4, 201, 100))

        # Single then transform then batch
        expected = []
        for i in range(4):
            expected.append(PSD()(spec[i], mask[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = PSD()(spec, mask)

        self.assertEqual(computed, expected)

    @parameterized.expand([
        [True],
        [False],
    ])
    def test_MVDR(self, multi_mask):
        spec = torch.rand((4, 6, 201, 100), dtype=torch.cdouble)
        if multi_mask:
            mask = torch.rand((4, 6, 201, 100))
        else:
            mask = torch.rand((4, 201, 100))

        # Single then transform then batch
        expected = []
        for i in range(4):
            expected.append(MVDR(multi_mask=multi_mask)(spec[i], mask[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = MVDR(multi_mask=multi_mask)(spec, mask)

        self.assertEqual(computed, expected)
