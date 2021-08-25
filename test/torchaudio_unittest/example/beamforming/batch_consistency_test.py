"""Test numerical consistency among single input and batched input."""
import torch
from beamforming.mvdr import PSD, MVDR
from parameterized import parameterized
from torchaudio_unittest import common_utils


class TestTransforms(common_utils.TorchaudioTestCase):
    def test_batch_PSD(self):
        spec = torch.rand((2, 6, 201, 100), dtype=torch.cdouble)

        # Single then transform then batch
        expected = PSD()(spec).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = PSD()(spec.repeat(3, 1, 1, 1))

        self.assertEqual(computed, expected)

    def test_batch_PSD_with_mask(self):
        spec = torch.rand((2, 6, 201, 100), dtype=torch.cdouble)
        mask = torch.rand((2, 201, 100))

        # Single then transform then batch
        expected = PSD()(spec, mask).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = PSD()(spec.repeat(3, 1, 1, 1), mask.repeat(3, 1, 1))

        self.assertEqual(computed, expected)

    @parameterized.expand([
        [True],
        [False],
    ])
    def test_MVDR(self, multi_mask):
        spec = torch.rand((2, 6, 201, 100), dtype=torch.cdouble)
        if multi_mask:
            mask = torch.rand((2, 6, 201, 100))
        else:
            mask = torch.rand((2, 201, 100))

        # Single then transform then batch
        expected = MVDR(multi_mask=multi_mask)(spec, mask).repeat(3, 1, 1)

        # Batch then transform
        if multi_mask:
            computed = MVDR(multi_mask=multi_mask)(spec.repeat(3, 1, 1, 1), mask.repeat(3, 1, 1, 1))
        else:
            computed = MVDR(multi_mask=multi_mask)(spec.repeat(3, 1, 1, 1), mask.repeat(3, 1, 1))

        self.assertEqual(computed, expected)
