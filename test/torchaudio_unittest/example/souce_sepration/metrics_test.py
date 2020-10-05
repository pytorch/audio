from itertools import product

import torch
from torch.testing._internal.common_utils import TestCase
from parameterized import parameterized

from . import sdr_reference
from utils import metrics


class TestSDR(TestCase):
    @parameterized.expand([(1, ), (2, ), (32, )])
    def test_sdr(self, batch_size):
        """sdr produces the same result as the reference implementation"""
        num_frames = 256

        estimation = torch.rand(batch_size, num_frames)
        origin = torch.rand(batch_size, num_frames)

        sdr_ref = sdr_reference.calc_sdr_torch(estimation, origin)
        sdr = metrics.sdr(estimation.unsqueeze(1), origin.unsqueeze(1)).squeeze(1)

        self.assertEqual(sdr, sdr_ref)

    @parameterized.expand(list(product([1, 2, 32], [2, 3, 4, 5])))
    def test_sdr_pit(self, batch_size, num_sources):
        """sdr_pit produces the same result as the reference implementation"""
        num_frames = 256

        estimation = torch.randn(batch_size, num_sources, num_frames)
        origin = torch.randn(batch_size, num_sources, num_frames)

        estimation -= estimation.mean(axis=2, keepdim=True)
        origin -= origin.mean(axis=2, keepdim=True)

        batch_sdr_ref = sdr_reference.batch_SDR_torch(estimation, origin)
        batch_sdr = metrics.sdr_pit(estimation, origin)

        self.assertEqual(batch_sdr, batch_sdr_ref)
