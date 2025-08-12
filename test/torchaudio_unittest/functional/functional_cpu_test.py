import unittest

import torch
import torchaudio.functional as F
from parameterized import parameterized
import unittest
from torchaudio_unittest.common_utils import PytorchTestCase

from .functional_impl import Functional, FunctionalCPUOnly


class TestFunctionalFloat32(Functional, FunctionalCPUOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")

    @unittest.expectedFailure
    def test_lfilter_9th_order_filter_stability(self):
        super().test_lfilter_9th_order_filter_stability()


class TestFunctionalFloat64(Functional, PytorchTestCase):
    dtype = torch.float64
    device = torch.device("cpu")
