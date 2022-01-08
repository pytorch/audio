import torch
from torchaudio_unittest.common_utils import PytorchTestCase
from torchaudio_unittest.prototype.sampler_test_impl import TestBucketizeBatchSampler


class SamplerCPUTest(TestBucketizeBatchSampler, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
