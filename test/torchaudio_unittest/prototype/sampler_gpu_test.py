import torch
from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase
from torchaudio_unittest.prototype.sampler_test_impl import TestBucketizeBatchSampler


@skipIfNoCuda
class SamplerGPUTest(TestBucketizeBatchSampler, PytorchTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
