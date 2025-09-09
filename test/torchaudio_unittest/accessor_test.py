import torch
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE

if _IS_TORCHAUDIO_EXT_AVAILABLE:
    def test_accessor():
        tensor = torch.randint(1000, (5,4,3))
        assert torch.ops.torchaudio._test_accessor(tensor)
