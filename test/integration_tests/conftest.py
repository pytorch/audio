import torch
from torchaudio_unittest.common_utils import get_asset_path
import pytest


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def forward(self, logits: torch.Tensor) -> str:
        """Given a sequence logits over labels, get the best path string

        Args:
            logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        best_path = torch.argmax(logits, dim=-1)  # [num_seq,]
        best_path = torch.unique_consecutive(best_path, dim=-1)
        hypothesis = []
        for i in best_path:
            char = self.labels[i]
            if char not in ['<s>', '<pad>']:
                hypothesis.append(char)
        return ''.join(hypothesis)


@pytest.fixture
def ctc_decoder():
    return GreedyCTCDecoder


@pytest.fixture
def sample_speech_16000_en():
    return get_asset_path('Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.flac')
