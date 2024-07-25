import torch
import torch.nn as nn


class NormalizeDB(nn.Module):
    r"""Normalize the spectrogram with a minimum db value"""

    def __init__(self, min_level_db, normalization):
        super().__init__()
        self.min_level_db = min_level_db
        self.normalization = normalization

    def forward(self, specgram):
        specgram = torch.log10(torch.clamp(specgram.squeeze(0), min=1e-5))
        if self.normalization:
            return torch.clamp((self.min_level_db - 20 * specgram) / self.min_level_db, min=0, max=1)
        return specgram


def normalized_waveform_to_bits(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    r"""Transform waveform [-1, 1] to label [0, 2 ** bits - 1]"""

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2**bits - 1) / 2
    return torch.clamp(waveform, 0, 2**bits - 1).int()


def bits_to_normalized_waveform(label: torch.Tensor, bits: int) -> torch.Tensor:
    r"""Transform label [0, 2 ** bits - 1] to waveform [-1, 1]"""

    return 2 * label / (2**bits - 1.0) - 1.0
