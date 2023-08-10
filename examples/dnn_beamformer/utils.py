from typing import Dict, List, Tuple

import torch
from torch import Tensor


class CollateFnL3DAS22:
    """The collate class for L3DAS22 dataset.
    Args:
        pad (bool): If ``True``, the waveforms and labels will be padded to the
            max length in the mini-batch. If ``pad`` is False, the waveforms
            and labels will be cropped to the minimum length in the mini-batch.
            (Default: False)
        rand_crop (bool): if ``True``, the starting index of the waveform
            and label is random if the length is longer than the minimum
            length in the mini-batch.
    """

    def __init__(
        self,
        audio_length: int = 16000 * 4,
        rand_crop: bool = True,
    ) -> None:
        self.audio_length = audio_length
        self.rand_crop = rand_crop

    def __call__(self, batch: List[Tuple[Tensor, Tensor, int, str]]) -> Dict:
        """
        Args:
            batch (List[Tuple(Tensor, Tensor, int)]):
                The list of tuples that contains:
                - mixture waveforms
                - clean waveform
                - sample rate
                - transcript

        Returns:
            Dictionary
                "input": Tuple of waveforms and lengths.
                    waveforms Tensor with dimensions `(batch, time)`.
                    lengths Tensor with dimension `(batch,)`.
                "label": None
        """
        waveforms_noisy, waveforms_clean = [], []
        for sample in batch:
            waveform_noisy, waveform_clean, _SAMPLE_RATE, transcript = sample
            if self.rand_crop:
                diff = waveform_noisy.size(-1) - self.audio_length
                frame_offset = torch.randint(diff, size=(1,))
            else:
                frame_offset = 0
            waveform_noisy = waveform_noisy[:, frame_offset : frame_offset + self.audio_length]
            waveform_clean = waveform_clean[:, frame_offset : frame_offset + self.audio_length]
            waveforms_noisy.append(waveform_noisy.unsqueeze(0))
            waveforms_clean.append(waveform_clean)
        waveforms_noisy = torch.cat(waveforms_noisy)
        waveforms_clean = torch.cat(waveforms_clean)
        return waveforms_noisy, waveforms_clean
