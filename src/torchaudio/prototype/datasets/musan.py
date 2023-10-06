from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform


_SUBSETS = ["music", "noise", "speech"]
_SAMPLE_RATE = 16_000


class Musan(Dataset):
    r"""*MUSAN* :cite:`musan2015` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top-level directory exists.
        subset (str): Subset of the dataset to use. Options: [``"music"``, ``"noise"``, ``"speech"``].
    """

    def __init__(self, root: Union[str, Path], subset: str):
        if subset not in _SUBSETS:
            raise ValueError(f"Invalid subset '{subset}' given. Please provide one of {_SUBSETS}")

        subset_path = Path(root) / subset
        self._walker = [str(p) for p in subset_path.glob("*/*.*")]

    def get_metadata(self, n: int) -> Tuple[str, int, str]:
        r"""Get metadata for the n-th sample in the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): Index of sample to be loaded.

        Returns:
            (str, int, str):
                str
                    Path to audio.
                int
                    Sample rate.
                str
                    File name.
        """
        audio_path = self._walker[n]
        return audio_path, _SAMPLE_RATE, Path(audio_path).name

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        r"""Return the n-th sample in the dataset.

        Args:
            n (int): Index of sample to be loaded.

        Returns:
            (torch.Tensor, int, str):
                torch.Tensor
                    Waveform.
                int
                    Sample rate.
                str
                    File name.
        """
        audio_path, sample_rate, filename = self.get_metadata(n)
        path = Path(audio_path)
        return _load_waveform(path.parent, path.name, sample_rate), sample_rate, filename

    def __len__(self) -> int:
        return len(self._walker)
