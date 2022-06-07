import os
from pathlib import Path
from typing import Tuple, Union, List, Optional

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

_URL = "https://zenodo.org/record/3338373/files/musdb18hq.zip"
_EXT = ".wav"
_SAMPLE_RATE = 44100


class MUSDB_HQ(Dataset):
    """Create *MUSDB_HQ Dataset* [:footcite:`Mir2015QUESST2014EQ`] Dataset

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"test"``].
        sources (List[str] or None, optional): Sources extract data from.
            Options: [``bass``, ``drums``, ``other``, ``mixture``, ``vocals``].
            If ``None``, dataset consists of tracks except mixture.
            (default: ``"[``bass``, ``drums``, ``other``, ``vocals``]"``)
        split (str, optional): Whether to split training set into train and validation set.
            If None, no splitting occurs. If ``train`` or ``validation``, returns respective set.
            (default: ``None``)
        validation (List[str] or None, optional): Tracks which are included in validation set.
            If ``None``, dataset consists of no tracks in validation set.
            (default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        sources: List[str] = ["bass", "drums", "other", "vocals"],
        split: Optional[str] = None,
        validation: Optional[List[str]] = None,
    ) -> None:
        self.sources = sources
        self.split = split
        self.validation = validation

        basename = os.path.basename(_URL)
        basename = basename.rsplit(".", 2)[0]
        assert subset in ["test", "train"], "`subset` must be one of ['test', 'train']"
        assert self.split is None or self.split in [
            "train",
            "validation",
        ], "`split` must be one of ['train', 'validation']"
        assert (
            self.split is None and self.validation is None or self.split and self.validation
        ), "`split` and `validation` must both be None or not be None"
        self._path = os.path.join(root, basename, subset)
        self.names = self._collect_songs()

    def _get_file(self, name, source):
        return Path(self._path) / name / f"{source}{_EXT}"

    def _load_sample(self, n: int) -> Tuple[torch.Tensor, int, str]:
        name = self.names[n]
        wavs = []
        sample_rate = None
        for source in self.sources:
            file = self._get_file(name, source)
            wav, new_samplerate = torchaudio.load(str(file))
            if sample_rate is None:
                sample_rate = new_samplerate
            elif sample_rate != new_samplerate:
                raise ValueError(
                    f"Invalid sample rate for file {file}: " f"expecting {sample_rate} but got {new_samplerate}."
                )
            wavs.append(wav)

        stacked = torch.stack(wavs)

        return stacked, sample_rate, name

    def _collect_songs(self):
        if self.split == "validation":
            return self.validation
        path = Path(self._path)
        names = []
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith(".") or folders or root == path:
                continue
            name = str(root.relative_to(path))
            if self.split and name in self.validation:
                continue
            names.append(name)
        return names

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str): ``(waveform, sample_rate, track_name)``
        """
        return self._load_sample(n)

    def __len__(self) -> int:
        return len(self.names)
