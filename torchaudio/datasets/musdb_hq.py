import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip

_URL = "https://zenodo.org/record/3338373/files/musdb18hq.zip"
_CHECKSUM = "baac80d0483c61d74b2e5f3be75fa557eec52898339e6aa45c1fa48833c5d21d"
_EXT = ".wav"
_SAMPLE_RATE = 44100
_VALIDATION_SET = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]


class MUSDB_HQ(Dataset):
    """*MUSDB_HQ* :cite:`MUSDB18HQ` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"test"``].
        sources (List[str] or None, optional): Sources extract data from.
            List can contain the following options: [``"bass"``, ``"drums"``, ``"other"``, ``"mixture"``, ``"vocals"``].
            If ``None``, dataset consists of tracks except mixture.
            (default: ``None``)
        split (str or None, optional): Whether to split training set into train and validation set.
            If ``None``, no splitting occurs. If ``train`` or ``validation``, returns respective set.
            (default: ``None``)
        download (bool, optional): Whether to download the dataset if it is not found at root path.
            (default: ``False``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        sources: Optional[List[str]] = None,
        split: Optional[str] = None,
        download: bool = False,
    ) -> None:
        self.sources = ["bass", "drums", "other", "vocals"] if not sources else sources
        self.split = split

        basename = os.path.basename(_URL)
        archive = os.path.join(root, basename)
        basename = basename.rsplit(".", 2)[0]

        if subset not in ["test", "train"]:
            raise ValueError("`subset` must be one of ['test', 'train']")
        if self.split is not None and self.split not in ["train", "validation"]:
            raise ValueError("`split` must be one of ['train', 'validation']")
        base_path = os.path.join(root, basename)
        self._path = os.path.join(base_path, subset)
        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                if not download:
                    raise RuntimeError("Dataset not found. Please use `download=True` to download")
                download_url_to_file(_URL, archive, hash_prefix=_CHECKSUM)
            os.makedirs(base_path, exist_ok=True)
            _extract_zip(archive, base_path)

        self.names = self._collect_songs()

    def _get_track(self, name, source):
        return Path(self._path) / name / f"{source}{_EXT}"

    def _load_sample(self, n: int) -> Tuple[torch.Tensor, int, int, str]:
        name = self.names[n]
        wavs = []
        num_frames = None
        for source in self.sources:
            track = self._get_track(name, source)
            wav, sr = torchaudio.load(str(track))
            if sr != _SAMPLE_RATE:
                raise ValueError(f"expected sample rate {_SAMPLE_RATE}, but got {sr}")
            if num_frames is None:
                num_frames = wav.shape[-1]
            else:
                if wav.shape[-1] != num_frames:
                    raise ValueError("num_frames do not match across sources")
            wavs.append(wav)

        stacked = torch.stack(wavs)

        return stacked, _SAMPLE_RATE, num_frames, name

    def _collect_songs(self):
        if self.split == "validation":
            return _VALIDATION_SET
        path = Path(self._path)
        names = []
        for root, folders, _ in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith(".") or folders or root == path:
                continue
            name = str(root.relative_to(path))
            if self.split and name in _VALIDATION_SET:
                continue
            names.append(name)
        return sorted(names)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            int:
                Num frames
            str:
                Track name
        """
        return self._load_sample(n)

    def __len__(self) -> int:
        return len(self.names)
