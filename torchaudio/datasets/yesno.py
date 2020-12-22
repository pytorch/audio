import os
from pathlib import Path
from typing import Any, List, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
FOLDER_IN_ARCHIVE = "waves_yesno"
_CHECKSUMS = {
    "http://www.openslr.org/resources/1/waves_yesno.tar.gz":
        "962ff6e904d2df1126132ecec6978786"
}


def load_yesno_item(fileid: str, path: str, ext_audio: str) -> Tuple[Tensor, int, List[int]]:
    # Read label
    labels = [int(c) for c in fileid.split("_")]

    # Read wav
    file_audio = os.path.join(path, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, labels


class YESNO(Dataset):
    """Create a Dataset for YesNo.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://www.openslr.org/resources/1/waves_yesno.tar.gz"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"waves_yesno"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        target_transform (callable, optional): Optional transform applied on utterance. (default: ``None``)
    """

    _ext_audio = ".wav"

    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False,
                 target_transform: Any = None) -> None:

        self.target_transform = target_transform

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, labels)``
        """
        fileid = self._walker[n]
        item = load_yesno_item(fileid, self._path, self._ext_audio)

        # TODO Upon deprecation, uncomment line below and remove following code
        # return item

        waveform, sample_rate, labels = item
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return waveform, sample_rate, labels

    def __len__(self) -> int:
        return len(self._walker)
