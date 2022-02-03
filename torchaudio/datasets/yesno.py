import os
from pathlib import Path
from typing import List, Tuple, Union

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    extract_archive,
)


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "waves_yesno",
        "url": "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
        "checksum": "c3f49e0cca421f96b75b41640749167b52118f232498667ca7a5f9416aef8e73",
    }
}


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
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False,
    ) -> None:

        self._parse_filesystem(root, url, folder_in_archive, download)

    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        root = Path(root)
        archive = os.path.basename(url)
        archive = root / archive

        self._path = root / folder_in_archive
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS["release1"]["checksum"]
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*.wav"))

    def _load_item(self, fileid: str, path: str):
        labels = [int(c) for c in fileid.split("_")]
        file_audio = os.path.join(path, fileid + ".wav")
        waveform, sample_rate = torchaudio.load(file_audio)
        return waveform, sample_rate, labels

    def __getitem__(self, n: int) -> Tuple[Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        fileid = self._walker[n]
        item = self._load_item(fileid, self._path)
        return item

    def __len__(self) -> int:
        return len(self._walker)
