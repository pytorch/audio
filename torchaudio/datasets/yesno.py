import os
from pathlib import Path
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "waves_yesno",
        "url": "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
        "checksum": "30301975fd8c5cac4040c261c0852f57cfa8adbbad2ce78e77e4986957445f27",
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
        root: str,
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False
        ) -> None:

        self._ext_audio = ".wav"
        self._parse_filesystem(root, url, folder_in_archive, download)

    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool):
        root = Path(root)
        archive = os.path.basename(url)
        archive = root / archive

        self._path = os.path.join(root, folder_in_archive)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS["release1"]["checksum"]
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*' + self._ext_audio))

    def _load_item(self, fileid: str, path: str, ext_audio: str):
        labels = [int(c) for c in fileid.split("_")]
        file_audio = os.path.join(path, fileid + ext_audio)
        waveform, sample_rate = torchaudio.load(file_audio)
        return waveform, sample_rate, labels

    def __getitem__(self, n: int):
        fileid = self._walker[n]
        item = self._load_item(fileid, self._path, self._ext_audio)
        return item

    def __len__(self):
        return len(self._walker)
