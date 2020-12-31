import os
import csv
from typing import List, Tuple, Union
from pathlib import Path

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive
from torch import Tensor
from torch.utils.data import Dataset

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "wavs",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "checksum": "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
    }
}


class LJSPEECH(Dataset):
    """Create a Dataset for LJSpeech-1.1.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(self,
                 root: Union[str, Path],
                 url: str = _RELEASE_CONFIGS["release1"]["url"],
                 folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
                 download: bool = False) -> None:

        self._parse_filesystem(root, url, folder_in_archive, download)
    
    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        root = Path(root)

        basename = os.path.basename(url)
        archive = root / basename

        basename = basename.split(".tar.bz2")[0]
        folder_in_archive = os.path.join(basename, folder_in_archive)

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root, basename, 'metadata.csv')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        with open(self._metadata_path, "r", newline='') as metadata:
            walker = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)
    
    def _load_item(self, line: List[str], path: str) -> Tuple[Tensor, int, str, str]:
        assert len(line) == 3
        fileid, transcript, normalized_transcript = line
        fileid_audio = fileid + ".wav"
        fileid_audio = os.path.join(path, fileid_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        return (
            waveform,
            sample_rate,
            transcript,
            normalized_transcript,
        )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, transcript, normalized_transcript)``
        """
        line = self._walker[n]
        item = self._load_item(line, self._path)
        return item

    def __len__(self) -> int:
        return len(self._walker)
