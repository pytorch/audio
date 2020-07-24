import os
import csv
from typing import List, Tuple

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch import Tensor
from torch.utils.data import Dataset

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FOLDER_IN_ARCHIVE = "wavs"
_CHECKSUMS = {
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2":
    "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5"
}


def load_ljspeech_item(line: List[str], path: str, ext_audio: str) -> Tuple[Tensor, int, str, str]:
    print(line)
    assert len(line) == 3
    fileid, transcript, normalized_transcript = line
    fileid_audio = fileid + ext_audio
    fileid_audio = os.path.join(path, fileid_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(fileid_audio)

    return (
        waveform,
        sample_rate,
        transcript,
        normalized_transcript,
    )


class LJSPEECH(Dataset):
    """
    Create a Dataset for LJSpeech-1.1. Each item is a tuple of the form:
    waveform, sample_rate, transcript, normalized_transcript
    """

    _ext_audio = ".wav"
    _ext_archive = '.tar.bz2'

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        folder_in_archive = os.path.join(basename, folder_in_archive)

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root, basename, 'metadata.csv')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        line = self._walker[n]
        return load_ljspeech_item(line, self._path, self._ext_audio)

    def __len__(self) -> int:
        return len(self._walker)
