import os
import csv

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    get_checksum
)
from torch.utils.data import Dataset

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FOLDER_IN_ARCHIVE = "wavs"


def load_ljspeech_item(line, path, ext_audio):
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

    def __init__(
            self, root, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False
    ):

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        folder_in_archive = os.path.join(basename, folder_in_archive)

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root, basename, 'metadata.csv')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = get_checksum(dataset=__class__.__name__, url=url)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)

    def __getitem__(self, n):
        line = self._walker[n]
        return load_ljspeech_item(line, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)
