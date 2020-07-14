import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriTTS"
_CHECKSUMS = {
    "http://www.openslr.org/60/dev-clean.tar.gz":
    "0c3076c1e5245bb3f0af7d82087ee207",
    "http://www.openslr.org/60/dev-other.tar.gz":
    "815555d8d75995782ac3ccd7f047213d",
    "http://www.openslr.org/60/test-clean.tar.gz":
    "7bed3bdb047c4c197f1ad3bc412db59f",
    "http://www.openslr.org/60/test-other.tar.gz":
    "ae3258249472a13b5abef2a816f733e4",
    "http://www.openslr.org/60/train-clean-100.tar.gz":
    "4a8c202b78fe1bc0c47916a98f3a2ea8",
    "http://www.openslr.org/60/train-clean-360.tar.gz":
    "a84ef10ddade5fd25df69596a2767b2d",
    "http://www.openslr.org/60/train-other-500.tar.gz":
    "7b181dd5ace343a5f38427999684aa6f"
}


def load_libritts_item(fileid: str,
                          path: str,
                          ext_audio: str,
                          ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        sample_rate,
        utterance,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


class LIBRISPEECH(Dataset):
    """
    Create a Dataset for LibriTTS. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".wav"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/60/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
