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
    "http://www.openslr.org/60/dev-clean.tar.gz": "0c3076c1e5245bb3f0af7d82087ee207",
    "http://www.openslr.org/60/dev-other.tar.gz": "815555d8d75995782ac3ccd7f047213d",
    "http://www.openslr.org/60/test-clean.tar.gz": "7bed3bdb047c4c197f1ad3bc412db59f",
    "http://www.openslr.org/60/test-other.tar.gz": "ae3258249472a13b5abef2a816f733e4",
    "http://www.openslr.org/60/train-clean-100.tar.gz": "4a8c202b78fe1bc0c47916a98f3a2ea8",
    "http://www.openslr.org/60/train-clean-360.tar.gz": "a84ef10ddade5fd25df69596a2767b2d",
    "http://www.openslr.org/60/train-other-500.tar.gz": "7b181dd5ace343a5f38427999684aa6f",
}


def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
) -> Tuple[Tensor, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid

    normalized_text = utterance_id + ext_normalized_txt
    normalized_text = os.path.join(path, speaker_id, chapter_id, normalized_text)

    original_text = utterance_id + ext_original_txt
    original_text = os.path.join(path, speaker_id, chapter_id, original_text)

    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load original text
    with open(original_text) as ft:
        original_text = ft.readline()

    # Load normalized text
    with open(normalized_text, "r") as ft:
        normalized_text = ft.readline()

    return (
        waveform,
        sample_rate,
        original_text,
        normalized_text,
        int(speaker_id),
        int(chapter_id),
        utterance_id,
    )


class LIBRITTS(Dataset):
    """
    Create a Dataset for LibriTTS. Each item is a tuple of the form:
    waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id
    """

    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:

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
            base_url = "http://www.openslr.org/resources/60/"

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

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
        fileid = self._walker[n]
        return load_libritts_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_original_txt,
            self._ext_normalized_txt,
        )

    def __len__(self) -> int:
        return len(self._walker)
