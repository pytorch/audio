import os
from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    extract_archive,
)

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriTTS"
_CHECKSUMS = {
    "http://www.openslr.org/resources/60/dev-clean.tar.gz": "da0864e1bd26debed35da8a869dd5c04dfc27682921936de7cff9c8a254dbe1a",  # noqa: E501
    "http://www.openslr.org/resources/60/dev-other.tar.gz": "d413eda26f3a152ac7c9cf3658ef85504dfb1b625296e5fa83727f5186cca79c",  # noqa: E501
    "http://www.openslr.org/resources/60/test-clean.tar.gz": "234ea5b25859102a87024a4b9b86641f5b5aaaf1197335c95090cde04fe9a4f5",  # noqa: E501
    "http://www.openslr.org/resources/60/test-other.tar.gz": "33a5342094f3bba7ccc2e0500b9e72d558f72eb99328ac8debe1d9080402f10d",  # noqa: E501
    "http://www.openslr.org/resources/60/train-clean-100.tar.gz": "c5608bf1ef74bb621935382b8399c5cdd51cd3ee47cec51f00f885a64c6c7f6b",  # noqa: E501
    "http://www.openslr.org/resources/60/train-clean-360.tar.gz": "ce7cff44dcac46009d18379f37ef36551123a1dc4e5c8e4eb73ae57260de4886",  # noqa: E501
    "http://www.openslr.org/resources/60/train-other-500.tar.gz": "e35f7e34deeb2e2bdfe4403d88c8fdd5fbf64865cae41f027a185a6965f0a5df",  # noqa: E501
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
    """Create a Dataset for LibriTTS.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
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

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, str, int, int, str):
            ``(waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id)``
        """
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
