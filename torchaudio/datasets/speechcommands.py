import os
from typing import Tuple, Optional

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files
)

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}


def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    """Create a Dataset for Speech Commands.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (Optional[str]):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. (default: ``None``)
    """

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False,
                 subset: Optional[str] = None,
                 ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        def _load_list(*filenames):
            output = []
            for filename in filenames:
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    output += [os.path.join(self._path, line.strip()) for line in fileobj]
            return output

        if subset == "validation":
            self._walker = _load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list("validation_list.txt", "testing_list.txt"))
            walker = walk_files(self._path, suffix=".wav", prefix=True)
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and w not in excludes]
        else:
            walker = walk_files(self._path, suffix=".wav", prefix=True)
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)
