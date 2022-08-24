import os
from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
_DATA_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}


def download_librispeech(root, url):
    base_url = "http://www.openslr.org/resources/12/"
    ext_archive = ".tar.gz"

    filename = url + ext_archive
    archive = os.path.join(root, filename)
    download_url = os.path.join(base_url, filename)
    if not os.path.isfile(archive):
        checksum = _CHECKSUMS.get(download_url, None)
        download_url_to_file(download_url, archive, hash_prefix=checksum)
    extract_archive(archive)


def get_librispeech_metadata(
    fileid: str, root: str, subset: str, ext_audio: str, ext_txt: str
) -> Tuple[str, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    filepath = os.path.join(subset, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")
    full_path = os.path.join(root, filepath)
    sample_rate = torchaudio.info(full_path).sample_rate

    # Load text
    file_text = f"{speaker_id}-{chapter_id}{ext_txt}"
    file_text = os.path.join(root, subset, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return (
        filepath,
        sample_rate,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


def load_librispeech_item(
    fileid: str, path: str, ext_audio: str, ext_txt: str
) -> Tuple[Tensor, int, str, int, int, int]:
    path = os.path.normpath(path)
    root = os.path.dirname(path)
    subset = os.path.basename(path)

    filepath, sample_rate, transcript, speaker_id, chapter_id, utterance_id = get_librispeech_metadata(
        fileid, root, subset, ext_audio, ext_txt
    )
    waveform, _ = torchaudio.load(os.path.join(root, filepath))

    return (
        waveform,
        sample_rate,
        transcript,
        speaker_id,
        chapter_id,
        utterance_id,
    )


class LibriSpeechBase(Dataset):
    """Create a Dataset for *LibriSpeech* [:footcite:`7178964`].

    Args:
        root (str or Path): Path to the directory where the dataset is found.
        subset (str, optional): Subset of LibriSpeech to use.
            Valid options: ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, root: Union[str, Path], subset: str = "train-clean-100") -> None:
        if subset not in _DATA_SUBSETS:
            raise ValueError(f"Invalid subset '{subset}' given; please provide one of {_DATA_SUBSETS}.")

        root = os.fspath(root)
        self._root = root
        self._subset = subset
        subset_path = os.path.join(root, subset)

        if not os.path.isdir(subset_path):
            raise RuntimeError(f"Dataset not found at {subset_path}.")

        self._walker = sorted(str(p.stem) for p in Path(subset_path).glob("*/*/*" + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (str, int, str, int, int, int):
            ``(filepath, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return get_librispeech_metadata(fileid, self._root, self._subset, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)


class LIBRISPEECH(LibriSpeechBase):
    """Create a Dataset for *LibriSpeech* [:footcite:`7178964`].

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:
        archive = os.path.join(root, folder_in_archive)
        self._path = os.path.join(archive, url)

        if not os.path.isdir(self._path) and download:
            download_librispeech(archive, url)

        super().__init__(root=archive, subset=url)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
