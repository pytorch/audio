import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform


URL = "https://speech.fit.vutbr.cz/files/quesst14Database.tgz"
SAMPLE_RATE = 8000
_CHECKSUM = "4f869e06bc066bbe9c5dde31dbd3909a0870d70291110ebbb38878dcbc2fc5e4"
_LANGUAGES = [
    "albanian",
    "basque",
    "czech",
    "nnenglish",
    "romanian",
    "slovak",
]


class QUESST14(Dataset):
    """*QUESST14* :cite:`Mir2015QUESST2014EQ` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        subset (str): Subset of the dataset to use. Options: [``"docs"``, ``"dev"``, ``"eval"``].
        language (str or None, optional): Language to get dataset for.
            Options: [``None``, ``albanian``, ``basque``, ``czech``, ``nnenglish``, ``romanian``, ``slovak``].
            If ``None``, dataset consists of all languages. (default: ``"nnenglish"``)
        download (bool, optional): Whether to download the dataset if it is not found at root path.
            (default: ``False``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        language: Optional[str] = "nnenglish",
        download: bool = False,
    ) -> None:
        if subset not in ["docs", "dev", "eval"]:
            raise ValueError("`subset` must be one of ['docs', 'dev', 'eval']")

        if language is not None and language not in _LANGUAGES:
            raise ValueError(f"`language` must be None or one of {str(_LANGUAGES)}")

        # Get string representation of 'root'
        root = os.fspath(root)

        basename = os.path.basename(URL)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        self._path = os.path.join(root, basename)

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                if not download:
                    raise RuntimeError("Dataset not found. Please use `download=True` to download")
                download_url_to_file(URL, archive, hash_prefix=_CHECKSUM)
            _extract_tar(archive, root)

        if subset == "docs":
            self.data = filter_audio_paths(self._path, language, "language_key_utterances.lst")
        elif subset == "dev":
            self.data = filter_audio_paths(self._path, language, "language_key_dev.lst")
        elif subset == "eval":
            self.data = filter_audio_paths(self._path, language, "language_key_eval.lst")

    def get_metadata(self, n: int) -> Tuple[str, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                File name
        """
        audio_path = self.data[n]
        relpath = os.path.relpath(audio_path, self._path)
        return relpath, SAMPLE_RATE, audio_path.with_suffix("").name

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                File name
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self.data)


def filter_audio_paths(
    path: str,
    language: str,
    lst_name: str,
):
    """Extract audio paths for the given language."""
    audio_paths = []

    path = Path(path)
    with open(path / "scoring" / lst_name) as f:
        for line in f:
            audio_path, lang = line.strip().split()
            if language is not None and lang != language:
                continue
            audio_path = re.sub(r"^.*?\/", "", audio_path)
            audio_paths.append(path / audio_path)

    return audio_paths
