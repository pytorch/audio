import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive


URL = "https://speech.fit.vutbr.cz/files/quesst14Database.tgz"
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
    """Create *QUESST14* [:footcite:`Mir2015QUESST2014EQ`] Dataset

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
        assert subset in ["docs", "dev", "eval"], "`subset` must be one of ['docs', 'dev', 'eval']"

        assert language is None or language in _LANGUAGES, f"`language` must be None or one of {str(_LANGUAGES)}"

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
            extract_archive(archive, root)

        if subset == "docs":
            self.data = filter_audio_paths(self._path, language, "language_key_utterances.lst")
        elif subset == "dev":
            self.data = filter_audio_paths(self._path, language, "language_key_dev.lst")
        elif subset == "eval":
            self.data = filter_audio_paths(self._path, language, "language_key_eval.lst")

    def _load_sample(self, n: int) -> Tuple[torch.Tensor, int, str]:
        audio_path = self.data[n]
        wav, sample_rate = torchaudio.load(audio_path)
        return wav, sample_rate, audio_path.with_suffix("").name

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str): ``(waveform, sample_rate, file_name)``
        """
        return self._load_sample(n)

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
