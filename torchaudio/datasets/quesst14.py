import os
import re
from pathlib import Path
from typing import Tuple, Union, Optional

import torch
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive
from torchaudio.sox_effects import apply_effects_file


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
    """Create QUESST14 Dataset

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        subset (str): The subset to use. Options: [``dev``, ``eval``]
        language (str, optional): Language to get dataset for.
            Options: [None, ``albanian``, ``basque``, ``czech``, `nnenglish``, ``romanian``, ``slovak``]
        folder_in_archive (str, optional): The top-level directory of the dataset. default: (``"quesst14Database"``)
        subset (str or None, optional): subset of the dataset to use. Options: [None, "dev", "eval"].
            None indicates the whole dataset, including dev and eval.
        download (bool, optional): Whether to download the dataset if it is not found at root path.
            (default: ``False``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        language: str = "nnenglish",
        download: bool = False,
        subset: Optional[str] = None,
    ) -> None:
        assert subset is None or subset in ["dev", "eval"], "`subset` must be one of [None, 'dev', 'eval']"

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

        doc_paths = filter_audio_paths(self._path, language, "language_key_utterances.lst")
        if subset is None:
            dev_paths = filter_audio_paths(self._path, language, "language_key_dev.lst")
            eval_paths = filter_audio_paths(self._path, language, "language_key_eval.lst")
            query_paths = dev_paths + eval_paths
        else:
            query_paths = filter_audio_paths(self._path, language, f"language_key_{subset}.lst")

        self.n_docs = len(doc_paths)
        self.n_queries = len(query_paths)
        self.data = query_paths + doc_paths

    def _load_sample(self, n: int) -> Tuple[torch.Tensor, str]:
        audio_path = self.data[n]
        wav, _ = apply_effects_file(
            str(audio_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["gain", "-3.0"],
            ],
        )
        wav = wav.squeeze(0)
        return wav, audio_path.with_suffix("").name

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str]:
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
