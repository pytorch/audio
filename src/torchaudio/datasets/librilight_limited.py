import os
from pathlib import Path
from typing import List, Tuple, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.librispeech import _get_librispeech_metadata
from torchaudio.datasets.utils import _extract_tar


_ARCHIVE_NAME = "librispeech_finetuning"
_URL = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
_CHECKSUM = "5d1efdc777b548194d7e09ba89126e2188026df9fd57aa57eb14408d2b2342af"
_SUBSET_MAP = {"10min": ["1h/0"], "1h": ["1h/*"], "10h": ["1h/*", "9h"]}


def _get_fileids_paths(path: Path, folders: List[str], _ext_audio: str) -> List[Tuple[str, str]]:
    """Get the file names and the corresponding file paths without `speaker_id`
    and `chapter_id` directories.
    The format of path is like:
        {root}/{_ARCHIVE_NAME}/1h/[0-5]/[clean, other] or
        {root}/{_ARCHIVE_NAME}/9h/[clean, other]

    Args:
        path (Path): Root path to the dataset.
        folders (List[str]): Folders that contain the desired audio files.
        _ext_audio (str): Extension of audio files.

    Returns:
        List[Tuple[str, str]]:
            List of tuples where the first element is the relative path to the audio file.
            The format of relative path is like:
            1h/[0-5]/[clean, other] or 9h/[clean, other]
            The second element is the file name without audio extension.
    """

    path = Path(path)
    files_paths = []
    for folder in folders:
        paths = [p.relative_to(path) for p in path.glob(f"{folder}/*/*/*/*{_ext_audio}")]
        files_paths += [(str(p.parent.parent.parent), str(p.stem)) for p in paths]  # get subset folder and file name
    files_paths.sort(key=lambda x: x[0] + x[1])
    return files_paths


class LibriLightLimited(Dataset):
    """Subset of Libri-light :cite:`librilight` dataset,
    which was used in HuBERT :cite:`hsu2021hubert` for supervised fine-tuning.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): The subset to use. Options: [``"10min"``, ``"1h"``, ``"10h"``]
            (Default: ``"10min"``).
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "10min",
        download: bool = False,
    ) -> None:
        if subset not in _SUBSET_MAP:
            raise ValueError(f"`subset` must be one of {_SUBSET_MAP.keys()}. Found: {subset}")
        folders = _SUBSET_MAP[subset]

        root = os.fspath(root)
        self._path = os.path.join(root, _ARCHIVE_NAME)
        archive = os.path.join(root, f"{_ARCHIVE_NAME}.tgz")
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError("Dataset not found. Please use `download=True` to download")
            if not os.path.isfile(archive):
                download_url_to_file(_URL, archive, hash_prefix=_CHECKSUM)
            _extract_tar(archive)
        self._fileids_paths = _get_fileids_paths(self._path, folders, self._ext_audio)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
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
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        file_path, fileid = self._fileids_paths[n]
        metadata = _get_librispeech_metadata(fileid, self._path, file_path, self._ext_audio, self._ext_txt)
        waveform, _ = torchaudio.load(os.path.join(self._path, metadata[0]))
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._fileids_paths)
