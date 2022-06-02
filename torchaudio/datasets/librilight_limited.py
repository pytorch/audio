import os
from pathlib import Path
from typing import List, Tuple, Union

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.librispeech import load_librispeech_item
from torchaudio.datasets.utils import extract_archive


_ARCHIVE_NAME = "librispeech_finetuning"
_URL = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
_CHECKSUM = "5d1efdc777b548194d7e09ba89126e2188026df9fd57aa57eb14408d2b2342af"


def _get_fileids_paths(path, subset, _ext_audio) -> List[Tuple[str, str]]:
    """Get the file names and the corresponding file paths without `speaker_id`
    and `chapter_id` directories.
    The format of path is like:
        {root}/{_ARCHIVE_NAME}/1h/[0-5]/[clean, other] or
        {root}/{_ARCHIVE_NAME}/9h/[clean, other]
    """
    if subset == "10min":
        files_paths = [
            (os.path.join(os.path.dirname(p), "..", ".."), str(p.stem))
            for p in Path(path).glob("1h/0/*/*/*/*" + _ext_audio)
        ]
    elif subset in ["1h", "10h"]:
        files_paths = [
            (os.path.join(os.path.dirname(p), "..", ".."), str(p.stem))
            for p in Path(path).glob("1h/*/*/*/*/*" + _ext_audio)
        ]
        if subset == "10h":
            files_paths += [
                (os.path.join(os.path.dirname(p), "..", ".."), str(p.stem))
                for p in Path(path).glob("9h/*/*/*/*" + _ext_audio)
            ]
    else:
        raise ValueError(f"Unsupported subset value. Found {subset}.")
    files_paths = sorted(files_paths, key=lambda x: x[0] + x[1])
    return files_paths


class LibriLightLimited(Dataset):
    """Create a Dataset for LibriLightLimited, which is the supervised subset of
        LibriLight dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): The subset to use. Options: [``10min``, ``1h``, ``10h``]
            (Default: ``10min``).
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
        assert subset in ["10min", "1h", "10h"], "`subset` must be one of ['10min', '1h', '10h']"

        root = os.fspath(root)
        self._path = os.path.join(root, _ARCHIVE_NAME)
        archive = os.path.join(root, f"{_ARCHIVE_NAME}.tgz")
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError("Dataset not found. Please use `download=True` to download")
            if not os.path.isfile(archive):
                download_url_to_file(_URL, archive, hash_prefix=_CHECKSUM)
            extract_archive(archive)
        self._fileids_paths = _get_fileids_paths(self._path, subset, self._ext_audio)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """
        file_path, fileid = self._fileids_paths[n]
        return load_librispeech_item(fileid, file_path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._fileids_paths)
