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


_URL = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
_CHECKSUM = "5d1efdc777b548194d7e09ba89126e2188026df9fd57aa57eb14408d2b2342af"


def _get_files(path, subset, _ext_audio):
    if subset == "10min":
        files = sorted(str(p) for p in Path(path).glob("1h/0/*/*/*/*" + _ext_audio))
    elif subset in ["1h", "10h"]:
        files = [str(p) for p in Path(path).glob("1h/*/*/*/*/*" + _ext_audio)]
        if subset == "10h":
            files += [str(p) for p in Path(path).glob("9h/*/*/*/*" + _ext_audio)]
        files = sorted(files)
    else:
        raise ValueError(f"Unsupported subset value. Found {subset}.")
    return files


def _load_item(file_path: str, ext_audio: str, ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    fileid = os.path.basename(file_path)
    path = os.path.dirname(file_path)
    speaker_id, chapter_id, utterance_id = fileid.replace(ext_audio, "").split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        sample_rate,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


class LibriLightLimited(Dataset):
    """Create a Dataset for LibriLightLimited, which is the supervised subset of
        LibriLight dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): The subset to use. Options: [``10min`, ``1h``, ``10h``]
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
        self._path = os.path.join(root, "librispeech_finetuning")
        archive = os.path.join(root, "librispeech_finetuning" + ".tgz")
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url_to_file(_URL, archive, hash_prefix=_CHECKSUM)
                extract_archive(archive)
        self._files = _get_files(self._path, subset, self._ext_audio)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """
        file_path = self._files[n]
        return _load_item(file_path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._files)
