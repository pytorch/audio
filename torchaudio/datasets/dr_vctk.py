import os
from pathlib import Path
from typing import Dict, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"
_CHECKSUMS = {
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip": "29e93debeb0e779986542229a81ff29b",
}


class DR_VCTK(Dataset):
    """Create a dataset for Device Recorded VCTK (Small subset version).

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found.
        download (bool):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str): The URL to download the dataset from.
            (default: ``"https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"``)
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
            (default: ``True``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        download: bool = False,
        url: str = URL,
        train: bool = True,
    ) -> None:
        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        archive = os.path.join(root, "DR-VCTK.zip")
        self._subset = "train" if train else "test"
        self._path = os.path.join(root, "DR-VCTK")
        self._clean_audio_dir = os.path.join(self._path, f"clean_{self._subset}set_wav_16k")
        self._noisy_audio_dir = os.path.join(self._path, f"device-recorded_{self._subset}set_wav_16k")
        self._config_filepath = os.path.join(self._path, "configurations", f"{self._subset}_ch_log.txt")

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self._config = self._load_config(self._config_filepath)
        self._walker = sorted(self._config)

    def _load_config(self, filepath: str) -> Dict[str, Tuple[str, int]]:
        # Skip header
        skip_rows = 2 if self._subset == "train" else 1

        config = {}
        with open(filepath) as f:
            for i, line in enumerate(f):
                if i < skip_rows or not line:
                    continue
                filename, source, channel_id = line.strip().split("\t")
                config[filename] = (source, int(channel_id))
        return config

    def _load_dr_vctk_item(self, filename: str) -> Tuple[Tensor, int, Tensor, int, str, str, str, int]:
        speaker_id, utterance_id = filename.split(".")[0].split("_")
        source, channel_id = self._config[filename]
        file_clean_audio = os.path.join(self._clean_audio_dir, filename)
        file_noisy_audio = os.path.join(self._noisy_audio_dir, filename)
        waveform_clean, sample_rate_clean = torchaudio.load(file_clean_audio)
        waveform_noisy, sample_rate_noisy = torchaudio.load(file_noisy_audio)
        return (
            waveform_clean,
            sample_rate_clean,
            waveform_noisy,
            sample_rate_noisy,
            speaker_id,
            utterance_id,
            source,
            channel_id,
        )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Tensor, int, str, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform_clean, sample_rate_clean, waveform_noisy, sample_rate_noisy, speaker_id, utterance_id,\
                source, channle_id)``
        """
        filename = self._walker[n]
        return self._load_dr_vctk_item(filename)

    def __len__(self) -> int:
        return len(self._walker)
