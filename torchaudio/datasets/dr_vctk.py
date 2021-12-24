from pathlib import Path
from typing import Dict, Tuple, Union

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    extract_archive,
)


_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"
_CHECKSUM = "781f12f4406ed36ed27ae3bce55da47ba176e2d8bae67319e389e07b2c9bd769"
_SUPPORTED_SUBSETS = {"train", "test"}


class DR_VCTK(Dataset):
    """Create a dataset for Device Recorded VCTK (Small subset version).

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found.
        subset (str): The subset to use. Can be one of ``"train"`` and ``"test"``. (default: ``"train"``).
        download (bool):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str): The URL to download the dataset from.
            (default: ``"https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train",
        *,
        download: bool = False,
        url: str = _URL,
    ) -> None:
        if subset not in _SUPPORTED_SUBSETS:
            raise RuntimeError(
                f"The subset '{subset}' does not match any of the supported subsets: {_SUPPORTED_SUBSETS}"
            )

        root = Path(root).expanduser()
        archive = root / "DR-VCTK.zip"

        self._subset = subset
        self._path = root / "DR-VCTK" / "DR-VCTK"
        self._clean_audio_dir = self._path / f"clean_{self._subset}set_wav_16k"
        self._noisy_audio_dir = self._path / f"device-recorded_{self._subset}set_wav_16k"
        self._config_filepath = self._path / "configurations" / f"{self._subset}_ch_log.txt"

        if not self._path.is_dir():
            if not archive.is_file():
                if not download:
                    raise RuntimeError("Dataset not found. Please use `download=True` to download it.")
                download_url_to_file(url, archive, hash_prefix=_CHECKSUM)
            extract_archive(archive, root)

        self._config = self._load_config(self._config_filepath)
        self._filename_list = sorted(self._config)

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
        file_clean_audio = self._clean_audio_dir / filename
        file_noisy_audio = self._noisy_audio_dir / filename
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
            (Tensor, int, Tensor, int, str, str, str, int):
            ``(waveform_clean, sample_rate_clean, waveform_noisy, sample_rate_noisy, speaker_id,\
                utterance_id, source, channel_id)``
        """
        filename = self._filename_list[n]
        return self._load_dr_vctk_item(filename)

    def __len__(self) -> int:
        return len(self._filename_list)
