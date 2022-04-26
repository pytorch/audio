import os
from pathlib import Path
from typing import Tuple, Union, Optional

import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    extract_archive,
)


_RELEASE_CONFIGS = {
    "dev": {
        "url": "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav.zip",
        "checksum": "ff3b4ce606718a2d221299d21f1dded47097907762c6783e47fe823cad9f001e",
    },
    "test": {
        "url": "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip",
        "checksum": "8de57f347fe22b2c24526e9f444f689ecf5096fc2a92018cf420ff6b5b15eaea",
    },
}


class VoxCeleb1(Dataset):
    """Create VoxCeleb1 Dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str or None, optional): Subset of the dataset to use. Options: ["dev", "test"]. (Default: ``None``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).
    """

    _ext_audio = ".wav"

    def __init__(self, root: Union[str, Path], subset: Optional[str] = None, download: bool = False) -> None:
        assert subset is None or subset in ["dev", "test"], "`subset` must be one of ['dev', 'test']"

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(_RELEASE_CONFIGS[subset]["url"])
        archive = os.path.join(root, basename)
        self._path = os.path.join(root, basename.replace(".zip", ""))
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS[subset]["checksum"]
                    url = _RELEASE_CONFIGS[subset]["url"]
                    # dev data is splited to
                    if subset == "dev":
                        with open(archive, "wb") as f:
                            for split in ["_partaa", "_partab", "_partac", "_partad"]:
                                download_url_to_file(url.replace(".zip", split), archive.replace(".zip", split))
                                with open(archive.replace(".zip", split), "rb") as f_split:
                                    f.write(f_split.read())
                    else:
                        download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive, self._path)

        self._walker = sorted(str(p) for p in Path(self._path).glob("wav/*/*/*" + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, int, str):
            ``(waveform, sample_rate, speaker_id, youtube_id)``
        """
        filename = self._walker[n]
        speaker, youtube_id, _ = filename.split("/")[-3:]
        speaker_id = int(speaker[3:])
        waveform, sample_rate = torchaudio.load(filename)
        return (waveform, sample_rate, speaker_id, youtube_id)

    def __len__(self) -> int:
        return len(self._walker)
