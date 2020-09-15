import os
import time
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, walk_files
import pandas
import numpy
from typing import Tuple

# Download URL and checksums
URL = {
    "esc-10": "https://github.com/karoldvl/ESC-50/archive/master.zip",
    "esc-50": "https://github.com/karoldvl/ESC-50/archive/master.zip",
    # "esc-us": None,
}

_CHECKSUMS = {
    "esc-10": None,
    "esc-50": None,
    # "esc-us": None,
}

# Constant
ARCHIVE_BASENAME = "ESC-50-master"
FOLDER_IN_ARCHIVE = "ESC-50-master"
AUDIO_FOLDER = "audio"
META_FOLDER = "meta"
AVAILABLE_VERSION = list(URL.keys())

# Default parameters
FOLDS = (1, 2, 3, 4, 5)


class ESC50(Dataset):
    """
    ESC datasets

    Args:
        root (string): Root directory of datasets where directory
            ``ESC-50-master`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, download the dataset from the internet
            and puts it in root directory. If datasets is already downloaded, it is
            not downloaded again.
    """
    NB_CLASS = 50
    def __init__(self,
                root: str,
                folds: tuple = FOLDS,
                download: bool = False) -> None:

        super().__init__()

        self.root = root
        self.folds = folds

        self.url = URL["esc-50"]
        self.nb_class = 50
        self.target_directory = os.path.join(self.root, FOLDER_IN_ARCHIVE)

        # Dataset must exist to continue
        if download:
            self.download()
        elif not self.check_integrity(self.target_directory):
            raise RuntimeError("Dataset not found or corrupted. \n\
                You can use download=True to download it.")

        self.metadata = self._load_metadata()

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (raw_audio, sr, target).
        """
        filename = self.metadata.iloc[index].name
        return self.load_item(filename)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_metadata(self) -> pandas.DataFrame:
        """Return the dataset medata into a pandas DataFrame."""
        path = os.path.join(self.target_directory, META_FOLDER, "esc50.csv")
        total = pandas.read_csv(path)
        total = total.set_index("filename")

        # Keep only the selected folds
        total = total.loc[total.fold.isin(self.folds)]

        return total

    def download(self) -> None:
        """Download the dataset and extract the archive"""
        if self.check_integrity(self.target_directory):
            print("Dataset already downloaded and verified.")
            
        else:
            archive_basename = os.path.basename(self.url)
            archive_path = os.path.join(self.root, FOLDER_IN_ARCHIVE + ".zip")

            download_url(self.url, self.root)
            extract_archive(archive_path, self.target_directory)

    def check_integrity(self, path, checksum=None) -> bool:
        """Check if the dataset already exist and if yes, if it is not corrupted.

        Returns:
            bool: False if the dataset doesn't exist or if it is corrupted.
        """
        if not os.path.isdir(path):
            return False

        # TODO add checksum verification
        return True

    def load_item(self, filename) -> Tuple[Tensor, int]:
        path = os.path.join(self.target_directory, AUDIO_FOLDER, filename)

        waveform, sample_rate = torchaudio.load(path)
        target = self.metadata.at[filename, "target"]

        return waveform, sample_rate, target


class ESC10(ESC50):
    def __init__(self, 
                root: str,
                folds: tuple = FOLDS,
                download: bool = False) -> None:
        super().__init__(root, folds, download)

        self.url = URL["esc-10"]
        self.nb_class = 10

    def _load_metadata(self) -> pandas.DataFrame:
        meta = super()._load_metadata()

        # Keep only the esc-10 relevant files
        meta = meta.loc[meta.esc10 == True]
        
        return meta