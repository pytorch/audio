import csv
import os
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform

SAMPLE_RATE = 16000


class FluentSpeechCommands(Dataset):
    """*Fluent Speech Commands* :cite:`fluent` dataset

    Args:
        root (str of Path): Path to the directory where the dataset is found.
        subset (str, optional): subset of the dataset to use.
            Options: [``"train"``, ``"valid"``, ``"test"``].
            (Default: ``"train"``)
    """

    def __init__(self, root: Union[str, Path], subset: str = "train"):
        if subset not in ["train", "valid", "test"]:
            raise ValueError("`subset` must be one of ['train', 'valid', 'test']")

        root = os.fspath(root)
        self._path = os.path.join(root, "fluent_speech_commands_dataset")

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        subset_path = os.path.join(self._path, "data", f"{subset}_data.csv")
        with open(subset_path) as subset_csv:
            subset_reader = csv.reader(subset_csv)
            data = list(subset_reader)

        self.header = data[0]
        self.data = data[1:]

    def get_metadata(self, n: int) -> Tuple[str, int, str, int, str, str, str, str]:
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
            int:
                Speaker ID
            str:
                Transcription
            str:
                Action
            str:
                Object
            str:
                Location
        """
        sample = self.data[n]

        file_name = sample[self.header.index("path")].split("/")[-1]
        file_name = file_name.split(".")[0]
        speaker_id, transcription, action, obj, location = sample[2:]
        file_path = os.path.join("wavs", "speakers", speaker_id, f"{file_name}.wav")

        return file_path, SAMPLE_RATE, file_name, speaker_id, transcription, action, obj, location

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, str, str, str, str]:
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
            int:
                Speaker ID
            str:
                Transcription
            str:
                Action
            str:
                Object
            str:
                Location
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]
