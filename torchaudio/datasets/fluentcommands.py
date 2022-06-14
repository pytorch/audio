import csv
import os
from pathlib import Path
from typing import Union

import torchaudio
from torch.utils.data import Dataset

FOLDER_IN_ARCHIVE = "fluent_speech_commands_dataset"


class FluentSpeechCommands(Dataset):
    """Create *Fluent Speech Commands* [:footcite:`fluent`] Dataset

    Args:
        root (str of Path): Path to the directory where the dataset is found.
        subset (str): subset of the dataset to use. Options: ["train", "valid", "test"].
            (Default: ``"train"``)
    """

    def __init__(self, root: Union[str, Path], subset: str = "train"):
        assert subset in ["train", "valid", "test"]

        root = os.fspath(root)
        self._path = os.path.join(root, FOLDER_IN_ARCHIVE)

        subset_path = os.path.join(self._path, "data", f"{subset}_data.csv")
        with open(subset_path) as subset_csv:
            subset_reader = csv.reader(subset_csv)
            data = list(subset_reader)

        self.header = data[0]
        self.data = data[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, Path, int, str, str, str, str):
            ``(waveform, sample_rate, path, speaker_id, transcription, action, object, location)``
        """

        sample = self.data[n]
        wav_path = os.path.join(self._path, sample[self.header.index("path")])
        wav, sample_rate = torchaudio.load(wav_path)

        path = Path(wav_path).stem
        speaker_id, transcription, action, obj, location = sample[2:]

        return wav, sample_rate, path, speaker_id, transcription, action, obj, location
