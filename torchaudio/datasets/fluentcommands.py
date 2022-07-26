import csv
import os
from pathlib import Path
from typing import Union

import torchaudio
from torch.utils.data import Dataset


class FluentSpeechCommands(Dataset):
    """Create *Fluent Speech Commands* [:footcite:`fluent`] Dataset

    Args:
        root (str of Path): Path to the directory where the dataset is found.
        subset (str, optional): subset of the dataset to use. Options: [`"train"`, `"valid"`, `"test"`].
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, int, str, str, str, str):
            ``(waveform, sample_rate, file_name, speaker_id, transcription, action, object, location)``
        """

        sample = self.data[n]

        file_name = sample[self.header.index("path")].split("/")[-1]
        file_name = file_name.split(".")[0]
        speaker_id, transcription, action, obj, location = sample[2:]

        wav_path = os.path.join(self._path, "wavs", "speakers", speaker_id, f"{file_name}.wav")
        wav, sample_rate = torchaudio.load(wav_path)

        return wav, sample_rate, file_name, speaker_id, transcription, action, obj, location
