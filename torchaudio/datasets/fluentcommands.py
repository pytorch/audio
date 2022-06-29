import csv
from pathlib import Path, PurePosixPath
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
        assert subset in ["train", "valid", "test"], "`subset` must be one of ['train', 'valid', 'test']"
        self._path = Path(root) / "fluent_speech_commands_dataset"

        subset_path = self._path / "data" / f"{subset}_data.csv"
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
        _, path, speaker_id, transcription, action, obj, location = self.data[n]
        path = PurePosixPath(path)

        wav_path = self._path / path
        wav, sample_rate = torchaudio.load(str(wav_path))

        return wav, sample_rate, str(path.stem), speaker_id, transcription, action, obj, location
