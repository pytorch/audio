import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform


_SAMPLE_RATE = 16000
_SPEAKERS = [
    "Aditi",
    "Amy",
    "Brian",
    "Emma",
    "Geraint",
    "Ivy",
    "Joanna",
    "Joey",
    "Justin",
    "Kendra",
    "Kimberly",
    "Matthew",
    "Nicole",
    "Raveena",
    "Russell",
    "Salli",
]


def _load_labels(file: Path, subset: str):
    """Load transcirpt, iob, and intent labels for all utterances.

    Args:
        file (Path): The path to the label file.
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"valid"``, ``"test"``].

    Returns:
        Dictionary of labels, where the key is the filename of the audio,
            and the label is a Tuple of transcript, Inside–outside–beginning (IOB) label, and intention label.
    """
    labels = {}
    with open(file, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            index = line[0]
            trans, iob_intent = " ".join(line[1:]).split("\t")
            trans = " ".join(trans.split(" ")[1:-1])
            iob = " ".join(iob_intent.split(" ")[1:-1])
            intent = iob_intent.split(" ")[-1]
            if subset in index:
                labels[index] = (trans, iob, intent)
    return labels


class Snips(Dataset):
    """*Snips* :cite:`coucke2018snips` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found.
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"valid"``, ``"test"``].
        speakers (List[str] or None, optional): The speaker list to include in the dataset. If ``None``,
            include all speakers in the subset. (Default: ``None``)
        audio_format (str, optional): The extension of the audios. Options: [``"mp3"``, ``"wav"``].
            (Default: ``"mp3"``)
    """

    _trans_file = "all.iob.snips.txt"

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        speakers: Optional[List[str]] = None,
        audio_format: str = "mp3",
    ) -> None:
        if subset not in ["train", "valid", "test"]:
            raise ValueError('`subset` must be one of ["train", "valid", "test"].')
        if audio_format not in ["mp3", "wav"]:
            raise ValueError('`audio_format` must be one of ["mp3", "wav].')

        root = Path(root)
        self._path = root / "SNIPS"
        self.audio_path = self._path / subset
        if speakers is None:
            speakers = _SPEAKERS

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        self.audio_paths = self.audio_path.glob(f"*.{audio_format}")
        self.data = []
        for audio_path in sorted(self.audio_paths):
            audio_name = str(audio_path.name)
            speaker = audio_name.split("-")[0]
            if speaker in speakers:
                self.data.append(audio_path)
        transcript_path = self._path / self._trans_file
        self.labels = _load_labels(transcript_path, subset)

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of the following items:

            str:
                Path to audio
            int:
                Sample rate
            str:
                File name
            str:
                Transcription of audio
            str:
                Inside–outside–beginning (IOB) label of transcription
            str:
                Intention label of the audio.
        """
        audio_path = self.data[n]
        relpath = os.path.relpath(audio_path, self._path)
        file_name = audio_path.with_suffix("").name
        transcript, iob, intent = self.labels[file_name]
        return relpath, _SAMPLE_RATE, file_name, transcript, iob, intent

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items:

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                File name
            str:
                Transcription of audio
            str:
                Inside–outside–beginning (IOB) label of transcription
            str:
                Intention label of the audio.
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self.data)
