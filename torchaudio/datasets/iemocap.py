import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform


SAMPLE_RATE = 16000


def _get_wavs_paths(data_dir):
    wav_dir = os.path.join(data_dir, "sentences/wav")
    wav_paths = sorted(str(p) for p in Path(wav_dir).glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths


class IEMOCAP(Dataset):
    """*IEMOCAP* :cite:`iemocap` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        sessions (Optional[List[int]], optional): List of sessions (1-5) to use. If ``None``, loads from all sessions.
    """

    def __init__(
        self,
        root: Union[str, Path],
        sessions: Optional[str] = None,
    ):
        root = os.fspath(root)
        self._path = os.path.join(root, "IEMOCAP")

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        all_data = []
        self.data = []
        self.mapping = defaultdict()
        if sessions is None:
            sessions = [1, 2, 3, 4, 5]

        for session in sessions:
            session_name = f"Session{session}"
            session_dir = os.path.join(self._path, session_name)

            # get wav paths
            wav_paths = _get_wavs_paths(session_dir)
            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
                all_data.append(wav_stem)

            # add labels
            label_dir = os.path.join(session_dir, "dialog/EmoEvaluation")
            label_paths = Path(label_dir).glob("*.txt")

            for label_path in label_paths:
                with open(label_path, "r") as f:
                    for line in f:
                        if line[0] != "[":
                            continue
                        line = re.split("[\t\n]", line)
                        wav_stem = line[1]
                        label = line[2]
                        if label == "exc":
                            label = "hap"
                        if wav_stem not in all_data:
                            continue
                        if label not in ["neu", "hap", "ang", "sad"]:
                            continue
                        self.mapping[wav_stem] = {}
                        self.mapping[wav_stem]["label"] = label

            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
                if wav_stem in self.mapping:
                    self.data.append(wav_stem)
                    self.mapping[wav_stem]["path"] = wav_path

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, str]:
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
            str:
                Label (one of `"neu"`, `"hap"`, `"ang"`, `"sad"`)
            str:
                Speaker
        """
        wav_stem = self.data[n]
        wav_path = self.mapping[wav_stem]["path"]
        label = self.mapping[wav_stem]["label"]
        speaker = wav_stem.split("_")[0]
        return (wav_path, SAMPLE_RATE, wav_stem, label, speaker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
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
            str:
                Label (one of `"neu"`, `"hap"`, `"ang"`, `"sad"`)
            str:
                Speaker
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self):
        return len(self.data)
