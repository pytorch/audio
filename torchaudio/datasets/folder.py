import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import walk_files

def load_audio_item(filepath: str, path: str, label_name: str) -> Tuple[Tensor, int, str, str]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    if label_name is not None:
        label = label_name
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, filename

class AudioFolder(Dataset):
    """Create a Dataset from Local Files.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        suffix (str) : Audio file type, defaulted to ".WAV".
        transform (callable, optional): A function/transform that  takes in the audio waveform
            and returns a transformed version. E.g, ``transforms.Spectrogram``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
    """


    def __init__(
            self,
            root: str,
            suffix: str = ".wav",
            new_sample_rate: int = None,
            spectrogram_transform: bool = False,
            label_name: str = None
        ):
        
        self._path = root
        self._spectrogram_transform = spectrogram_transform
        self._new_sample_rate = new_sample_rate
        self._label_name = label_name

        walker = walk_files(self._path, suffix=suffix, prefix=True)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label, filename)``
        """
        fileid = self._walker[n]

        waveform, sample_rate, label, filename =  load_audio_item(fileid, self._path, self._label_name)

        if self._new_sample_rate is not None:
            waveform = torchaudio.transforms.Resample(sample_rate, self._new_sample_rate)(waveform)
            sample_rate = self._new_sample_rate
        if self._spectrogram_transform is not None:
           waveform = torchaudio.transforms.Spectrogram()(waveform)

        return waveform, sample_rate, label, filename



    def __len__(self) -> int:
        return len(self._walker)
