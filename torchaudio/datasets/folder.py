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
        new_sample_rate (int) : Resample audio to new sample rate specified.
        spectrogram_transform (bool): If True transform the audio waveform and returns it  
        transformed into a spectrogram tensor.
        label_name (str): The label is pulled from the folders in the path, this allows you to statically
        define the label string.
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
            n (int): The index of the file to be loaded

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
