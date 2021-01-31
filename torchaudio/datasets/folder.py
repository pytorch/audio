import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import walk_files

def load_audio_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
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
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Optional[Callable] = None,
            is_valid_file: Optional[Callable] = None
        ):
        
        self._path = root
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
        return load_audio_item(fileid, self._path)
        

        #path, target = self.samples[index]
        #sample = self.loader(path)
        #if self.transform is not None:
        #    sample = self.transform(sample)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        #return sample, target



    def __len__(self) -> int:
        return len(self._walker)
