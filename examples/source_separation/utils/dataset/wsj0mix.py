from pathlib import Path
from typing import Union, Tuple, List

import torch
from torch.utils.data import Dataset

import torchaudio

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class WSJ0Mix(Dataset):
    """Create a Dataset for wsj0-mix.

    Args:
        root (str or Path): Path to the directory where the dataset is found.
        num_speakers (int): The number of speakers, which determines the directories
            to traverse. The Dataset will traverse ``s1`` to ``sN`` directories to collect
            N source audios.
        sample_rate (int): Expected sample rate of audio files. If any of the audio has a
            different sample rate, raises ``ValueError``.
        audio_ext (str): The extension of audio files to find. (default: ".wav")
    """
    def __init__(
        self,
        root: Union[str, Path],
        num_speakers: int,
        sample_rate: int,
        audio_ext: str = ".wav",
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.mix_dir = (self.root / "mix").resolve()
        self.src_dirs = [(self.root / f"s{i+1}").resolve() for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob(f"*{audio_ext}")]
        self.files.sort()

    def _load_audio(self, path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"The dataset contains audio file of sample rate {sample_rate}. "
                "Where the requested sample rate is {self.sample_rate}."
            )
        return waveform

    def _load_sample(self, filename) -> SampleType:
        mixed = self._load_audio(str(self.mix_dir / filename))
        srcs = []
        for i, dir_ in enumerate(self.src_dirs):
            src = self._load_audio(str(dir_ / filename))
            if mixed.shape != src.shape:
                raise ValueError(
                    f"Different waveform shapes. mixed: {mixed.shape}, src[{i}]: {src.shape}"
                )
            srcs.append(src)
        return self.sample_rate, mixed, srcs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            tuple: ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return self._load_sample(self.files[key])
