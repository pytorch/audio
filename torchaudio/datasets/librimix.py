from pathlib import Path
from typing import Union, Tuple, List

import torch
import torchaudio
from torch.utils.data import Dataset

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class LibriMix(Dataset):
    r"""Create the LibriMix dataset.

    Args:
        root (str or Path): The path to the directory where the directory ``Libri2Mix`` or
            ``Libri3Mix`` is stored.
        subset (str, optional): The subset to use. Options: [``train-360`, ``train-100``,
            ``dev``, and ``test``] (Default: ``train-360``).
        num_speakers (int, optional): The number of speakers, which determines the directories
            to traverse. The Dataset will traverse ``s1`` to ``sN`` directories to collect
            N source audios. (Default: 2)
        sample_rate (int, optional): sample rate of audio files. The ``sample_rate`` determines
            which subdirectory the audio are fetched. If any of the audio has a different sample
            rate, raises ``ValueError``. Options: [8000, 16000] (Default: 8000)
        task (str, optional): the task of LibriMix.
            Options: [``enh_single``, ``enh_both``, ``sep_clean``, ``sep_noisy``]
            (Default: ``sep_clean``)

    Note:
        The LibriMix dataset needs to be manually generated. Please check https://github.com/JorisCos/LibriMix
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        task: str = "sep_clean",
    ):
        self.root = Path(root) / f"Libri{num_speakers}Mix"
        if sample_rate == 8000:
            self.root = self.root / "wav8k/min" / subset
        elif sample_rate == 16000:
            self.root = self.root / "wav16k/min" / subset
        else:
            raise ValueError(f"Unsupported sample rate. Found {sample_rate}.")
        self.sample_rate = sample_rate
        self.task = task
        self.mix_dir = (self.root / f"mix_{task.split('_')[1]}").resolve()
        self.src_dirs = [(self.root / f"s{i+1}").resolve() for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob("*wav")]
        self.files.sort()

    def _load_audio(self, path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"The dataset contains audio file of sample rate {sample_rate}, "
                f"but the requested sample rate is {self.sample_rate}."
            )
        return waveform

    def _load_sample(self, filename) -> SampleType:
        mixed = self._load_audio(str(self.mix_dir / filename))
        srcs = []
        for i, dir_ in enumerate(self.src_dirs):
            src = self._load_audio(str(dir_ / filename))
            if mixed.shape != src.shape:
                raise ValueError(f"Different waveform shapes. mixed: {mixed.shape}, src[{i}]: {src.shape}")
            srcs.append(src)
        return self.sample_rate, mixed, srcs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return self._load_sample(self.files[key])
