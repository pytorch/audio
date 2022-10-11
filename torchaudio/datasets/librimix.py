from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]

_TASKS_TO_MIXTURE = {
    "sep_clean": "mix_clean",
    "enh_single": "mix_single",
    "enh_both": "mix_both",
    "sep_noisy": "mix_both",
}


class LibriMix(Dataset):
    r"""*LibriMix* :cite:`cosentino2020librimix` dataset.

    Args:
        root (str or Path): The path to the directory where the directory ``Libri2Mix`` or
            ``Libri3Mix`` is stored.
        subset (str, optional): The subset to use. Options: [``"train-360"``, ``"train-100"``,
            ``"dev"``, and ``"test"``] (Default: ``"train-360"``).
        num_speakers (int, optional): The number of speakers, which determines the directories
            to traverse. The Dataset will traverse ``s1`` to ``sN`` directories to collect
            N source audios. (Default: 2)
        sample_rate (int, optional): Sample rate of audio files. The ``sample_rate`` determines
            which subdirectory the audio are fetched. If any of the audio has a different sample
            rate, raises ``ValueError``. Options: [8000, 16000] (Default: 8000)
        task (str, optional): The task of LibriMix.
            Options: [``"enh_single"``, ``"enh_both"``, ``"sep_clean"``, ``"sep_noisy"``]
            (Default: ``"sep_clean"``)
        mode (str, optional): The mode when creating the mixture. If set to ``"min"``, the lengths of mixture
            and sources are the minimum length of all sources. If set to ``"max"``, the lengths of mixture and
            sources are zero padded to the maximum length of all sources.
            Options: [``"min"``, ``"max"``]
            (Default: ``"min"``)

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
        mode: str = "min",
    ):
        self.root = Path(root) / f"Libri{num_speakers}Mix"
        if mode not in ["max", "min"]:
            raise ValueError(f'Expect ``mode`` to be one in ["min", "max"]. Found {mode}.')
        if sample_rate == 8000:
            self.root = self.root / "wav8k" / mode / subset
        elif sample_rate == 16000:
            self.root = self.root / "wav16k" / mode / subset
        else:
            raise ValueError(f"Unsupported sample rate. Found {sample_rate}.")
        self.sample_rate = sample_rate
        self.task = task
        self.mix_dir = (self.root / _TASKS_TO_MIXTURE[task]).resolve()
        if task == "enh_both":
            self.src_dirs = [(self.root / "mix_clean")]
        else:
            self.src_dirs = [(self.root / f"s{i+1}").resolve() for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob("*.wav")]
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
            Tuple of the following items;

            int:
                Sample rate
            Tensor:
                Mixture waveform
            list of Tensors:
                List of source waveforms
        """
        return self._load_sample(self.files[key])
