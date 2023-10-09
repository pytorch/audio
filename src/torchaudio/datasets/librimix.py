import os
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform

_TASKS_TO_MIXTURE = {
    "sep_clean": "mix_clean",
    "enh_single": "mix_single",
    "enh_both": "mix_both",
    "sep_noisy": "mix_both",
}


class LibriMix(Dataset):
    r"""*LibriMix* :cite:`cosentino2020librimix` dataset.

    Args:
        root (str or Path): The path where the directory ``Libri2Mix`` or
            ``Libri3Mix`` is stored. Not the path of those directories.
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
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"The path {self.root} doesn't exist. "
                "Please check the ``root`` path and ``num_speakers`` or download the dataset manually."
            )
        if mode not in ["max", "min"]:
            raise ValueError(f'Expect ``mode`` to be one in ["min", "max"]. Found {mode}.')
        if sample_rate == 8000:
            mix_dir = self.root / "wav8k" / mode / subset
        elif sample_rate == 16000:
            mix_dir = self.root / "wav16k" / mode / subset
        else:
            raise ValueError(f"Unsupported sample rate. Found {sample_rate}.")
        self.sample_rate = sample_rate
        self.task = task

        self.mix_dir = mix_dir / _TASKS_TO_MIXTURE[task]
        if task == "enh_both":
            self.src_dirs = [(mix_dir / "mix_clean")]
        else:
            self.src_dirs = [(mix_dir / f"s{i+1}") for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob("*.wav")]
        self.files.sort()

    def _load_sample(self, key) -> Tuple[int, torch.Tensor, List[torch.Tensor]]:
        metadata = self.get_metadata(key)
        mixed = _load_waveform(self.root, metadata[1], metadata[0])
        srcs = []
        for i, path_ in enumerate(metadata[2]):
            src = _load_waveform(self.root, path_, metadata[0])
            if mixed.shape != src.shape:
                raise ValueError(f"Different waveform shapes. mixed: {mixed.shape}, src[{i}]: {src.shape}")
            srcs.append(src)
        return self.sample_rate, mixed, srcs

    def get_metadata(self, key: int) -> Tuple[int, str, List[str]]:
        """Get metadata for the n-th sample from the dataset.

        Args:
            key (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            int:
                Sample rate
            str:
                Path to mixed audio
            List of str:
                List of paths to source audios
        """
        filename = self.files[key]
        mixed_path = os.path.relpath(self.mix_dir / filename, self.root)
        srcs_paths = []
        for dir_ in self.src_dirs:
            src = os.path.relpath(dir_ / filename, self.root)
            srcs_paths.append(src)
        return self.sample_rate, mixed_path, srcs_paths

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> Tuple[int, torch.Tensor, List[torch.Tensor]]:
        """Load the n-th sample from the dataset.

        Args:
            key (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            int:
                Sample rate
            Tensor:
                Mixture waveform
            List of Tensors:
                List of source waveforms
        """
        return self._load_sample(key)
