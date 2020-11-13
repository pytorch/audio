import sys
import platform
from unittest import skipIf
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torchaudio

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExtension,
    get_whitenoise,
    save_wav,
)


class RandomPerturbationFile(torch.utils.data.Dataset):
    """Given flist, apply random speed perturbation"""
    def __init__(self, flist: List[str], sample_rate: int):
        super().__init__()
        self.flist = flist
        self.sample_rate = sample_rate
        self.rng = None

    def __getitem__(self, index):
        speed = self.rng.uniform(0.5, 2.0)
        effects = [
            ['gain', '-n', '-10'],
            ['speed', f'{speed:.5f}'],  # duration of data is 0.5 ~ 2.0 seconds.
            ['rate', f'{self.sample_rate}'],
            ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
            ['trim', '0', '2'],  # get the first 2 seconds
        ]
        data, _ = torchaudio.sox_effects.apply_effects_file(self.flist[index], effects)
        return data

    def __len__(self):
        return len(self.flist)


class RandomPerturbationTensor(torch.utils.data.Dataset):
    """Apply speed purturbation to (synthetic) Tensor data"""
    def __init__(self, signals: List[Tuple[torch.Tensor, int]], sample_rate: int):
        super().__init__()
        self.signals = signals
        self.sample_rate = sample_rate
        self.rng = None

    def __getitem__(self, index):
        speed = self.rng.uniform(0.5, 2.0)
        effects = [
            ['gain', '-n', '-10'],
            ['speed', f'{speed:.5f}'],  # duration of data is 0.5 ~ 2.0 seconds.
            ['rate', f'{self.sample_rate}'],
            ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
            ['trim', '0', '2'],  # get the first 2 seconds
        ]
        tensor, sample_rate = self.signals[index]
        data, _ = torchaudio.sox_effects.apply_effects_tensor(tensor, sample_rate, effects)
        return data

    def __len__(self):
        return len(self.signals)


def init_random_seed(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.rng = np.random.RandomState(worker_id)


@skipIfNoExtension
@skipIf(
    platform.system() == 'Darwin' and
    sys.version_info.major == 3 and
    sys.version_info.minor in [6, 7],
    'This test is known to get stuck for macOS with Python < 3.8. '
    'See https://github.com/pytorch/pytorch/issues/46409'
)
class TestSoxEffectsDataset(TempDirMixin, PytorchTestCase):
    """Test `apply_effects_file` in multi-process dataloader setting"""

    def _generate_dataset(self, num_samples=128):
        flist = []
        for i in range(num_samples):
            sample_rate = np.random.choice([8000, 16000, 44100])
            dtype = np.random.choice(['float32', 'int32', 'int16', 'uint8'])
            data = get_whitenoise(n_channels=2, sample_rate=sample_rate, duration=1, dtype=dtype)
            path = self.get_temp_path(f'{i:03d}_{dtype}_{sample_rate}.wav')
            save_wav(path, data, sample_rate)
            flist.append(path)
        return flist

    def test_apply_effects_file(self):
        sample_rate = 12000
        flist = self._generate_dataset()
        dataset = RandomPerturbationFile(flist, sample_rate)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, num_workers=16,
            worker_init_fn=init_random_seed,
        )
        for batch in loader:
            assert batch.shape == (32, 2, 2 * sample_rate)

    def _generate_signals(self, num_samples=128):
        signals = []
        for _ in range(num_samples):
            sample_rate = np.random.choice([8000, 16000, 44100])
            data = get_whitenoise(
                n_channels=2, sample_rate=sample_rate, duration=1, dtype='float32')
            signals.append((data, sample_rate))
        return signals

    def test_apply_effects_tensor(self):
        sample_rate = 12000
        signals = self._generate_signals()
        dataset = RandomPerturbationTensor(signals, sample_rate)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, num_workers=16,
            worker_init_fn=init_random_seed,
        )
        for batch in loader:
            assert batch.shape == (32, 2, 2 * sample_rate)


def speed(path):
    wav, sample_rate = torchaudio.load(path)
    effects = [
        ['speed', '1.03756523535464655'],
        ['rate', f'{sample_rate}'],
    ]
    return torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)[0]


@skipIfNoExtension
class TestProcessPoolExecutor(TempDirMixin, PytorchTestCase):
    backend = "sox_io"

    def setUp(self):
        sample_rate = 16000
        self.flist = []
        for i in range(10):
            path = self.get_temp_path(f'{i}.wav')
            data = get_whitenoise(n_channels=1, sample_rate=sample_rate, duration=1, dtype='float')
            save_wav(path, data, sample_rate)
            self.flist.append(path)

    def test_executor(self):
        """Test that apply_effects_tensor with speed + rate does not crush

        https://github.com/pytorch/audio/issues/1021
        """
        executor = ProcessPoolExecutor(1)
        futures = [executor.submit(speed, path) for path in self.flist]
        for future in futures:
            future.result()
