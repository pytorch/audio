import torch
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    TempDirMixin,
    get_asset_path
)

from torchaudio.datasets import utils as dataset_utils


class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, n):
        sample_rate = 8000
        waveform = n * torch.ones(2, 256)
        return waveform, sample_rate

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class TestIterator(TorchaudioTestCase, TempDirMixin):
    backend = 'default'

    def test_disckcache_iterator(self):
        data = dataset_utils.diskcache_iterator(Dataset(), self.get_base_temp_dir())
        # Save
        data[0]
        # Load
        data[0]

    def test_bg_iterator(self):
        data = dataset_utils.bg_iterator(Dataset(), 5)
        for _ in data:
            pass
