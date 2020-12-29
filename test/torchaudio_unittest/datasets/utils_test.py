import torch
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    TempDirMixin,
    get_asset_path
)

from torchaudio.datasets import utils as dataset_utils


class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, n):
        return torch.randn(32, 2, 256), 2

    def __len__(self) -> int:
        return 32

    def __iter__(self):
        return iter(range(32))


class TestIterator(TorchaudioTestCase, TempDirMixin):
    @classmethod
    def setUpClass(cls):
        cls.location = cls.get_base_temp_dir()
        cls.dataset = Dataset()

    backend = 'default'

    def test_disckcache_iterator(self):
        data = dataset_utils.diskcache_iterator(self.dataset, location=self.location)
        # Save
        data[0]
        # Load
        data[0]

    def test_bg_iterator(self):
        data = dataset_utils.bg_iterator(self.dataset, 5)
        for _ in data:
            pass
