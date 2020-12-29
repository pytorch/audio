import torch
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    TempDirMixin,
    get_asset_path
)

from torchaudio.datasets import utils as dataset_utils
from torchaudio.datasets.commonvoice import COMMONVOICE

original_ext_audio = COMMONVOICE._ext_audio


class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, n):
        return torch.randn(32, 2, 256)

    def __len__(self) -> int:
        return 32


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
        data = dataset_utils.bg_iterator(self.dataset, 2)
        for _ in data:
            pass
