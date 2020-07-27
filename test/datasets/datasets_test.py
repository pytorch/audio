from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.utils import diskcache_iterator, bg_iterator
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.ljspeech import LJSPEECH
from torchaudio.datasets.cmuarctic import CMUARCTIC

from ..common_utils import (
    TorchaudioTestCase,
    get_asset_path,
)


class TestDatasets(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_vctk(self):
        data = VCTK(self.path)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_ljspeech(self):
        data = LJSPEECH(self.path)
        data[0]

    def test_cmuarctic(self):
        data = CMUARCTIC(self.path)
        data[0]


class TestCommonVoice(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_commonvoice(self):
        data = COMMONVOICE(self.path, url="tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_commonvoice_bg(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = bg_iterator(data, 5)
        for _ in data:
            pass
