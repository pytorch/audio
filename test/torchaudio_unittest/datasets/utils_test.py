from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    get_asset_path,
)

from torchaudio.datasets import utils as dataset_utils
from torchaudio.datasets.commonvoice import COMMONVOICE

original_ext_audio = COMMONVOICE._ext_audio


class TestIterator(TorchaudioTestCase):
    @classmethod
    def setUpClass(cls):
        COMMONVOICE._ext_audio = ".wav"

    @classmethod
    def tearDownClass(cls):
        COMMONVOICE._ext_audio = original_ext_audio

    backend = 'default'
    path = get_asset_path('CommonVoice', 'cv-corpus-4-2019-12-10', 'tt')

    def test_disckcache_iterator(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = dataset_utils.diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_bg_iterator(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = dataset_utils.bg_iterator(data, 5)
        for _ in data:
            pass
