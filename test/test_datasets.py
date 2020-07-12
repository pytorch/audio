import unittest

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.datasets.utils import diskcache_iterator, bg_iterator
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO
from torchaudio.datasets.ljspeech import LJSPEECH
from torchaudio.datasets.gtzan import GTZAN
from torchaudio.datasets.cmuarctic import CMUARCTIC

from . import common_utils


class TestDatasets(common_utils.TorchaudioTestCase):
    backend = 'default'
    path = common_utils.get_asset_path()

    def test_yesno(self):
        data = YESNO(self.path)
        data[0]

    def test_vctk(self):
        data = VCTK(self.path)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_ljspeech(self):
        data = LJSPEECH(self.path)
        data[0]

    def test_speechcommands(self):
        data = SPEECHCOMMANDS(self.path)
        data[0]

    def test_gtzan(self):
        data = GTZAN(self.path)
        data[0]

    def test_cmuarctic(self):
        data = CMUARCTIC(self.path)
        data[0]


class TestCommonVoice(common_utils.TorchaudioTestCase):
    path = common_utils.get_asset_path()

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


if __name__ == "__main__":
    unittest.main()
