from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.datasets.vctk import VCTK
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

    def test_cmuarctic(self):
        data = CMUARCTIC(self.path)
        data[0]
