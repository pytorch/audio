import unittest
from pprint import pprint

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO


class TestDatasets(unittest.TestCase):
    def test_yesno(self):
        data = YESNO("./yesnotest")
        data[0]

    def test_vctk(self):
        data = VCTK("./vctktest/")
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH("./librispeechtest/", "dev-clean")
        data[0]

    def test_commonvoice(self):
        data = COMMONVOICE("./commonvoicetest/", "train.tsv", "tatar")
        data[0]


if __name__ == "__main__":
    unittest.main()
