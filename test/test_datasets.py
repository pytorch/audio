import unittest
from pprint import pprint

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO
from torchaudio.datasets.utils import DiskCache


class TestDatasets(unittest.TestCase):
    def test_yesno(self):
        data = YESNO("./yesnotest", return_dict=True)
        data[0]

    def test_vctk(self):
        data = VCTK("./vctktest/", return_dict=True)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH("./librispeechtest/", "dev-clean")
        data[0]

    def test_commonvoice(self):
        data = COMMONVOICE("./commonvoicetest/", "train.tsv", "tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        data = COMMONVOICE("./commonvoicetest/", "train.tsv", "tatar")
        data = DiskCache(data)
        # Save
        data[0]
        # Load
        data[0]


if __name__ == "__main__":
    unittest.main()
