import unittest
from pprint import pprint

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO


class TestDatasets(unittest.TestCase):
    def test_yesno(self):
        data = YESNO("./yesnotest")
        pprint("YESNO")
        pprint(next(data))

    def test_vctk(self):
        data = VCTK("./vctktest/")
        pprint("VCTK")
        pprint(next(data))

    def test_librispeech(self):
        data = LIBRISPEECH("./librispeechtest/", "dev-clean")
        pprint("LIBRISPEECH")
        pprint(next(data))

    def test_commonvoice(self):
        data = COMMONVOICE("./commonvoicetest/", "tatar", "train.tsv")
        pprint("COMMONVOICE")
        pprint(next(data))


if __name__ == "__main__":
    unittest.main()
